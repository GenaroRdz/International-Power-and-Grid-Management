from machine import Pin, SPI
from mcp2515 import MCP2515
import setup
import INA226

# CAN bus objects/helpers live in setup.py (that's where the bus is created).
from setup import can, CAN_OK, Current_Device, ADDR, ADDR_NAMES, id_of, name_of

# ── Outputs ───────────────────────────────────────────────────────────────────
OUTPUTS = {
    "ECU1_BAT": Pin(setup.ECU1_BAT, Pin.OUT),
    "ECU1_ACC": Pin(setup.ECU1_ACC, Pin.OUT),
    "ECU1_IGN": Pin(setup.ECU1_IGN, Pin.OUT),
    "ECU2_BAT": Pin(setup.ECU2_BAT, Pin.OUT),
    "ECU2_ACC": Pin(setup.ECU2_ACC, Pin.OUT),
    "ECU2_IGN": Pin(setup.ECU2_IGN, Pin.OUT),
    "ECU3_BAT": Pin(setup.ECU3_BAT, Pin.OUT),
    "ECU3_ACC": Pin(setup.ECU3_ACC, Pin.OUT),
    "ECU3_IGN": Pin(setup.ECU3_IGN, Pin.OUT),
    "ECU4_BAT": Pin(setup.ECU4_BAT, Pin.OUT),
    "ECU4_ACC": Pin(setup.ECU4_ACC, Pin.OUT),
    "ECU4_IGN": Pin(setup.ECU4_IGN, Pin.OUT),
}

initialized = False
streaming = False   # True while the INA226 readings are being streamed out

# ── CAN messaging ───────────────────────────────────────────────────────────────
# A CAN frame here = 1 target byte + short text  (e.g. "BAT ON").
# WHO sent it travels inside the CAN id, so the receiver gets both From and To.
# Keep the text 7 characters or fewer (1 of the 8 payload bytes is the target).

def can_send(target, text):
    """Relay a short text command to another device over CAN.
    Returns True on success, False if CAN is down or the send fails.
    Never raises -- a CAN problem must not break the command loop."""
    if not CAN_OK:
        return False
    try:
        payload = bytes([ADDR[target]]) + text.encode()
        can.send_message(id_of(Current_Device), payload)
        return True
    except Exception:
        return False


def can_receive():
    """Non-blocking. Return (sender, target, text) for one waiting frame,
    or None if nothing is pending."""
    if not CAN_OK:
        return None
    if can.check_receive():
        msg_id, data = can.read_message()
        if data and len(data) >= 1:
            sender = name_of(msg_id)
            target = ADDR_NAMES.get(data[0], "?")
            text   = bytes(data[1:]).decode()
            return sender, target, text
    return None


def can_selftest():
    """Loopback self-test: proves the MCP2515 + SPI on THIS board work,
    without needing the bus wiring or the other ECU."""
    if not CAN_OK:
        return ("CANTEST: FAIL - CAN never initialized (CAN_OK=False). "
                "Fix SPI_CS pin / crystal / SPI wiring in setup.py.")
    try:
        import time
        can.set_loopback_mode()
        can.send_message(id_of(Current_Device), b"\x01BAT ON")
        time.sleep_ms(50)
        msg_id, data = can.read_message()
        can.set_normal_mode()
        if msg_id is None:
            return "CANTEST: FAIL - sent but nothing looped back (check CS / wiring)."
        return "CANTEST: PASS - chip+SPI OK. Got id=0x{:03X} data={}".format(
            msg_id, bytes(data))
    except Exception as e:
        try:
            can.set_normal_mode()
        except Exception:
            pass
        return "CANTEST: ERROR - {}".format(e)


# ── Command handlers ──────────────────────────────────────────────────────────

def handle_ping(cmd):
    global initialized
    if cmd == "ping":
        if initialized:
            print("pong_init")
        else:
            print("pong_noinit")


def commands(cmd):
    global initialized, streaming
    cmd = cmd.strip().upper()
    if not cmd:
        return

    if cmd == "PING":      # already answered by handle_ping(); don't echo an error
        return

    if cmd == "A":
        print(initialized)
    elif cmd == "*INIT":
        initialized = True
        INA226.init_INAs()      # missing/faulty sensors are non-fatal
        print("Initialized")
    elif cmd == "*RST":
        reset()
    elif cmd == "*IDN?":
        print("{},{},{},{}".format(setup.FABRICANTE, setup.MODELO,
                                   setup.SERIAL, setup.VERSION))

    # ── CAN diagnostics ─────────────────────────────────────────────────────────
    elif cmd == "CANTEST":
        print(can_selftest())
    elif cmd == "CAN?":
        print("CAN_OK = {}".format(CAN_OK))

    # ── INA226 commands ─────────────────────────────────────────────────────────
    elif cmd == "INA_START":
        if not initialized:
            print("ERROR: System not initialized, send *INIT first")
            return
        streaming = True
        print("INA streaming ON")

    elif cmd == "INA_STOP":
        streaming = False
        print("INA streaming OFF")

    elif cmd == "INA?":
        if not initialized:
            print("ERROR: System not initialized, send *INIT first")
            return
        print(INA226.read_all_INAs_str())

    # ── Output control ────────────────────────────────────────────────────────
    else:
        if cmd == "ON ALL":
            if not initialized:
                print("ERROR: System not initialized, send *INIT first")
                return
            for pin in OUTPUTS.values():
                pin.on()
        elif cmd == "OFF ALL":
            if not initialized:
                print("ERROR: System not initialized, send *INIT first")
                return
            for pin in OUTPUTS.values():
                pin.off()
        else:
            parts = cmd.split()
            if len(parts) == 2 and parts[1] in ("ON", "OFF"):
                name = parts[0]
                if name not in OUTPUTS:
                    print("ERROR: Output {} not available".format(name))
                    return
                if not initialized:
                    print("ERROR: System not initialized, send *INIT first")
                    return
                # Local GPIO on the MAIN board.
                OUTPUTS[name].on() if parts[1] == "ON" else OUTPUTS[name].off()
                # Relay the same action to the matching ECU over CAN.
                # "ECU1_BAT" -> device "ECU1", peripheral "BAT" -> text "BAT ON".
                bits = name.split("_")          # split is always available; partition isn't
                if len(bits) == 2 and bits[0] in ADDR:
                    dev, periph = bits[0], bits[1]
                    ok = can_send(dev, "{} {}".format(periph, parts[1]))
                    # DEBUG: shows whether the frame went out. Remove once confirmed.
                    print("CAN-> {} : {} {}  [{}]".format(
                        dev, periph, parts[1], "sent" if ok else "FAILED"))
            else:
                print("ERROR: Unknown command: {}".format(cmd))


def reset():
    global initialized, streaming
    initialized = False
    streaming = False
    for pin in OUTPUTS.values():
        pin.off()

