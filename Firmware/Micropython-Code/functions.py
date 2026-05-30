from machine import Pin, SPI
from mcp2515 import MCP2515
import setup
import INA226

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
        INA226.init_INAs()      # set up the four sensors once
        print("Initialized")
    elif cmd == "*RST":
        reset()
    elif cmd == "*IDN?":
        print(f"{setup.FABRICANTE},{setup.MODELO},{setup.SERIAL},{setup.VERSION}")


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

    elif cmd == "INA?":          # single reading on demand (no streaming needed)
        if not initialized:
            print("ERROR: System not initialized, send *INIT first")
            return
        print(INA226.read_all_INAs_str())



    # ── CAN commands ──────────────────────────────────────────────────────────
    elif cmd == "CAN_READ":
        if not initialized:
            print("ERROR: System not initialized, send *INIT first")
            return
        can_read()

    elif cmd.startswith("CAN_SEND"):
        if not initialized:
            print("ERROR: System not initialized, send *INIT first")
            return
        parts = cmd.split()
        can_send(parts[1:])   # everything after "CAN_SEND"

    elif cmd == "CAN_STATUS":
        if not initialized:
            print("ERROR: System not initialized, send *INIT first")
            return
        can_status()



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
                    print(f"ERROR: Output {name} not available "
                          f"(valid: BAT1-4, ACC1-4, IGN1-4)")
                    return
                if not initialized:
                    print("ERROR: System not initialized, send *INIT first")
                    return
                OUTPUTS[name].on() if parts[1] == "ON" else OUTPUTS[name].off()
            else:
                print(f"ERROR: Unknown command: {cmd}")



def reset():
    global initialized, streaming
    initialized = False
    streaming = False
    for pin in OUTPUTS.values():
        pin.off()
