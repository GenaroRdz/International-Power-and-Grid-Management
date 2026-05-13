from machine import Pin
import setup
import INA226

# Initialize output pins using GPIO numbers defined in setup
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

def commands(cmd):
    global initialized
    cmd = cmd.strip().upper()

    if not cmd:
        return

    if cmd == "A":
        print(initialized)

    elif cmd == "*INIT":
        initialized = True
        print("Initialized")

    elif cmd == "*RST":
        reset()

    elif cmd == "*IDN?":
        print(f"{setup.FABRICANTE},{setup.MODELO},{setup.SERIAL},{setup.VERSION}")

    elif cmd == "ON ALL":
        if not initialized:
            print("ERROR: System not initialized, send *INIT first")
            return
        for pin in OUTPUTS.values():
            pin.on()
        print("All outputs ON")

    elif cmd == "OFF ALL":
        if not initialized:
            print("ERROR: System not initialized, send *INIT first")
            return
        for pin in OUTPUTS.values():
            pin.off()
        print("All outputs OFF")

    elif cmd == "READ INA":
        print(INA226.lectura_INA())

    else:
        parts = cmd.split()
        if len(parts) == 2 and parts[0] in OUTPUTS and parts[1] in ("ON", "OFF"):
            if not initialized:
                print("ERROR: System not initialized, send *INIT first")
                return
            if parts[1] == "ON":
                OUTPUTS[parts[0]].on()
            else:
                OUTPUTS[parts[0]].off()
        else:
            print(f"ERROR: Unknown command: {cmd}")

def reset():
    global initialized
    initialized = False
    for pin in OUTPUTS.values():
        pin.off()