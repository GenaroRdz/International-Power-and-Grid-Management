from machine import Pin
import setup

# Initialize output pins using GPIO numbers defined in setup
OUTPUTS = {
    1:  Pin(setup.PIN_OUT1,  Pin.OUT),
    2:  Pin(setup.PIN_OUT2,  Pin.OUT),
    3:  Pin(setup.PIN_OUT3,  Pin.OUT),
    4:  Pin(setup.PIN_OUT4,  Pin.OUT),
    5:  Pin(setup.PIN_OUT5,  Pin.OUT),
    6:  Pin(setup.PIN_OUT6,  Pin.OUT),
    7:  Pin(setup.PIN_OUT7,  Pin.OUT),
    8:  Pin(setup.PIN_OUT8,  Pin.OUT),
    9:  Pin(setup.PIN_OUT9,  Pin.OUT),
    10: Pin(setup.PIN_OUT10, Pin.OUT),
    11: Pin(setup.PIN_OUT11, Pin.OUT),
    12: Pin(setup.PIN_OUT12, Pin.OUT),
}
# Tracks whether *INIT has been received before allowing output control
initialized = False

def commands(cmd):
    global initialized
    cmd = cmd.strip().upper()
    
        # Ignore empty lines
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
    # Turn all outputs ON or OFF
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
                
        # OUTn ON / OUTn OFF
        else:
            parts = cmd.split()
            if len(parts) == 2 and parts[0].startswith("OUT") and parts[1] in ("ON", "OFF"):
                try:
                    n = int(parts[0][3:])
                except ValueError:
                    print(f"ERROR: Unknown command: {cmd}")
                    return

                if n not in OUTPUTS:
                    print(f"ERROR: Output {n} not available (valid: 1-12)")
                    return

                if not initialized:
                    print("ERROR: System not initialized, send *INIT first")
                    return

                OUTPUTS[n].on() if parts[1] == "ON" else OUTPUTS[n].off()

            else:
                print(f"ERROR: Unknown command: {cmd}")


def reset():
    # Turn off all outputs and clear initialization flag
    global initialized
    initialized = False
    for pin in OUTPUTS.values():
        pin.off()