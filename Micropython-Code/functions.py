from machine import Pin
import setup

# Initialize output pins using GPIO numbers defined in setup
output1 = Pin(setup.PIN_OUT1, Pin.OUT)
output2 = Pin(setup.PIN_OUT2, Pin.OUT)

# Tracks whether *INIT has been received before allowing output control
initialized = False

def commands(cmd):
    global initialized
    cmd = cmd.strip().upper()

    # Debug command: prints current initialization state
    if cmd == "A":
        print(initialized)

    # *INIT: enables output control
    elif cmd == "*INIT":
        initialized = True
        print("Initialized")

    # *RST: resets system to default state
    elif cmd == "*RST":
        reset()

    # *IDN?: returns device identification string
    elif cmd == "*IDN?":
        print(f"{setup.FABRICANTE},{setup.MODELO},{setup.SERIAL},{setup.VERSION}")

    # Output commands: only allowed after *INIT
    elif cmd in ("OUT1 ON", "OUT1 OFF", "OUT2 ON", "OUT2 OFF"):
        if not initialized:
            print("ERROR: System not initialized, send *INIT first")
        else:
            if cmd == "OUT1 ON":
                output1.on()
            elif cmd == "OUT1 OFF":
                output1.off()
            elif cmd == "OUT2 ON":
                output2.on()
            elif cmd == "OUT2 OFF":
                output2.off()

def reset():
    # Turn off all outputs and clear initialization flag
    global initialized
    initialized = False
    output1.off()
    output2.off()