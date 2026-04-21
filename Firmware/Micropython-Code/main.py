import sys
import time
import setup
import functions

# Reset all outputs to their default state before starting
functions.reset()
print("Sistema listo")

while True:
    try:
        # Read incoming command from serial and process it
        cmd = sys.stdin.readline()
        functions.commands(cmd)
    except Exception as e:
        # Show the actual error to help with debugging
        print("Commando no encontrado")
    # Small delay to prevent CPU overload without affecting response time
    time.sleep(0.05)
