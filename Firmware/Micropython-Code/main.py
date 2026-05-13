import sys
import time
import setup
import functions
from INA226 import lectura_INA

# Reset all outputs to their default state before starting
functions.reset()
print("Sistema listo")


while True:
    try:
        cmd = sys.stdin.readline()
        functions.commands(cmd)
    except Exception as e:
        print("Error:", e)   # ← muestra el error real
    time.sleep(0.05)
