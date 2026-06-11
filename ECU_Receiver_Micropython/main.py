# main.py  --  ECU board.  (In canbus.py set  ME = "ECU1"  (or ECU2, ECU3, ECU4))

import time
from functions import send_message, receive_message
from can_setup import Current_Device

print(Current_Device, "ready (ecu)")

while True:
    msg = receive_message()
    if msg:
        sender, target, text = msg

        if target == Current_Device:                       # this command is for ME
            print("got:", text, "from", sender)

            send_message(sender, "OK")
            
        # else: it's for another ECU -- ignore it

    time.sleep_ms(10)
