from machine import SPI, Pin
from mcp2515 import MCP2515

# 1. Initialize SPI Bus (VSPI default pins)
# Baudrate for SPI to MCP2515 can be up to 10 MHz
spi = SPI(2, baudrate=10000000, polarity=0, phase=0, 
          sck=Pin(18), mosi=Pin(23), miso=Pin(19))

# 2. Initialize MCP2515
# cs=5 is the Chip Select pin; crystal=8 (usually 8MHz or 16MHz)
can = MCP2515(spi, cs=Pin(5), crystal=8) 
can.reset()
can.set_bitrate(500) # Set CAN bus speed to 500kbps
can.set_normal_mode()

# 3. Send a CAN Message
# ID=0x123, Data=[1, 2, 3, 4, 5, 6, 7, 8]
can.send_message(0x123, [1, 2, 3, 4, 5, 6, 7, 8])
print("Message sent!")

# 4. Receive a CAN Message
if can.check_receive():
    msg_id, msg_data = can.read_message()
    print(f"Received ID: {hex(msg_id)} Data: {msg_data}")

