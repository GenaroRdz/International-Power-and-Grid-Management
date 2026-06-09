# Device identification strings for *IDN? SCPI command
FABRICANTE = "International"
MODELO = "ESP32"
SERIAL = "0001"
VERSION = "v1.0"

# GPIO pin assignments for output control
ECU1_BAT = 25 
ECU1_ACC = 33
ECU1_IGN = 32


ECU2_BAT = 13  
ECU2_ACC = 17
ECU2_IGN = 5 


ECU3_BAT = 2
ECU3_ACC = 4
ECU3_IGN = 16


ECU4_BAT = 14 
ECU4_ACC = 27
ECU4_IGN = 26


I2C_SDA = 21
I2C_SCL = 22

SPI_SCK = 18
SPI_MOSI = 23
SPI_MISO = 19
SPI_CS = 15


# was 35 -- GPIO35 is INPUT-ONLY on the ESP32 and CANNOT drive a
              # chip-select (CS must be an output). 15 is free and output-capable.
              # Change it to match the pin your MCP2515 CS is actually wired to.


# ── CAN bus (MCP2515) ───────────────────────────────────────────────────────────
# Moved here from the old can_setup.py so the whole project shares one config.
from machine import Pin, SPI
from mcp2515 import MCP2515

# This board's role on the CAN bus. The MAIN talks to ECU1..ECU4.
Current_Device = "MAIN"

# 11-bit CAN id for every device (0x000-0x7FF). Lower id = higher priority,
# so MAIN is kept lowest to win the bus. Edit freely, just keep them different.
# The CAN id says WHO SENT a frame (it travels in the frame header).
IDS = {
    "MAIN": 0x100,
    "ECU1": 0x101,
    "ECU2": 0x102,
    "ECU3": 0x103,
    "ECU4": 0x104,
}

# 1-byte TARGET address that goes in payload byte 0 of every CAN frame (it says
# WHO the frame is for). This is written out EXPLICITLY on purpose:
#
#   * It must NOT depend on dict insertion order. Deriving it with
#     enumerate(IDS) works only as long as the order is perfect; one reordered
#     line silently shuffles every address and the GUI shows the wrong routes
#     (e.g. pressing ECU1 appears as ECU2). Writing it out makes that
#     impossible and lets you eyeball it at a glance.
#   * It MUST match the table the GUI CAN monitor uses, which is exactly:
#         MAIN = 0, ECU1 = 1, ECU2 = 2, ECU3 = 3, ECU4 = 4
#     Keep these identical on every board (MAIN and all ECUs).
ADDR = {
    "MAIN": 0,
    "ECU1": 1,
    "ECU2": 2,
    "ECU3": 3,
    "ECU4": 4,
}

NAMES      = {can_id: name for name, can_id in IDS.items()}   # CAN id -> name
ADDR_NAMES = {addr: name for name, addr in ADDR.items()}      # addr   -> name


def id_of(name):       # "ECU1" -> 0x101
    return IDS[name]


def name_of(can_id):   # 0x101 -> "ECU1"
    return NAMES.get(can_id, "unknown")

try:
    can_spi = SPI(1, baudrate=10_000_000, polarity=0, phase=0,
                  sck=Pin(SPI_SCK), mosi=Pin(SPI_MOSI), miso=Pin(SPI_MISO))
    can_cs  = Pin(SPI_CS, Pin.OUT)
    can = MCP2515(can_spi, can_cs, crystal=8)   # crystal MUST match your module (8 or 16)
    can.reset()
    can.set_bitrate(500)        
    can.set_normal_mode()
    CAN_OK = True
except Exception as e:
    can = None
    CAN_OK = False
    print("CAN module not connected properly:", e)
