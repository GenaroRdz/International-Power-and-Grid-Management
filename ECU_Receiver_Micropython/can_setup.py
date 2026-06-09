
from machine import Pin, SPI
from mcp2515 import MCP2515

Current_Device = "ECU1"        # <-- change to ECU2 / ECU3 / ECU4 on each board
bit_rate = 500                 # MUST match every other board on the bus


IDS = {
    "MAIN": 0x100,
    "ECU1": 0x101,
    "ECU2": 0x102,
    "ECU3": 0x103,
    "ECU4": 0x104,
}

NAMES = {can_id: name for name, can_id in IDS.items()}

ADDR = {
    "MAIN": 0,
    "ECU1": 1,
    "ECU2": 2,
    "ECU3": 3,
    "ECU4": 4,
}
ADDR_NAMES = {addr: name for name, addr in ADDR.items()}   # addr -> name (by value, safe)


def id_of(name):       # device name -> CAN id   ("ECU1" -> 0x101)
    return IDS[name]

def name_of(can_id):   # CAN id -> device name   (0x101 -> "ECU1")
    return NAMES.get(can_id, "unknown")


# ===== Hardware: makes 'can' =====
spi = SPI(1, baudrate=10_000_000, polarity=0, phase=0)
cs  = Pin(5, Pin.OUT)

can = MCP2515(spi, cs, crystal=8)   # crystal MUST match your module (8 or 16)

try:
    can.reset()
    can.set_bitrate(bit_rate)        # SAME value on every board
    can.set_normal_mode()
    CAN_OK = True
except Exception as e:
    CAN_OK = False
    print("CAN module not connected properly:", e)