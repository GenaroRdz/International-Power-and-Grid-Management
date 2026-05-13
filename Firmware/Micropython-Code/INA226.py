from machine import I2C, Pin
import struct

# --- INA226 register map ---
REG_CONFIG      = 0x00
REG_SHUNT_VOLT  = 0x01
REG_BUS_VOLT    = 0x02
REG_POWER       = 0x03
REG_CURRENT     = 0x04
REG_CALIBRATION = 0x05
REG_MANUF_ID    = 0xFE
REG_DIE_ID      = 0xFF

INA226_ADDR    = 0x40
R_SHUNT        = 0.1
MAX_EXPECTED_A = 0.8

class INA226:
    def __init__(self, i2c, addr=INA226_ADDR, r_shunt=R_SHUNT, i_max=MAX_EXPECTED_A):
        self.i2c = i2c
        self.addr = addr
        self.r_shunt = r_shunt
        self.current_lsb = i_max / 32768.0
        cal = int(0.00512 / (self.current_lsb * self.r_shunt))
        self._w16(REG_CALIBRATION, cal)
        self._w16(REG_CONFIG, 0x4527)

    def _w16(self, reg, val):
        self.i2c.writeto_mem(self.addr, reg, struct.pack(">H", val & 0xFFFF))

    def _r_u16(self, reg):
        return struct.unpack(">H", self.i2c.readfrom_mem(self.addr, reg, 2))[0]

    def _r_s16(self, reg):
        return struct.unpack(">h", self.i2c.readfrom_mem(self.addr, reg, 2))[0]

    def manufacturer_id(self): return self._r_u16(REG_MANUF_ID)
    def die_id(self):          return self._r_u16(REG_DIE_ID)
    def shunt_voltage(self):   return self._r_s16(REG_SHUNT_VOLT) * 2.5e-6
    def bus_voltage(self):     return self._r_u16(REG_BUS_VOLT)   * 1.25e-3
    def current(self):         return self._r_s16(REG_CURRENT)    * self.current_lsb
    def power(self):           return self._r_u16(REG_POWER)      * 25 * self.current_lsb

def lectura_INA():
    """Returns a single reading from the INA226 as a formatted string."""
    try:
        i2c = I2C(0, scl=Pin(22), sda=Pin(21), freq=400000)
        devices = i2c.scan()

        if INA226_ADDR not in devices:
            return "ERROR: INA226 not detected at 0x{:02X}".format(INA226_ADDR)

        ina = INA226(i2c)

        mid, did = ina.manufacturer_id(), ina.die_id()
        if mid != 0x5449 or did != 0x2260:
            return "WARNING: INA226 IDs don't match -- check wiring/address."

        vbus = ina.bus_voltage()
        vsh  = ina.shunt_voltage() * 1000.0
        i_ma = ina.current() * 1000.0
        p    = ina.power()

        return "Vbus={:6.3f} V | Vshunt={:+8.3f} mV | I={:+8.2f} mA | P={:6.3f} W".format(
            vbus, vsh, i_ma, p)

    except Exception as e:
        return "ERROR: INA226 read failed: {}".format(e)
