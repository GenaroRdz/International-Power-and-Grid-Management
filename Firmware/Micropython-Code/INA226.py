from machine import I2C, Pin
import struct
import setup

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
R_SHUNT        = 0.01
MAX_EXPECTED_A = 0.8

# The four INA226 sensors on the bus, in the order they will be reported.
INA_ADDRS = (0x44, 0x41, 0x45,0x40)

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

def lectura_INA(addr=INA226_ADDR):
    """Returns a single reading from the INA226 as a formatted string."""
    try:
        i2c = I2C(0, scl=Pin(22), sda=Pin(21), freq=400000)
        devices = i2c.scan()

        if not devices:
            return "ERROR: No I2C devices found"

        if addr not in devices:
            return "ERROR: INA226 not detected at 0x{:02X} | Found: {}".format(
                addr, str([hex(d) for d in devices]))

        ina = INA226(i2c, addr=addr)

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

# ── Multi-sensor manager (used by the async streaming task) ────────────────────
#
# Unlike lectura_INA() above -- which re-creates the bus and re-configures the
# chip on every single call -- these helpers set everything up ONCE and then
# just read the live values. That keeps each read fast (a few milliseconds),
# which is what lets the async loop stay responsive to commands and pings.

_i2c = None          # the shared I2C bus, created once in init_INAs()
_sensors = {}        # {address: INA226 instance} for the sensors actually found


def init_INAs(addrs=INA_ADDRS):
    """Create the I2C bus and configure each INA226 found on it (continuous mode).

    Call this once at startup (we call it from *INIT). Missing sensors are
    simply skipped -- they will report 'nan' later instead of crashing.
    Returns the dict of sensors that were detected.
    """
    global _i2c, _sensors
    _i2c = I2C(0, scl=Pin(setup.I2C_SCL), sda=Pin(setup.I2C_SDA), freq=400000)
    found = _i2c.scan()
    _sensors = {}
    for addr in addrs:
        if addr in found:
            _sensors[addr] = INA226(_i2c, addr=addr)
    return _sensors


def read_all_INAs(addrs=INA_ADDRS):
    """Read every sensor and return a list of (voltage_V, current_A, power_W)
    tuples, aligned to addrs.

    A missing or failed sensor contributes (nan, nan, nan). The three values
    per sensor are exactly what the GUI scope wants to plot per channel.
    """
    nan = float("nan")
    values = []
    for addr in addrs:
        ina = _sensors.get(addr)
        if ina is None:
            values.append((nan, nan, nan))
            continue
        try:
            values.append((ina.bus_voltage(), ina.current(), ina.power()))
        except Exception:
            values.append((nan, nan, nan))
    return values


def read_all_INAs_str(addrs=INA_ADDRS):
    """Return a single host-ready line:
    'INA,v1,i1,p1,v2,i2,p2,v3,i3,p3,v4,i4,p4'  (volts, amps, watts).

    The 'INA,' prefix lets the GUI tell streamed sensor data apart from
    ordinary command replies on the same serial line.
    """
    parts = []
    for v, i, p in read_all_INAs(addrs):
        parts.append("{:.3f}".format(v))
        parts.append("{:.3f}".format(i))
        parts.append("{:.3f}".format(p))
    return "INA," + ",".join(parts)
