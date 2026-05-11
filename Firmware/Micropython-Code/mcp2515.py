# MCP2515 CAN Bus Controller Library for MicroPython
# Compatible with ESP32 and other MicroPython boards
# Based on the MCP2515 datasheet

# ─── Register Addresses ───────────────────────────────────────────────────────
CANSTAT   = 0x0E
CANCTRL   = 0x0F
CNF3      = 0x28
CNF2      = 0x29
CNF1      = 0x2A
CANINTE   = 0x2B
CANINTF   = 0x2C
EFLG      = 0x2D

TXB0CTRL  = 0x30
TXB0SIDH  = 0x31
TXB0SIDL  = 0x32
TXB0EID8  = 0x33
TXB0EID0  = 0x34
TXB0DLC   = 0x35
TXB0D0    = 0x36

RXB0CTRL  = 0x60
RXB0SIDH  = 0x61
RXB0SIDL  = 0x62
RXB0EID8  = 0x63
RXB0EID0  = 0x64
RXB0DLC   = 0x65
RXB0D0    = 0x66

RXB1CTRL  = 0x70
RXB1SIDH  = 0x71
RXB1SIDL  = 0x72
RXB1DLC   = 0x75
RXB1D0    = 0x76

# ─── SPI Instructions ─────────────────────────────────────────────────────────
CMD_RESET      = 0xC0
CMD_READ       = 0x03
CMD_WRITE      = 0x02
CMD_RTS        = 0x80  # Request To Send
CMD_READ_STATUS= 0xA0
CMD_BIT_MODIFY = 0x05

# ─── Mode bits (CANCTRL register) ─────────────────────────────────────────────
MODE_NORMAL     = 0x00
MODE_SLEEP      = 0x20
MODE_LOOPBACK   = 0x40
MODE_LISTENONLY = 0x60
MODE_CONFIG     = 0x80
MODE_MASK       = 0xE0

# ─── Bitrate configs [CNF1, CNF2, CNF3] for 8 MHz and 16 MHz crystals ─────────
# Format: { speed_kbps: (CNF1, CNF2, CNF3) }
BITRATE_CFG_8MHZ = {
    125:  (0x03, 0xB6, 0x04),
    250:  (0x01, 0xB6, 0x04),
    500:  (0x00, 0xB6, 0x04),
    1000: (0x00, 0x92, 0x02),
}

BITRATE_CFG_16MHZ = {
    125:  (0x07, 0xB6, 0x04),
    250:  (0x03, 0xB6, 0x04),
    500:  (0x01, 0xB6, 0x04),
    1000: (0x00, 0xB6, 0x04),
}


class MCP2515:
    def __init__(self, spi, cs, crystal=16):
        """
        spi     : machine.SPI instance
        cs      : machine.Pin instance (Chip Select, active LOW)
        crystal : crystal frequency in MHz (8 or 16)
        """
        self._spi    = spi
        self._cs     = cs
        self._crystal = crystal
        self._cs.init(self._cs.OUT, value=1)

    # ── Low-level SPI helpers ─────────────────────────────────────────────────

    def _cs_low(self):
        self._cs.value(0)

    def _cs_high(self):
        self._cs.value(1)

    def _spi_transfer(self, data: bytes) -> bytes:
        buf = bytearray(len(data))
        self._spi.write_readinto(bytearray(data), buf)
        return bytes(buf)

    # ── Register access ───────────────────────────────────────────────────────

    def _read_reg(self, addr: int) -> int:
        self._cs_low()
        self._spi_transfer([CMD_READ, addr, 0x00])
        # Re-do as single transaction
        self._cs_high()
        self._cs_low()
        buf = bytearray(3)
        self._spi.write_readinto(bytearray([CMD_READ, addr, 0x00]), buf)
        self._cs_high()
        return buf[2]

    def _write_reg(self, addr: int, value: int):
        self._cs_low()
        self._spi.write(bytearray([CMD_WRITE, addr, value & 0xFF]))
        self._cs_high()

    def _bit_modify(self, addr: int, mask: int, value: int):
        self._cs_low()
        self._spi.write(bytearray([CMD_BIT_MODIFY, addr, mask & 0xFF, value & 0xFF]))
        self._cs_high()

    def _read_status(self) -> int:
        self._cs_low()
        buf = bytearray(2)
        self._spi.write_readinto(bytearray([CMD_READ_STATUS, 0x00]), buf)
        self._cs_high()
        return buf[1]

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self):
        """Hardware reset via SPI command. Places MCP2515 in Config mode."""
        self._cs_low()
        self._spi.write(bytearray([CMD_RESET]))
        self._cs_high()
        import time
        time.sleep_ms(10)

    def set_bitrate(self, kbps: int):
        """
        Set CAN bus speed.
        kbps: 125, 250, 500, or 1000
        Must be called while in Config mode (after reset()).
        """
        if self._crystal == 8:
            cfg = BITRATE_CFG_8MHZ
        else:
            cfg = BITRATE_CFG_16MHZ

        if kbps not in cfg:
            raise ValueError(f"Unsupported bitrate {kbps} kbps for {self._crystal} MHz crystal. "
                             f"Choose from {list(cfg.keys())}")

        cnf1, cnf2, cnf3 = cfg[kbps]
        self._write_reg(CNF1, cnf1)
        self._write_reg(CNF2, cnf2)
        self._write_reg(CNF3, cnf3)

    def _set_mode(self, mode: int, timeout_ms: int = 200):
        import time
        self._bit_modify(CANCTRL, MODE_MASK, mode)
        start = time.ticks_ms()
        while True:
            status = self._read_reg(CANSTAT)
            if (status & MODE_MASK) == mode:
                return
            if time.ticks_diff(time.ticks_ms(), start) > timeout_ms:
                raise RuntimeError(f"MCP2515: timeout switching to mode 0x{mode:02X} "
                                   f"(CANSTAT=0x{status:02X})")
            time.sleep_ms(1)

    def set_normal_mode(self):
        """Switch to Normal (operational) mode."""
        self._set_mode(MODE_NORMAL)
        # Accept all messages on RXB0 and RXB1
        self._write_reg(RXB0CTRL, 0x60)
        self._write_reg(RXB1CTRL, 0x60)

    def set_loopback_mode(self):
        """Switch to Loopback mode (TX is echoed back to RX — useful for testing)."""
        self._set_mode(MODE_LOOPBACK)

    def set_listen_only_mode(self):
        """Switch to Listen-Only mode (no ACK transmitted)."""
        self._set_mode(MODE_LISTENONLY)

    def set_config_mode(self):
        """Switch back to Configuration mode."""
        self._set_mode(MODE_CONFIG)

    # ── Transmit ──────────────────────────────────────────────────────────────

    def send_message(self, msg_id: int, data, extended: bool = False):
        """
        Send a CAN frame using TX Buffer 0.

        msg_id   : 11-bit standard ID (0x000–0x7FF) or 29-bit extended ID
        data     : list/bytes/bytearray of up to 8 bytes
        extended : True for extended (29-bit) frame
        """
        if len(data) > 8:
            raise ValueError("CAN data payload must be 8 bytes or fewer")

        # Wait for TX buffer 0 to be free
        import time
        deadline = time.ticks_ms() + 100
        while self._read_reg(TXB0CTRL) & 0x08:  # TXREQ bit
            if time.ticks_diff(time.ticks_ms(), deadline) > 0:
                raise RuntimeError("MCP2515: TX buffer 0 busy timeout")
            time.sleep_ms(1)

        if extended:
            # 29-bit extended ID
            self._write_reg(TXB0SIDH, (msg_id >> 21) & 0xFF)
            self._write_reg(TXB0SIDL, ((msg_id >> 13) & 0xE0) | 0x08 |
                            ((msg_id >> 16) & 0x03))
            self._write_reg(TXB0EID8, (msg_id >> 8) & 0xFF)
            self._write_reg(TXB0EID0,  msg_id & 0xFF)
        else:
            # 11-bit standard ID
            self._write_reg(TXB0SIDH, (msg_id >> 3) & 0xFF)
            self._write_reg(TXB0SIDL, (msg_id & 0x07) << 5)
            self._write_reg(TXB0EID8, 0x00)
            self._write_reg(TXB0EID0, 0x00)

        dlc = len(data)
        self._write_reg(TXB0DLC, dlc)

        for i, byte in enumerate(data):
            self._write_reg(TXB0D0 + i, byte)

        # Request transmission
        self._cs_low()
        self._spi.write(bytearray([CMD_RTS | 0x01]))  # RTS for TXB0
        self._cs_high()

    # ── Receive ───────────────────────────────────────────────────────────────

    def check_receive(self) -> bool:
        """Return True if a message is waiting in RXB0 or RXB1."""
        status = self._read_status()
        return bool(status & 0x03)  # bit0=RXB0, bit1=RXB1 (from READ_STATUS)

    def read_message(self):
        """
        Read the oldest available CAN message.
        Returns (msg_id, data_bytes) or (None, None) if no message.
        """
        status = self._read_status()

        if status & 0x01:       # RXB0 has a message
            sidh    = self._read_reg(RXB0SIDH)
            sidl    = self._read_reg(RXB0SIDL)
            eid8    = self._read_reg(RXB0EID8)
            eid0    = self._read_reg(RXB0EID0)
            dlc_reg = self._read_reg(RXB0DLC)
            dlc     = dlc_reg & 0x0F
            data    = bytes(self._read_reg(RXB0D0 + i) for i in range(dlc))
            # Clear RXB0 interrupt flag
            self._bit_modify(CANINTF, 0x01, 0x00)
            buf_sidh, buf_sidl, buf_eid8, buf_eid0 = sidh, sidl, eid8, eid0

        elif status & 0x02:     # RXB1 has a message
            sidh    = self._read_reg(RXB1SIDH)
            sidl    = self._read_reg(RXB1SIDL)
            eid8    = self._read_reg(0x73)  # RXB1EID8
            eid0    = self._read_reg(0x74)  # RXB1EID0
            dlc_reg = self._read_reg(RXB1DLC)
            dlc     = dlc_reg & 0x0F
            data    = bytes(self._read_reg(RXB1D0 + i) for i in range(dlc))
            # Clear RXB1 interrupt flag
            self._bit_modify(CANINTF, 0x02, 0x00)
            buf_sidh, buf_sidl, buf_eid8, buf_eid0 = sidh, sidl, eid8, eid0

        else:
            return None, None

        # Decode message ID
        if buf_sidl & 0x08:  # Extended frame
            msg_id = ((buf_sidh << 21) |
                      ((buf_sidl & 0xE0) << 13) |
                      ((buf_sidl & 0x03) << 16) |
                      (buf_eid8 << 8) |
                      buf_eid0)
        else:                # Standard frame
            msg_id = (buf_sidh << 3) | (buf_sidl >> 5)

        return msg_id, data

    # ── Filters & Masks (optional helpers) ───────────────────────────────────

    def set_filter(self, filter_num: int, ext: bool, value: int):
        """
        Configure an acceptance filter (0–5).
        Must be called in Config mode.
        """
        filter_regs = [0x00, 0x04, 0x08, 0x10, 0x14, 0x18]
        if filter_num < 0 or filter_num > 5:
            raise ValueError("filter_num must be 0–5")
        base = filter_regs[filter_num]
        self._write_filter_mask(base, ext, value)

    def set_mask(self, mask_num: int, ext: bool, value: int):
        """
        Configure acceptance mask 0 (covers RXB0) or 1 (covers RXB1).
        Must be called in Config mode.
        """
        mask_regs = [0x20, 0x24]
        if mask_num not in (0, 1):
            raise ValueError("mask_num must be 0 or 1")
        self._write_filter_mask(mask_regs[mask_num], ext, value)

    def _write_filter_mask(self, base: int, ext: bool, value: int):
        if ext:
            self._write_reg(base,     (value >> 21) & 0xFF)
            self._write_reg(base + 1, ((value >> 13) & 0xE0) | 0x08 |
                            ((value >> 16) & 0x03))
            self._write_reg(base + 2, (value >> 8) & 0xFF)
            self._write_reg(base + 3,  value & 0xFF)
        else:
            self._write_reg(base,     (value >> 3) & 0xFF)
            self._write_reg(base + 1, (value & 0x07) << 5)
            self._write_reg(base + 2, 0x00)
            self._write_reg(base + 3, 0x00)