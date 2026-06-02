import serial
import serial.tools.list_ports
import threading
import time

from settings_store import load_settings, save_settings


class ESP32Connection:

    def __init__(self, port=None, baudrate=None):
        cfg = load_settings()
        self.port = port or cfg["port"]
        self.baudrate = baudrate or cfg["baudrate"]
        self.ser = None
        self.lock = threading.Lock()      # protects open / close / write only

        # Set by the GUI. Called from the reader thread for every line that is
        # not a pong/boot status line (e.g. the 'INA,...' stream).
        self.on_data = None

        # Reader thread and the small bits of state it updates.
        self._reader_thread = None
        self._reader_stop = threading.Event()
        self._pong_status = None          # 'init' / 'noinit' / None
        self._boot_seen = False

    def list_ports(self):
        return [p.device for p in serial.tools.list_ports.comports()]

    def list_ports_detailed(self):
        """Return [(device, description), ...] for every detected serial port,
        e.g. ('COM7', 'Silicon Labs CP210x USB to UART Bridge'). Used by the
        config 'detector' so the user can tell which port is the ESP32."""
        out = []
        for p in serial.tools.list_ports.comports():
            out.append((p.device, p.description or ""))
        return out

    def set_port(self, port, baudrate=None):
        """Update the target port (and optionally baudrate) and persist it.

        The caller must disconnect/reconnect for the change to take effect --
        re-opening a live port mid-read crashes the reader on Windows.
        """
        self.port = port
        if baudrate is not None:
            self.baudrate = baudrate
        save_settings({"port": self.port, "baudrate": self.baudrate})

    # ── Connect / disconnect ──────────────────────────────────────────────────
    def connect(self):
        # Stop any previous reader BEFORE touching the port. Closing a port while
        # the reader is mid-read crashes on Windows, so the reader must be gone
        # first. (This is the main cause of the reconnect/disconnect loop.)
        self._stop_reader()
        with self.lock:
            # Close anything already open first, so a leaked handle can't block
            # the re-open (a common cause of reconnect failures on Windows).
            self._kill_locked()
            try:
                self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
                # ESP32 usually resets when the port opens -- let it settle.
                time.sleep(1.5)
                try:
                    self.ser.reset_input_buffer()
                except Exception:
                    pass
            except (serial.SerialException, OSError) as e:
                print(f"Connect error: {e}")
                self._kill_locked()
                return False
        self._start_reader()              # begin reading (outside the lock)
        return True

    def disconnect(self):
        self._stop_reader()
        with self.lock:
            self._kill_locked()

    def _kill_locked(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    # ── Reader thread (the only reader of the port) ───────────────────────────
    def _start_reader(self):
        self._pong_status = None
        self._boot_seen = False
        self._reader_stop.clear()
        self._reader_thread = threading.Thread(target=self._reader_loop,
                                               daemon=True)
        self._reader_thread.start()

    def _stop_reader(self):
        self._reader_stop.set()
        t = self._reader_thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._reader_thread = None

    def _reader_loop(self):
        while not self._reader_stop.is_set():
            ser = self.ser
            if ser is None:
                break
            try:
                raw = ser.readline()              # blocks up to timeout (1 s)
            except Exception:
                # The port was closed/unplugged mid-read. On Windows this raises
                # TypeError (byref on a NULL handle), NOT SerialException -- so we
                # must catch everything and just exit the reader cleanly.
                break
            if not raw:
                continue                          # timed out -> just loop again
            line = raw.decode(errors="ignore").strip()
            if not line:
                continue

            low = line.lower()
            if "pong" in low:
                self._pong_status = "noinit" if "noinit" in low else "init"
            elif "boot" in low or "sistema listo" in low:
                self._boot_seen = True
            elif self.on_data is not None:
                try:
                    self.on_data(line)
                except Exception as e:
                    print(f"on_data error: {e}")

    # ── Send / ping ───────────────────────────────────────────────────────────
    def send(self, msg):
        with self.lock:
            if self.ser is None:
                return False
            try:
                self.ser.write((msg + '\n').encode())
                return True
            except (serial.SerialException, OSError) as e:
                print(f"Send error: {e}")
                self._kill_locked()
                return False

    def ping(self, timeout=1.0):
        """Ask the ESP32 'ping' and wait briefly for the reader to see a reply.

        Returns:
            'init'   -- pong received, ESP32 is initialized
            'noinit' -- pong received, ESP32 is NOT initialized
            'boot'   -- a BOOT message was seen (ESP32 rebooted)
            None     -- no response within the timeout
        This does NOT read the port itself; the reader thread does.
        """
        if not self.is_open:
            return None
        self._pong_status = None              # forget the previous pong
        if not self.send("ping"):
            return None
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._boot_seen:
                self._boot_seen = False       # consume the reboot event
                return "boot"
            if self._pong_status is not None:
                return self._pong_status
            time.sleep(0.02)
        return None

    @property
    def is_open(self):
        return self.ser is not None and self.ser.is_open