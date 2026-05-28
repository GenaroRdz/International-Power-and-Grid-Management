import serial
import serial.tools.list_ports
import threading
import time

class ESP32Connection:
    def __init__(self, port='COM3', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.lock = threading.Lock()

    def list_ports(self):
        return [p.device for p in serial.tools.list_ports.comports()]

    def connect(self):
        with self.lock:
            try:
                self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
                # ESP32 often resets when the port opens — let MicroPython settle
                time.sleep(1.5)
                try:
                    self.ser.reset_input_buffer()
                except Exception:
                    pass
                return True
            except (serial.SerialException, OSError) as e:
                print(f"Connect error: {e}")
                self._kill_locked()
                return False

    def disconnect(self):
        with self.lock:
            self._kill_locked()

    def _kill_locked(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

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
        """Returns:
        'init'   — pong received, ESP32 is initialized
        'noinit' — pong received, ESP32 is NOT initialized (it rebooted)
        'boot'   — host saw a BOOT message (ESP32 just rebooted)
        None     — no response at all
        """
        with self.lock:
            if self.ser is None:
                return None
            try:
                try:
                    self.ser.reset_input_buffer()
                except Exception:
                    pass
                self.ser.write(b"ping\n")
                deadline = time.time() + timeout
                while time.time() < deadline:
                    line = self.ser.readline()
                    if not line:
                        continue
                    s = line.decode(errors="ignore").strip().lower()
                    if "boot" in s:
                        return "boot"
                    if "pong" in s:
                        return "noinit" if "noinit" in s else "init"
                return None
            except (serial.SerialException, OSError) as e:
                print(f"Ping error: {e}")
                self._kill_locked()
                return None


    @property
    def is_open(self):
        return self.ser is not None and self.ser.is_open