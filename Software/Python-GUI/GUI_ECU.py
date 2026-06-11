import tkinter as tk
from tkinter import messagebox, ttk
import threading
import time
import queue
import serial.tools.list_ports
from New_device_connection_2 import ESP32Connection
from Scope_tab import ScopeTab
from settings import load_settings, save_settings
from CAN_Monitor_V7  import CanMonitor

# ── Colours ───────────────────────────────────────────────────────────────────
BG          = "#0d0f12"
PANEL       = "#161a1f"
CARD        = "#1c2128"
BORDER      = "#2a3040"
ACCENT_BLUE = "#00aaff"
ACCENT_CYAN = "#00e5cc"
GREEN       = "#00e676"
RED         = "#ff1744"
YELLOW      = "#ffd600"
LILA        = "#B565FC"
ORANGE      = "#ff8c00"
TEXT_SEC    = "#7a8899"
TEXT_DIM    = "#3d4a5a"

# Fonts are (re)generated at startup from the screen size, so they start as
# sensible defaults here and get overwritten by _apply_scaling().
FONT_LABEL  = ("Courier New", 8, "bold")
FONT_TITLE  = ("Courier New", 18, "bold")
FONT_STATUS = ("Courier New", 9)
FONT_CH     = ("Courier New", 11, "bold")
FONT_BTN    = ("Courier New", 8, "bold")
FONT_MODE   = ("Courier New", 8, "bold")

MODES = [
    ("BATTERY", GREEN),
    ("ACC",     ACCENT_BLUE),
    ("IGN",     YELLOW),
]

# ── LED widget ────────────────────────────────────────────────────────────────
class LED(tk.Canvas):
    def __init__(self, parent, size=10, bg_color=CARD, **kw):
        super().__init__(parent, width=size, height=size,
                         bg=bg_color, highlightthickness=0, **kw)
        self._oval = self.create_oval(1, 1, size-1, size-1,
                                      fill=TEXT_DIM, outline="")
        self._glow = self.create_oval(3, 3, size-3, size-3,
                                      fill="", outline="")

    def set(self, color=None, on=False):
        if on and color:
            self.itemconfig(self._oval, fill=color)
            self.itemconfig(self._glow, fill=self._lighten(color))
        else:
            self.itemconfig(self._oval, fill=TEXT_DIM)
            self.itemconfig(self._glow, fill="")

    @staticmethod
    def _lighten(hex_color):
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f"#{min(255,r+70):02x}{min(255,g+70):02x}{min(255,b+70):02x}"


# ── Configuration dialog (live COM port detector) ────────────────────────────
class ConfigDialog(tk.Toplevel):
    BAUDS = ["9600", "19200", "38400", "57600", "115200", "230400",
             "460800", "921600"]
    SCAN_MS = 1500          # how often the detector re-scans the ports

    def __init__(self, parent, esp32, on_apply):
        super().__init__(parent)
        self.esp32 = esp32
        self.on_apply = on_apply
        self._ports = []        # device strings, parallel to the listbox rows
        self._after_id = None

        self.title("Device Configuration")
        self.configure(bg=PANEL)
        self.resizable(False, False)
        self.transient(parent)

        body = tk.Frame(self, bg=PANEL, padx=20, pady=18)
        body.pack(fill="both", expand=True)

        tk.Label(body, text="⚙  DETECTED DEVICES", font=FONT_CH,
                 fg=ACCENT_CYAN, bg=PANEL).pack(anchor="w", pady=(0, 4))
        self.hint = tk.Label(body, text="Scanning…", font=FONT_LABEL,
                             fg=TEXT_SEC, bg=PANEL)
        self.hint.pack(anchor="w", pady=(0, 10))

        # ── Detector list ──
        list_frame = tk.Frame(body, bg=BORDER)
        list_frame.pack(fill="both")
        self.listbox = tk.Listbox(
            list_frame, height=6, width=46, font=FONT_STATUS,
            bg=CARD, fg="#cdd6e0",
            selectbackground=ACCENT_BLUE, selectforeground=BG,
            highlightthickness=1, highlightbackground=BORDER,
            relief="flat", bd=0, activestyle="none")
        sb = tk.Scrollbar(list_frame, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.listbox.pack(side="left", fill="both", expand=True)
        self.listbox.bind("<Double-Button-1>", lambda e: self._apply())

        self._dialog_btn(body, "⟳  Rescan", ACCENT_BLUE,
                         self._scan).pack(anchor="w", pady=(10, 0))

        # ── Baudrate ──
        baud_row = tk.Frame(body, bg=PANEL)
        baud_row.pack(fill="x", pady=(14, 0))
        tk.Label(baud_row, text="BAUDRATE", font=FONT_LABEL, fg=TEXT_SEC,
                 bg=PANEL).pack(side="left", padx=(0, 8))
        self.baud_var = tk.StringVar(value=str(esp32.baudrate))
        ttk.Combobox(baud_row, textvariable=self.baud_var,
                     values=self.BAUDS, width=10).pack(side="left")

        # ── Buttons ──
        btns = tk.Frame(body, bg=PANEL)
        btns.pack(fill="x", pady=(18, 0))
        self._dialog_btn(btns, "Select", GREEN, self._apply).pack(
            side="right", padx=(8, 0))
        self._dialog_btn(btns, "Cancel", TEXT_SEC, self._close).pack(
            side="right")

        self._scan()                    # populate + start the live re-scan loop
        self.update_idletasks()
        self._center_on(parent)
        self.grab_set()
        self.listbox.focus_set()
        self.bind("<Escape>", lambda e: self._close())
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _dialog_btn(self, parent, text, color, cmd):
        return tk.Button(parent, text=text, command=cmd, font=FONT_BTN,
                         width=10, fg=color, bg=CARD, activeforeground=BG,
                         activebackground=color, relief="flat", bd=0,
                         highlightbackground=color, highlightthickness=1,
                         cursor="hand2", pady=5)

    def _center_on(self, parent):
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        w, h = self.winfo_width(), self.winfo_height()
        self.geometry(f"+{px + (pw - w)//2}+{py + (ph - h)//2}")

    def _detect_ports(self):
        """Return [(device, description), ...] for every serial port found.
        Scans pyserial directly so it works regardless of the connection
        module, with a name-only fallback if anything goes wrong."""
        try:
            return [(p.device, p.description or "")
                    for p in serial.tools.list_ports.comports()]
        except Exception:
            try:
                return [(d, "") for d in self.esp32.list_ports()]
            except Exception:
                return []

    def _scan(self):
        """Detect the currently connected serial ports and refresh the list.
        Re-runs itself every SCAN_MS so newly plugged devices appear on their
        own. The user's current selection is preserved across rescans."""
        # Remember what was selected so a rescan doesn't move the highlight.
        sel = self.listbox.curselection()
        prev = self._ports[sel[0]] if sel else self.esp32.port

        detected = self._detect_ports()
        self._ports = [dev for dev, _ in detected]

        self.listbox.delete(0, tk.END)
        for dev, desc in detected:
            self.listbox.insert(tk.END, f"  {dev}    {desc}".rstrip())

        if detected:
            self.hint.configure(
                text=f"{len(detected)} device(s) detected — pick one and Select",
                fg=GREEN)
            idx = self._ports.index(prev) if prev in self._ports else 0
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(idx)
            self.listbox.see(idx)
        else:
            self.hint.configure(
                text="No devices detected — plug in the ESP32 and wait",
                fg=YELLOW)

        self._after_id = self.after(self.SCAN_MS, self._scan)

    def _apply(self):
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showwarning("Configuration",
                                   "Select a detected port first.", parent=self)
            return
        port = self._ports[sel[0]]
        try:
            baud = int(self.baud_var.get().strip())
        except ValueError:
            messagebox.showwarning("Configuration",
                                   "Baudrate must be a number.", parent=self)
            return
        self._close()
        self.on_apply(port, baud)

    def _close(self):
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        self.destroy()


# ── Main App ──────────────────────────────────────────────────────────────────
class ECUSupplyController(tk.Tk):
    NUM_CHANNELS = 4

    ST_DISCONNECTED = "DISCONNECTED"
    ST_CONNECTING   = "CONNECTING"
    ST_CONNECTED    = "CONNECTED"
    ST_RECONNECTING = "RECONNECTING"

    PING_INTERVAL   = 2.0
    MISS_THRESHOLD  = 4
    RECONNECT_DELAY = 2.0

    def __init__(self):
        super().__init__()
        self.title("ECU Supply Controller")
        self.configure(bg=BG)

        # ── PANTALLA COMPLETA ─────────────────────────────────────────────────
        try:
            self.state('zoomed')                 # Maximizado en Windows
        except tk.TclError:
            self.attributes('-fullscreen', True)  # Fullscreen en Linux/Mac
        self.bind("<Escape>", lambda e: self.attributes("-fullscreen", False))

        # ── Settings + scaling (must run before building the UI) ──────────────
        self.settings = load_settings()
        self._apply_scaling()

        self.esp32 = ESP32Connection(port=self.settings["port"],
                                     baudrate=self.settings["baudrate"])
        self.esp32.on_data = self._on_serial_line   # INA stream -> graphs
        self.conn_state      = self.ST_DISCONNECTED
        self._want_connected = False
        self._worker_stop    = threading.Event()

        self.initialized = False
        self.connected   = False
        self._init_time  = 0.0          # when *INIT was last sent (race guard)

        self._ina_queue = queue.Queue()

        self.ch_states = [[False] * len(MODES) for _ in range(self.NUM_CHANNELS)]

        self.current_vals = [0.0] * self.NUM_CHANNELS
        self.power_vals   = [0.0] * self.NUM_CHANNELS

        self._build_ui()
        self._update_status_bar()
        self._start_worker()
        self._poll_ina_queue()

    # ── Responsive scaling ────────────────────────────────────────────────────
    def _apply_scaling(self):
        """Pick a scale factor from the screen size and regenerate the fonts.

        Reference design is 1920x1080. On a smaller laptop panel everything is
        scaled down (clamped so it never gets unreadably small or larger than
        the original)."""
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        scale = min(sw / 1920.0, sh / 1080.0)
        self.scale = max(0.60, min(scale, 1.0))

        s = self.scale
        global FONT_LABEL, FONT_TITLE, FONT_STATUS, FONT_CH, FONT_BTN, FONT_MODE

        def fnt(px, weight="bold"):
            return ("Courier New", max(6, int(round(px * s))), weight)

        FONT_LABEL  = fnt(8)
        FONT_TITLE  = fnt(18)
        FONT_STATUS = ("Courier New", max(7, int(round(9 * s))))
        FONT_CH     = fnt(11)
        FONT_BTN    = fnt(8)
        FONT_MODE   = fnt(8)

    def _s(self, n):
        """Scale a pixel padding/size by the current factor (min 1)."""
        return max(1, int(round(n * self.scale)))

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Status bar pinned to the bottom of the WINDOW
        self._build_status_bar(self)

        container = tk.Frame(self, bg=BG)
        container.pack(side="top", fill="both", expand=True)

        canvas = tk.Canvas(container, bg=BG, highlightthickness=0)
        vscroll = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        outer = tk.Frame(canvas, bg=BG, padx=self._s(18), pady=self._s(18))
        win_id = canvas.create_window((0, 0), window=outer, anchor="nw")

        def _on_inner_config(_):
            canvas.configure(scrollregion=canvas.bbox("all"))
        outer.bind("<Configure>", _on_inner_config)

        def _on_canvas_config(e):
            canvas.itemconfig(win_id, width=e.width)
        canvas.bind("<Configure>", _on_canvas_config)

        # Mouse wheel scrolling
        def _wheel(e):
            delta = -1 if (getattr(e, "num", None) == 5 or e.delta < 0) else 1
            canvas.yview_scroll(-delta, "units")
        canvas.bind_all("<MouseWheel>", _wheel)
        canvas.bind_all("<Button-4>", _wheel)
        canvas.bind_all("<Button-5>", _wheel)
        self._canvas = canvas

        self._build_title(outer)

        # ── TAB SYSTEM BAR ────────────────────────────────────────────────────
        tab_bar = tk.Frame(outer, bg=BG)
        tab_bar.pack(fill="x", pady=(0, self._s(10)))

        self.tab_btn_ecu = tk.Button(tab_bar, text="⬡ ECU CONTROL", font=FONT_CH,
                                     fg=BG, bg=ACCENT_BLUE, relief="flat", bd=0,
                                     activebackground=ACCENT_CYAN, cursor="hand2",
                                     command=lambda: self._switch_tab("ECU"), 
                                     padx=15, pady=5)
        self.tab_btn_ecu.pack(side="left", padx=(0, 5))

        self.tab_btn_can = tk.Button(tab_bar, text="⬡ CAN MONITOR", font=FONT_CH,
                                     fg=TEXT_SEC, bg=CARD, relief="flat", bd=0,
                                     activebackground=BORDER, cursor="hand2",
                                     command=lambda: self._switch_tab("CAN"), 
                                     padx=15, pady=5)
        self.tab_btn_can.pack(side="left")

        # ── TAB CONTAINERS ────────────────────────────────────────────────────
        self.frame_ecu = tk.Frame(outer, bg=BG)
        self.frame_ecu.pack(fill="both", expand=True)

        self.frame_can = tk.Frame(outer, bg=BG)
        # We don't pack frame_can yet; it starts hidden

        # --- ECU CONTROL TAB CONTENT ---
        main = tk.Frame(self.frame_ecu, bg=PANEL, bd=0,
                        highlightbackground=BORDER, highlightthickness=1)
        main.pack(fill="both", expand=True, pady=(0, self._s(12)))

        pad = dict(padx=self._s(18), pady=self._s(14))
        self._build_init_row(main, pad)
        self._build_separator(main)
        self._build_channels(main, pad)
        self._build_separator(main)

        # Embedded Scope
        self._scope_tab = ScopeTab(main, bg=PANEL)
        self._scope_tab.pack(fill="both", expand=True,
                             padx=self._s(18), pady=self._s(10))

        self._build_separator(main)
        self._build_global_buttons(main, pad)

        # --- CAN MONITOR TAB CONTENT ---
        self._can_monitor = CanMonitor(self.frame_can)
        self._can_monitor.pack(fill="both", expand=True)

    def _switch_tab(self, tab):
        """Handles switching between the ECU Control and CAN Monitor frames."""
        if tab == "ECU":
            self.frame_can.pack_forget()
            self.frame_ecu.pack(fill="both", expand=True)
            self.tab_btn_ecu.configure(bg=ACCENT_BLUE, fg=BG)
            self.tab_btn_can.configure(bg=CARD, fg=TEXT_SEC)
            self._canvas.yview_moveto(0) # Reset scroll position
        else:
            self.frame_ecu.pack_forget()
            self.frame_can.pack(fill="both", expand=True)
            self.tab_btn_ecu.configure(bg=CARD, fg=TEXT_SEC)
            self.tab_btn_can.configure(bg=ACCENT_BLUE, fg=BG)
            self._canvas.yview_moveto(0) # Reset scroll position

    def _build_title(self, parent):
        hdr = tk.Frame(parent, bg=BG)
        hdr.pack(fill="x", pady=(0, self._s(14)))
        tk.Label(hdr, text="⬡ ECU SUPPLY CONTROLLER",
                 font=FONT_TITLE, fg=ACCENT_BLUE, bg=BG).pack(side="left")

        # ── Right corner: Configure button + current-port indicator ──
        self.config_btn = self._make_btn(hdr, "⚙  CONFIGURE", ACCENT_CYAN,
                                          self._open_config, width=14)
        self.config_btn.pack(side="right", padx=(self._s(8), 0), pady=self._s(4))

        self.port_var = tk.StringVar(value=self.esp32.port)
        tk.Label(hdr, textvariable=self.port_var, font=FONT_LABEL,
                 fg=ACCENT_CYAN, bg=BG).pack(side="right", padx=(0, self._s(10)))
        tk.Label(hdr, text="PORT:", font=FONT_LABEL,
                 fg=TEXT_DIM, bg=BG).pack(side="right")

    def _build_init_row(self, parent, pad):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", **pad)

        self.init_btn = self._make_btn(row, "⚡  INIT", ACCENT_BLUE,
                                       self._do_init, width=12)
        self.init_btn.pack(side="left", padx=(0, self._s(14)))

        self.init_led = LED(row, size=self._s(14), bg_color=PANEL)
        self.init_led.pack(side="left", padx=(0, 2))

        tk.Label(row, text="INITIALIZED", font=FONT_LABEL,
                 fg=TEXT_SEC, bg=PANEL).pack(side="left", padx=(0, self._s(24)))

        self.conn_led = LED(row, size=self._s(14), bg_color=PANEL)
        self.conn_led.pack(side="left", padx=(0, self._s(5)))
        self.conn_label = tk.Label(row, text="DEVICE NOT CONNECTED",
                                   font=FONT_LABEL, fg=TEXT_SEC, bg=PANEL)
        self.conn_label.pack(side="left")

        self.connect_button = self._make_btn(row, "Connect", GREEN,
                                             self._connect, width=14)
        self.connect_button.pack(side="left", padx=(self._s(80), self._s(8)))

        self.disconnect_button = self._make_btn(row, "Disconnect", RED,
                                                self._disconnect, width=14)
        self.disconnect_button.pack(side="left", padx=(self._s(5), self._s(8)))

    def _build_channels(self, parent, pad):
        tk.Label(parent, text="CHANNEL OUTPUTS",
                 font=FONT_LABEL, fg=TEXT_DIM, bg=PANEL).pack(
            anchor="w", padx=self._s(18), pady=(self._s(4), 0))

        grid = tk.Frame(parent, bg=PANEL)
        grid.pack(fill="x", **pad)

        ch_colors = [ACCENT_BLUE, ACCENT_CYAN, GREEN, ORANGE]

        self.ch_leds           = []
        self.ch_master_btns    = []
        self.mode_leds         = []
        self.mode_toggle_btns  = []
        self.all_mode_leds     = []
        self.all_toggle_btns   = []

        cpad = self._s(10)
        all_card = tk.Frame(grid, bg=CARD,
                            highlightbackground=BORDER, highlightthickness=1,
                            padx=cpad, pady=cpad)
        all_card.grid(row=0, column=0, padx=self._s(5), sticky="nsew")
        grid.columnconfigure(0, weight=1)

        tk.Label(all_card, text="ALL", font=FONT_CH,
                 fg=GREEN, bg=CARD).pack(pady=(0, self._s(6)))

        for m_idx, (m_label, m_color) in enumerate(MODES):
            row_f = tk.Frame(all_card, bg=CARD)
            row_f.pack(fill="x", pady=self._s(3))

            led = LED(row_f, size=self._s(10), bg_color=CARD)
            led.pack(side="left", padx=(0, self._s(4)))
            self.all_mode_leds.append(led)

            tk.Label(row_f, text=m_label, font=FONT_MODE,
                     fg=m_color, bg=CARD, width=7,
                     anchor="w").pack(side="left", padx=(0, self._s(6)))

            btn = self._make_toggle_btn(
                row_f, lambda mi=m_idx: self._all_mode_toggle(mi))
            btn.pack(side="right")
            self.all_toggle_btns.append(btn)

        for ch in range(self.NUM_CHANNELS):
            col = ch + 1
            card = tk.Frame(grid, bg=CARD,
                            highlightbackground=BORDER, highlightthickness=1,
                            padx=cpad, pady=cpad)
            card.grid(row=0, column=col, padx=self._s(5), sticky="nsew")
            grid.columnconfigure(col, weight=1)

            hdr = tk.Frame(card, bg=CARD)
            hdr.pack(fill="x", pady=(0, self._s(6)))
            ch_led = LED(hdr, size=self._s(12), bg_color=CARD)
            ch_led.pack(side="left", padx=(0, self._s(5)))
            self.ch_leds.append(ch_led)
            tk.Label(hdr, text=f"CH {ch+1:02d}", font=FONT_CH,
                     fg=ch_colors[ch], bg=CARD).pack(side="left")

            master_btn = self._make_toggle_btn(
                hdr, lambda c=ch: self._ch_master_toggle(c))
            master_btn.pack(side="right")
            self.ch_master_btns.append(master_btn)

            ch_mode_leds      = []
            ch_mode_tog_btns  = []

            for m_idx, (m_label, m_color) in enumerate(MODES):
                row_f = tk.Frame(card, bg=CARD)
                row_f.pack(fill="x", pady=self._s(3))

                led = LED(row_f, size=self._s(10), bg_color=CARD)
                led.pack(side="left", padx=(0, self._s(4)))
                ch_mode_leds.append(led)

                tk.Label(row_f, text=m_label, font=FONT_MODE,
                         fg=m_color, bg=CARD, width=7,
                         anchor="w").pack(side="left", padx=(0, self._s(6)))

                btn = self._make_toggle_btn(
                    row_f, lambda c=ch, mi=m_idx: self._ch_mode_toggle(c, mi))
                btn.pack(side="right")
                ch_mode_tog_btns.append(btn)

            self.mode_leds.append(ch_mode_leds)
            self.mode_toggle_btns.append(ch_mode_tog_btns)

    def update_reading(self, ch, voltage, current, power):
        """Envía los datos a las gráficas y actualiza sumatorias globales."""
        self._scope_tab.push(ch, voltage, current, power)

        self.current_vals[ch] = current
        self.power_vals[ch]   = power

        tot_c = sum(self.current_vals)
        tot_p = sum(self.power_vals)

        self.total_curr_var.set(f"{tot_c:.2f} A")
        self.total_pwr_var.set(f"{tot_p:.2f} W")

    # ── INA226 stream from the ESP32 ──────────────────────────────────────────
    def _on_serial_line(self, line):
        # INA226 stream -> scope/graphs
        if line.startswith("INA,"):
            nums = [self._safe_float(p) for p in line.split(",")[1:]]
            self._ina_queue.put(nums)
            return
        # Every other line that isn't a pong/boot status (those are consumed by
        # ESP32Connection before we get here) is forwarded to the CAN monitor.
        # feed_line() is thread-safe: it only touches a queue / plain attribute,
        # never a Tk widget, so calling it from the reader thread is fine.
        cm = getattr(self, "_can_monitor", None)
        if cm is not None:
            cm.feed_line(line)

    def _poll_ina_queue(self):
        try:
            while True:
                nums = self._ina_queue.get_nowait()
                self._apply_readings(nums)
        except queue.Empty:
            pass
        self.after(50, self._poll_ina_queue)

    def _apply_readings(self, nums):
        for ch in range(self.NUM_CHANNELS):
            base = ch * 3
            if base + 2 < len(nums):
                self.update_reading(ch, nums[base], nums[base + 1],
                                    nums[base + 2])

    @staticmethod
    def _safe_float(s):
        try:
            x = float(s)
            return 0.0 if x != x else x
        except ValueError:
            return 0.0

    def _reset_readings(self):
        self._scope_tab.reset_buffers()
        self.current_vals = [0.0] * self.NUM_CHANNELS
        self.power_vals   = [0.0] * self.NUM_CHANNELS
        self.total_curr_var.set("0.00 A")
        self.total_pwr_var.set("0.00 W")

    def _build_global_buttons(self, parent, pad):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", **pad)

        self.all_on_btn = self._make_btn(row, "▶  ALL ON", GREEN,
                                         self._all_on, width=14)
        self.all_on_btn.pack(side="left", padx=(0, self._s(8)))

        self.all_off_btn = self._make_btn(row, "■  ALL OFF", RED,
                                          self._all_off, width=14)
        self.all_off_btn.pack(side="left", padx=(0, self._s(8)))

        tk.Frame(row, bg=PANEL).pack(side="left", expand=True)

        self.reset_btn = self._make_btn(row, "↺  RESET", YELLOW,
                                        self._do_reset, width=12)
        self.reset_btn.pack(side="right")

    def _build_status_bar(self, parent):
        bar = tk.Frame(parent, bg="#0a0c0f",
                       highlightbackground=BORDER, highlightthickness=1)
        bar.pack(side="bottom", fill="x")

        inner = tk.Frame(bar, bg="#0a0c0f", padx=self._s(14), pady=self._s(8))
        inner.pack(fill="x")

        tk.Label(inner, text="⬡", font=("Courier New", max(8, self._s(11))),
                 fg=ACCENT_BLUE, bg="#0a0c0f").pack(side="left", padx=(0, 6))
        tk.Label(inner, text="MICROCHIP STATUS:", font=FONT_STATUS,
                 fg=TEXT_DIM, bg="#0a0c0f").pack(side="left")

        self.status_led = LED(inner, size=self._s(10), bg_color="#0a0c0f")
        self.status_led.pack(side="left", padx=self._s(8))

        self.status_var = tk.StringVar(value="NOT INITIALIZED")
        self.status_lbl = tk.Label(inner, textvariable=self.status_var,
                                   font=FONT_STATUS, fg=RED, bg="#0a0c0f")
        self.status_lbl.pack(side="left")

        self.active_ch_var = tk.StringVar(value="0 / 12")
        tk.Label(inner, textvariable=self.active_ch_var,
                 font=FONT_STATUS, fg=TEXT_SEC, bg="#0a0c0f").pack(side="right")
        tk.Label(inner, text="ACTIVE OUTPUTS:", font=FONT_STATUS,
                 fg=TEXT_DIM, bg="#0a0c0f").pack(side="right", padx=(0, 4))

        tk.Frame(inner, width=self._s(20), bg="#0a0c0f").pack(side="right")

        self.total_pwr_var = tk.StringVar(value="0.00 W")
        tk.Label(inner, textvariable=self.total_pwr_var,
                 font=FONT_STATUS, fg=LILA, bg="#0a0c0f").pack(side="right")
        tk.Label(inner, text="Σ PWR:", font=FONT_STATUS,
                 fg=TEXT_DIM, bg="#0a0c0f").pack(side="right", padx=(0, 4))

        tk.Frame(inner, width=self._s(15), bg="#0a0c0f").pack(side="right")

        self.total_curr_var = tk.StringVar(value="0.00 A")
        tk.Label(inner, textvariable=self.total_curr_var,
                 font=FONT_STATUS, fg=ACCENT_CYAN, bg="#0a0c0f").pack(
            side="right")
        tk.Label(inner, text="Σ CURR:", font=FONT_STATUS,
                 fg=TEXT_DIM, bg="#0a0c0f").pack(side="right", padx=(0, 4))

    def _build_separator(self, parent):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=self._s(14))

    # ── Button factories ──────────────────────────────────────────────────────
    def _make_btn(self, parent, text, color, cmd, width=10):
        btn = tk.Button(parent, text=text, command=cmd,
                        font=FONT_BTN, width=width,
                        fg=color, bg=CARD,
                        activeforeground=BG, activebackground=color,
                        relief="flat", bd=0,
                        highlightbackground=color, highlightthickness=1,
                        cursor="hand2", pady=self._s(6))
        btn.bind("<Enter>", lambda e, b=btn, c=color: b.configure(bg=c, fg=BG))
        btn.bind("<Leave>", lambda e, b=btn, c=color: (
            b.configure(bg=CARD, fg=c) if str(b["state"]) != "disabled"
            else None))
        return btn

    def _make_toggle_btn(self, parent, cmd):
        btn = tk.Button(parent, text="OFF", command=cmd,
                        font=FONT_MODE, width=5,
                        fg=RED, bg=CARD,
                        activeforeground=BG, activebackground=GREEN,
                        relief="flat", bd=0,
                        highlightbackground=RED, highlightthickness=1,
                        cursor="hand2", pady=self._s(2))
        btn._is_on = False

        def _enter(e, b=btn):
            if str(b["state"]) == "disabled":
                return
            b.configure(bg=GREEN if b._is_on else RED, fg=BG)

        def _leave(e, b=btn):
            if str(b["state"]) == "disabled":
                return
            if b._is_on:
                b.configure(bg=GREEN, fg=BG)
            else:
                b.configure(bg=CARD, fg=RED)

        btn.bind("<Enter>", _enter)
        btn.bind("<Leave>", _leave)
        return btn

    def _set_toggle(self, btn, on):
        btn._is_on = on
        if on:
            btn.configure(text="ON", fg=BG, bg=GREEN,
                          highlightbackground=GREEN)
        else:
            btn.configure(text="OFF", fg=RED, bg=CARD,
                          highlightbackground=RED)

    # ── Configuration ─────────────────────────────────────────────────────────
    def _open_config(self):
        # Changing the port while connected would point at a different device,
        # so disconnect first (with the user's consent).
        if self.conn_state != self.ST_DISCONNECTED:
            if not messagebox.askyesno(
                    "Configuration",
                    "Changing the port requires disconnecting from the "
                    "current device.\n\nDisconnect now?", parent=self):
                return
            self._shutdown_link()
        ConfigDialog(self, self.esp32, self._apply_config)

    def _apply_config(self, port, baud):
        self.esp32.set_port(port, baud)
        self.settings["port"] = port
        self.settings["baudrate"] = baud
        save_settings(self.settings)
        self.port_var.set(port)

    # ── Guard ─────────────────────────────────────────────────────────────────
    def _guard(self):
        if not self.connected:
            messagebox.showwarning(
                "Device Not Connected",
                "⚠ Device is not connected.\n\n"
                "Please connect to the device.", parent=self)
            return False
        if not self.initialized:
            messagebox.showwarning(
                "Device Not Initialized",
                "⚠  Device is not initialized.\n\n"
                "Please press INIT first.", parent=self)
            return False
        return True

    # ── Init ──────────────────────────────────────────────────────────────────
    def _do_init(self):
        if self.initialized:
            return
        if not self.connected:
            messagebox.showwarning(
                "Device Not Connected",
                "⚠ Connect to the device before initializing.",
                parent=self)
            return
        self.esp32.send("*INIT")
        self.initialized = True
        self._init_time  = time.time()
        self.init_led.set(GREEN, on=True)
        self.init_btn.configure(fg=GREEN, highlightbackground=GREEN)
        self.esp32.send("INA_START")
        # Ask the board once whether its CAN controller initialised OK, so the
        # CAN monitor can display real bus health (reply: 'CAN_OK = True/False').
        self.esp32.send("CAN?")
        self._update_status_bar()

    # ── Connect / Disconnect (user actions) ───────────────────────────────────
    def _connect(self):
        if self.conn_state in (self.ST_CONNECTING, self.ST_CONNECTED,
                               self.ST_RECONNECTING):
            return
        self._want_connected = True
        self._set_conn_state(self.ST_CONNECTING)

    def _disconnect(self):
        self._shutdown_link()

    def _set_conn_state(self, state):
        self.conn_state = state
        self.connected  = (state == self.ST_CONNECTED)

        if state == self.ST_CONNECTED:
            self.conn_led.set(GREEN, on=True)
            self.conn_label.configure(text="DEVICE CONNECTED", fg=GREEN)
        elif state == self.ST_CONNECTING:
            self.conn_led.set(YELLOW, on=True)
            self.conn_label.configure(text="CONNECTING…", fg=YELLOW)
        elif state == self.ST_RECONNECTING:
            self.conn_led.set(YELLOW, on=True)
            self.conn_label.configure(text="RECONNECTING…", fg=YELLOW)
        else:
            self.conn_led.set(RED, on=True)
            self.conn_label.configure(text="DEVICE NOT CONNECTED", fg=TEXT_SEC)

        self._update_status_bar()
        self._scope_tab.set_connected(self.connected)
        cm = getattr(self, "_can_monitor", None)
        if cm is not None:
            cm.set_connected(self.connected)

    # ── Reset ─────────────────────────────────────────────────────────────────
    def _do_reset(self):
        if not messagebox.askyesno("Reset",
                                   "Reset all channels and disconnect?",
                                   parent=self):
            return
        self._shutdown_link()

    def _shutdown_link(self):
        self._want_connected = False
        if self.connected:
            try:
                self.esp32.send("*RST")
                time.sleep(0.05)
            except Exception:
                pass
        self.esp32.disconnect()

        self.initialized = False
        self.ch_states = [[False] * len(MODES) for _ in range(self.NUM_CHANNELS)]

        for ch in range(self.NUM_CHANNELS):
            for m_idx in range(len(MODES)):
                self._apply_ch_mode_ui(ch, m_idx, False)
            self._update_ch_led(ch)
            self._set_toggle(self.ch_master_btns[ch], False)
        for m_idx in range(len(MODES)):
            self._update_all_led(m_idx)
            self._set_toggle(self.all_toggle_btns[m_idx], False)

        self._reset_readings()
        self.init_led.set(on=False)
        self.init_btn.configure(state="normal", text="⚡  INIT",
                                fg=ACCENT_BLUE, highlightbackground=ACCENT_BLUE)
        self._set_conn_state(self.ST_DISCONNECTED)

    # ── Channel / mode toggles ────────────────────────────────────────────────
    def _ch_master_toggle(self, ch):
        if not self._guard():
            return
        new_state = not all(self.ch_states[ch])
        for m_idx in range(len(MODES)):
            self.ch_states[ch][m_idx] = new_state
            self._apply_ch_mode_ui(ch, m_idx, new_state)
            self._update_all_led(m_idx)
        self._update_ch_led(ch)
        self._set_toggle(self.ch_master_btns[ch], new_state)
        self._update_status_bar()

    def _ch_mode_toggle(self, ch, m_idx):
        if not self._guard():
            return
        new_state = not self.ch_states[ch][m_idx]
        self.ch_states[ch][m_idx] = new_state
        self._apply_ch_mode_ui(ch, m_idx, new_state)
        self._update_ch_led(ch)
        self._update_all_led(m_idx)
        self._set_toggle(self.ch_master_btns[ch], all(self.ch_states[ch]))
        self._update_status_bar()

    def _apply_ch_mode_ui(self, ch, m_idx, on):
        _, m_color = MODES[m_idx]
        self.mode_leds[ch][m_idx].set(m_color if on else None, on=on)
        self._set_toggle(self.mode_toggle_btns[ch][m_idx], on)

        if ch == 0 and m_idx == 0 and on:   #Battery Channel 1
            print("CH1 BATTERY is ON")
            self.esp32.send("ECU1_BAT on")

        if ch == 0 and m_idx == 0 and not on:
            print("CH1 BATTERY is OFF")
            self.esp32.send("ECU1_BAT off")
               #Battery Channel 1

        if ch == 0 and m_idx == 1 and on:   #ACC Channel 1
            print("CH1 ACC is ON")
            self.esp32.send("ECU1_ACC on")

        if ch == 0 and m_idx == 1 and not on:   #ACC Channel 1
            self.esp32.send("ECU1_ACC off")

        if ch == 0 and m_idx == 2 and on:   #IGN Channel 1
            print("CH1 IGN is ON")
            self.esp32.send("ECU1_IGN on")
        if ch == 0 and m_idx == 2 and not on:   #IGN Channel 1
            self.esp32.send("ECU1_IGN off")

        #CH02

        if ch == 1 and m_idx == 0 and on:   #Battery Channel 2
            print("CH2 BATTERY is on")
            self.esp32.send("ECU2_BAT on")
        if ch == 1 and m_idx == 0 and not on:   #Battery Channel 2
            print("CH2 BATTERY is off")
            self.esp32.send("ECU2_BAT off")

        if ch == 1 and m_idx == 1 and on:   #ACC Channel 2
            print("CH2 ACC is off")
            self.esp32.send("ECU2_ACC on")
        if ch == 1 and m_idx == 1 and not on:   #ACC Channel 2
            print("CH2 ACC is OFF")
            self.esp32.send("ECU2_ACC off")

        if ch == 1 and m_idx == 2 and on:   #IGN Channel 2
            print("CH2 IGN is Off")
            self.esp32.send("ECU2_IGN on")
        if ch == 1 and m_idx == 2 and not on:   #IGN Channel 2
            print("CH2 IGN is OFF")
            self.esp32.send("ECU2_IGN off")

        #CH03

        if ch == 2 and m_idx == 0 and on:   #Battery Channel 3
            print("CH3 BATTERY is OFF")
            self.esp32.send("ECU3_BAT on")
        if ch == 2 and m_idx == 0 and not on:   #Battery Channel 3
            print("CH3 BATTERY is OFF")
            self.esp32.send("ECU3_BAT off")

        if ch == 2 and m_idx == 1 and on:   #ACC Channel 3
            print("CH3 ACC is ON")
            self.esp32.send("ECU3_ACC on")
        if ch == 2 and m_idx == 1 and not on:   #ACC Channel 3
            print("CH3 ACC is OFF")
            self.esp32.send("ECU3_ACC off")

        if ch == 2 and m_idx == 2 and on:   #IGN Channel 3
            print("CH3 IGN is ON")
            self.esp32.send("ECU3_IGN on")
        if ch == 2 and m_idx == 2 and not on:   #IGN Channel 3
            print("CH3 IGN is OFF")
            self.esp32.send("ECU3_IGN off")

        #CH04

        if ch == 3 and m_idx == 0 and on:   #Battery Channel 4
            print("CH4 BATTERY is ON")
            self.esp32.send("ECU4_BAT on")
        if ch == 3 and m_idx == 0 and not on:   #Battery Channel 4
            print("CH4 BATTERY is OFF")
            self.esp32.send("ECU4_BAT off")

        if ch == 3 and m_idx == 1 and on:   #ACC Channel 4
            print("CH4 ACC is ON")
            self.esp32.send("ECU4_ACC on")
        if ch == 3 and m_idx == 1 and not on:   #ACC Channel 4
            print("CH4 ACC is OFF")
            self.esp32.send("ECU4_ACC off")

        if ch == 3 and m_idx == 2 and on:   #IGN Channel 4
            print("CH4 IGN is ON")
            self.esp32.send("ECU4_IGN on")
        if ch == 3 and m_idx == 2 and not on:   #IGN Channel 4
            print("CH4 IGN is OFF")
            self.esp32.send("ECU4_IGN off")

    def _update_ch_led(self, ch):
        ch_colors = [ACCENT_BLUE, ACCENT_CYAN, GREEN, ORANGE]
        any_on = any(self.ch_states[ch])
        self.ch_leds[ch].set(ch_colors[ch] if any_on else None, on=any_on)

    def _update_all_led(self, m_idx):
        _, m_color = MODES[m_idx]
        all_on = all(self.ch_states[ch][m_idx]
                     for ch in range(self.NUM_CHANNELS))
        any_on = any(self.ch_states[ch][m_idx]
                     for ch in range(self.NUM_CHANNELS))
        color  = m_color if all_on else (YELLOW if any_on else None)
        self.all_mode_leds[m_idx].set(color, on=any_on)

    def _all_mode_toggle(self, m_idx):
        if not self._guard():
            return
        new_state = not all(self.ch_states[ch][m_idx]
                            for ch in range(self.NUM_CHANNELS))
        for ch in range(self.NUM_CHANNELS):
            self.ch_states[ch][m_idx] = new_state
            self._apply_ch_mode_ui(ch, m_idx, new_state)
            self._update_ch_led(ch)
        self._update_all_led(m_idx)
        self._set_toggle(self.all_toggle_btns[m_idx], new_state)
        self._update_status_bar()

    def _all_on(self):
        if not self._guard():
            return
        for ch in range(self.NUM_CHANNELS):
            for m_idx in range(len(MODES)):
                self.ch_states[ch][m_idx] = True
                self._apply_ch_mode_ui(ch, m_idx, True)
            self._update_ch_led(ch)
            self._set_toggle(self.ch_master_btns[ch], True)
        for m_idx in range(len(MODES)):
            self._update_all_led(m_idx)
            self._set_toggle(self.all_toggle_btns[m_idx], True)
        self._update_status_bar()

    def _all_off(self):
        if not self._guard():
            return
        for ch in range(self.NUM_CHANNELS):
            for m_idx in range(len(MODES)):
                self.ch_states[ch][m_idx] = False
                self._apply_ch_mode_ui(ch, m_idx, False)
            self._update_ch_led(ch)
            self._set_toggle(self.ch_master_btns[ch], False)
        for m_idx in range(len(MODES)):
            self._update_all_led(m_idx)
            self._set_toggle(self.all_toggle_btns[m_idx], False)
        self._update_status_bar()

    # ── Status bar ────────────────────────────────────────────────────────────
    def _update_status_bar(self):
        total  = self.NUM_CHANNELS * len(MODES)
        active = sum(self.ch_states[ch][m]
                     for ch in range(self.NUM_CHANNELS)
                     for m in range(len(MODES)))
        self.active_ch_var.set(f"{active} / {total}")

        if not self.initialized and self.conn_state != self.ST_CONNECTED:
            self.status_var.set("NOT INITIALIZED")
            self.status_lbl.configure(fg=RED)
            self.status_led.set(RED, on=True)
        elif self.conn_state == self.ST_CONNECTED and self.initialized:
            self.status_var.set("CONNECTED  —  OPERATIONAL")
            self.status_lbl.configure(fg=GREEN)
            self.status_led.set(GREEN, on=True)
        elif self.conn_state == self.ST_CONNECTED:
            self.status_var.set("CONNECTED  —  AWAITING INIT")
            self.status_lbl.configure(fg=YELLOW)
            self.status_led.set(YELLOW, on=True)
        elif self.conn_state in (self.ST_CONNECTING, self.ST_RECONNECTING):
            self.status_var.set(self.conn_state)
            self.status_lbl.configure(fg=YELLOW)
            self.status_led.set(YELLOW, on=True)
        else:
            self.status_var.set("DISCONNECTED")
            self.status_lbl.configure(fg=RED)
            self.status_led.set(RED, on=True)

    # ── Background worker ─────────────────────────────────────────────────────
    def _start_worker(self):
        t = threading.Thread(target=self._worker_loop, daemon=True)
        t.start()

    def _worker_loop(self):
        misses = 0
        while not self._worker_stop.is_set():
            state = self.conn_state

            if state == self.ST_DISCONNECTED:
                time.sleep(0.2)
                continue

            if state == self.ST_CONNECTING:
                if self.esp32.connect():
                    misses = 0
                    self.esp32.ping(timeout=0.5)
                    self.after(0, lambda: self._set_conn_state(self.ST_CONNECTED))
                else:
                    if not self._want_connected:
                        self.after(0, lambda: self._set_conn_state(
                            self.ST_DISCONNECTED))
                        continue
                    for _ in range(int(self.RECONNECT_DELAY * 10)):
                        if self._worker_stop.is_set():
                            return
                        time.sleep(0.1)
                continue

            if state == self.ST_CONNECTED:
                if not self.esp32.is_open:
                    self.after(0, self._on_link_lost)
                    continue

                result = self.esp32.ping()

                if result is None:
                    misses += 1
                    if misses >= self.MISS_THRESHOLD:
                        misses = 0
                        self.esp32.disconnect()
                        self.after(0, self._on_link_lost)
                        continue
                else:
                    misses = 0
                    if (self.initialized and result != "init"
                            and (time.time() - self._init_time) > 2.5):
                        self.after(0, self._on_device_reboot)

                for _ in range(int(self.PING_INTERVAL * 10)):
                    if self._worker_stop.is_set():
                        return
                    time.sleep(0.1)
                continue

            if state == self.ST_RECONNECTING:
                if not self._want_connected:
                    self.after(0, lambda: self._set_conn_state(
                        self.ST_DISCONNECTED))
                    continue
                if self.esp32.connect():
                    misses = 0
                    self.esp32.ping(timeout=0.5)
                    self.after(0, lambda: self._set_conn_state(self.ST_CONNECTED))
                else:
                    self.esp32.disconnect()
                    for _ in range(int(self.RECONNECT_DELAY * 10)):
                        if self._worker_stop.is_set():
                            return
                        time.sleep(0.1)
                continue

    def _reset_channel_ui(self):
        self.ch_states = [[False] * len(MODES) for _ in range(self.NUM_CHANNELS)]
        for ch in range(self.NUM_CHANNELS):
            for m_idx in range(len(MODES)):
                self.mode_leds[ch][m_idx].set(None, on=False)
                self._set_toggle(self.mode_toggle_btns[ch][m_idx], False)
            self._update_ch_led(ch)
            self._set_toggle(self.ch_master_btns[ch], False)
        for m_idx in range(len(MODES)):
            self._update_all_led(m_idx)
            self._set_toggle(self.all_toggle_btns[m_idx], False)

    def _on_link_lost(self):
        self.initialized = False
        self.init_led.set(on=False)
        self.init_btn.configure(fg=ACCENT_BLUE, highlightbackground=ACCENT_BLUE)
        self._reset_channel_ui()
        self._reset_readings()
        self._set_conn_state(self.ST_RECONNECTING)

    def _on_device_reboot(self):
        if not self.initialized:
            return
        self.initialized = False
        self.init_led.set(on=False)
        self.init_btn.configure(fg=ACCENT_BLUE, highlightbackground=ACCENT_BLUE)
        self._reset_channel_ui()
        self._reset_readings()
        self._update_status_bar()

    # ── Shutdown ──────────────────────────────────────────────────────────────
    def destroy(self):
        self._worker_stop.set()
        try:
            self.esp32.disconnect()
        except Exception:
            pass
        super().destroy()


if __name__ == "__main__":
    app = ECUSupplyController()
    app.mainloop()