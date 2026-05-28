import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
import threading
import time
from New_device_connection_2 import ESP32Connection
from Scope_tab import ScopeTab

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
FONT_LABEL  = ("Courier New", 8, "bold")
FONT_TITLE  = ("Courier New", 18, "bold")
FONT_STATUS = ("Courier New", 9)
FONT_CH     = ("Courier New", 11, "bold")
FONT_BTN    = ("Courier New", 8, "bold")
FONT_MODE   = ("Courier New", 8, "bold")

MODES = [
    ("BATTERY", ACCENT_BLUE),
    ("ACC",     ACCENT_CYAN),
    ("IGN",     LILA),
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
        self.resizable(False, False)

        self.esp32 = ESP32Connection()
        self.conn_state      = self.ST_DISCONNECTED
        self._want_connected = False
        self._worker_stop    = threading.Event()

        self.initialized = False
        self.connected   = False

        self.ch_states = [[False] * len(MODES) for _ in range(self.NUM_CHANNELS)]

        self._build_ui()
        self._update_status_bar()
        self._start_worker()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        outer = tk.Frame(self, bg=BG, padx=18, pady=18)
        outer.pack()

        self._build_title(outer)

        # ── CAMBIO 1: Notebook con dos pestañas ───────────────────────────────
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Dark.TNotebook",
                        background=BG, borderwidth=0)
        style.configure("Dark.TNotebook.Tab",
                        background=CARD, foreground=TEXT_SEC,
                        font=("Courier New", 8, "bold"),
                        padding=[14, 6])
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", PANEL)],
                  foreground=[("selected", ACCENT_BLUE)])

        nb = ttk.Notebook(outer, style="Dark.TNotebook")
        nb.pack(fill="both", expand=True, pady=(0, 12))

        # Pestaña 1: Control (todo el contenido original va aquí)
        ctrl = tk.Frame(nb, bg=BG)
        nb.add(ctrl, text="  ⬡  CONTROL  ")

        # Pestaña 2: Osciloscopio
        self._scope_tab = ScopeTab(nb)
        nb.add(self._scope_tab, text="  📡  LIVE SCOPE  ")
        # ─────────────────────────────────────────────────────────────────────

        main = tk.Frame(ctrl, bg=PANEL, bd=0,
                        highlightbackground=BORDER, highlightthickness=1)
        main.pack(fill="x", pady=(0, 12))

        pad = dict(padx=18, pady=14)
        self._build_init_row(main, pad)
        self._build_separator(main)
        self._build_channels(main, pad)
        self._build_separator(main)
        self._build_readings(main, pad)
        self._build_separator(main)
        self._build_global_buttons(main, pad)
        self._build_status_bar(ctrl)

    def _build_title(self, parent):
        hdr = tk.Frame(parent, bg=BG)
        hdr.pack(fill="x", pady=(0, 14))
        tk.Label(hdr, text="⬡ ECU SUPPLY CONTROLLER",
                 font=FONT_TITLE, fg=ACCENT_BLUE, bg=BG).pack(side="left")
        tk.Label(hdr, text=" v2.1 ", font=FONT_LABEL,
                 fg=ACCENT_CYAN, bg=BG, relief="solid", bd=1,
                 highlightbackground=ACCENT_CYAN,
                 highlightthickness=1).pack(side="right", padx=4, pady=6)

    def _build_init_row(self, parent, pad):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", **pad)

        self.init_btn = self._make_btn(row, "⚡  INIT", ACCENT_BLUE,
                                       self._do_init, width=12)
        self.init_btn.pack(side="left", padx=(0, 14))

        self.init_led = LED(row, size=14, bg_color=PANEL)
        self.init_led.pack(side="left", padx=(0, 2))

        tk.Label(row, text="INITIALIZED", font=FONT_LABEL,
                 fg=TEXT_SEC, bg=PANEL).pack(side="left", padx=(0, 24))

        self.conn_led = LED(row, size=14, bg_color=PANEL)
        self.conn_led.pack(side="left", padx=(0, 5))
        self.conn_label = tk.Label(row, text="DEVICE NOT CONNECTED",
                                   font=FONT_LABEL, fg=TEXT_SEC, bg=PANEL)
        self.conn_label.pack(side="left")

        self.connect_button = self._make_btn(row, "Connect", GREEN,
                                             self._connect, width=14)
        self.connect_button.pack(side="left", padx=(100, 8))

        self.disconnect_button = self._make_btn(row, "Disconnect", RED,
                                                self._disconnect, width=14)
        self.disconnect_button.pack(side="left", padx=(5, 8))

    def _build_channels(self, parent, pad):
        tk.Label(parent, text="CHANNEL OUTPUTS",
                 font=FONT_LABEL, fg=TEXT_DIM, bg=PANEL).pack(
            anchor="w", padx=18, pady=(4, 0))

        grid = tk.Frame(parent, bg=PANEL)
        grid.pack(fill="x", **pad)

        ch_colors = [ACCENT_BLUE, ACCENT_CYAN, GREEN, ORANGE]

        self.ch_leds           = []
        self.ch_master_btns    = []
        self.mode_leds         = []
        self.mode_toggle_btns  = []
        self.all_mode_leds     = []
        self.all_toggle_btns   = []

        all_card = tk.Frame(grid, bg=CARD,
                            highlightbackground=BORDER, highlightthickness=1,
                            padx=10, pady=10)
        all_card.grid(row=0, column=0, padx=5, sticky="nsew")
        grid.columnconfigure(0, weight=1)

        tk.Label(all_card, text="ALL", font=FONT_CH,
                 fg=GREEN, bg=CARD).pack(pady=(0, 6))

        for m_idx, (m_label, m_color) in enumerate(MODES):
            row_f = tk.Frame(all_card, bg=CARD)
            row_f.pack(fill="x", pady=3)

            led = LED(row_f, size=10, bg_color=CARD)
            led.pack(side="left", padx=(0, 4))
            self.all_mode_leds.append(led)

            tk.Label(row_f, text=m_label, font=FONT_MODE,
                     fg=m_color, bg=CARD, width=7,
                     anchor="w").pack(side="left", padx=(0, 6))

            btn = self._make_toggle_btn(
                row_f, lambda mi=m_idx: self._all_mode_toggle(mi))
            btn.pack(side="right")
            self.all_toggle_btns.append(btn)

        for ch in range(self.NUM_CHANNELS):
            col = ch + 1
            card = tk.Frame(grid, bg=CARD,
                            highlightbackground=BORDER, highlightthickness=1,
                            padx=10, pady=10)
            card.grid(row=0, column=col, padx=5, sticky="nsew")
            grid.columnconfigure(col, weight=1)

            hdr = tk.Frame(card, bg=CARD)
            hdr.pack(fill="x", pady=(0, 6))
            ch_led = LED(hdr, size=12, bg_color=CARD)
            ch_led.pack(side="left", padx=(0, 5))
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
                row_f.pack(fill="x", pady=3)

                led = LED(row_f, size=10, bg_color=CARD)
                led.pack(side="left", padx=(0, 4))
                ch_mode_leds.append(led)

                tk.Label(row_f, text=m_label, font=FONT_MODE,
                         fg=m_color, bg=CARD, width=7,
                         anchor="w").pack(side="left", padx=(0, 6))

                btn = self._make_toggle_btn(
                    row_f, lambda c=ch, mi=m_idx: self._ch_mode_toggle(c, mi))
                btn.pack(side="right")
                ch_mode_tog_btns.append(btn)

            self.mode_leds.append(ch_mode_leds)
            self.mode_toggle_btns.append(ch_mode_tog_btns)

    # ── Readings row (Voltage / Current / Power) ──────────────────────────────
    def _build_readings(self, parent, pad):
        tk.Label(parent, text="CHANNEL READINGS",
                 font=FONT_LABEL, fg=TEXT_DIM, bg=PANEL).pack(
            anchor="w", padx=18, pady=(4, 0))

        grid = tk.Frame(parent, bg=PANEL)
        grid.pack(fill="x", **pad)

        ch_colors = [ACCENT_BLUE, ACCENT_CYAN, GREEN, ORANGE]

        self.voltage_vars = []
        self.current_vars = []
        self.power_vars   = []

        all_card = tk.Frame(grid, bg=CARD,
                            highlightbackground=BORDER, highlightthickness=1,
                            padx=10, pady=10)
        all_card.grid(row=0, column=0, padx=5, sticky="nsew")
        grid.columnconfigure(0, weight=1)

        tk.Label(all_card, text="ALL  Σ", font=FONT_CH,
                 fg=GREEN, bg=CARD).pack(pady=(0, 6))

        self.all_voltage_var = tk.StringVar(value="0.00 V")
        self.all_current_var = tk.StringVar(value="0.00 A")
        self.all_power_var   = tk.StringVar(value="0.00 W")

        self._build_reading_row(all_card, "VOLT", ACCENT_BLUE, self.all_voltage_var)
        self._build_reading_row(all_card, "CURR", ACCENT_CYAN, self.all_current_var)
        self._build_reading_row(all_card, "PWR",  LILA,        self.all_power_var)

        for ch in range(self.NUM_CHANNELS):
            col = ch + 1
            card = tk.Frame(grid, bg=CARD,
                            highlightbackground=BORDER, highlightthickness=1,
                            padx=10, pady=10)
            card.grid(row=0, column=col, padx=5, sticky="nsew")
            grid.columnconfigure(col, weight=1)

            tk.Label(card, text=f"CH {ch+1:02d}", font=FONT_CH,
                     fg=ch_colors[ch], bg=CARD).pack(pady=(0, 6))

            v_var = tk.StringVar(value="0.00 V")
            c_var = tk.StringVar(value="0.00 A")
            p_var = tk.StringVar(value="0.00 W")

            self.voltage_vars.append(v_var)
            self.current_vars.append(c_var)
            self.power_vars.append(p_var)

            self._build_reading_row(card, "VOLT", ACCENT_BLUE, v_var)
            self._build_reading_row(card, "CURR", ACCENT_CYAN, c_var)
            self._build_reading_row(card, "PWR",  LILA,        p_var)

    def _build_reading_row(self, parent, label, color, var):
        row_f = tk.Frame(parent, bg=CARD)
        row_f.pack(fill="x", pady=3)

        LED(row_f, size=10, bg_color=CARD).pack(side="left", padx=(0, 4))

        tk.Label(row_f, text=label, font=FONT_MODE,
                 fg=color, bg=CARD, width=7,
                 anchor="w").pack(side="left", padx=(0, 6))

        tk.Label(row_f, textvariable=var, font=FONT_MODE,
                 fg=TEXT_SEC, bg=CARD, anchor="e").pack(side="right")

    def update_reading(self, ch, voltage, current, power):
        """Call this (via self.after(0, ...) from worker threads) when new
        measurements arrive for a specific channel."""
        self.voltage_vars[ch].set(f"{voltage:.2f} V")
        self.current_vars[ch].set(f"{current:.2f} A")
        self.power_vars[ch].set(f"{power:.2f} W")
        self._refresh_all_readings()
        # ── CAMBIO 2: push datos al scope ─────────────────────────────────────
        self._scope_tab.push(ch, voltage, current, power)

    def _refresh_all_readings(self):
        total_v = total_c = total_p = 0.0
        for ch in range(self.NUM_CHANNELS):
            try:
                total_v += float(self.voltage_vars[ch].get().split()[0])
                total_c += float(self.current_vars[ch].get().split()[0])
                total_p += float(self.power_vars[ch].get().split()[0])
            except (ValueError, IndexError):
                pass
        self.all_voltage_var.set(f"{total_v:.2f} V")
        self.all_current_var.set(f"{total_c:.2f} A")
        self.all_power_var.set(f"{total_p:.2f} W")

    def _reset_readings(self):
        for ch in range(self.NUM_CHANNELS):
            self.voltage_vars[ch].set("0.00 V")
            self.current_vars[ch].set("0.00 A")
            self.power_vars[ch].set("0.00 W")
        self._refresh_all_readings()

    def _build_global_buttons(self, parent, pad):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", **pad)

        self.all_on_btn = self._make_btn(row, "▶  ALL ON", GREEN,
                                         self._all_on, width=14)
        self.all_on_btn.pack(side="left", padx=(0, 8))

        self.all_off_btn = self._make_btn(row, "■  ALL OFF", RED,
                                          self._all_off, width=14)
        self.all_off_btn.pack(side="left", padx=(0, 8))

        tk.Frame(row, bg=PANEL).pack(side="left", expand=True)

        self.reset_btn = self._make_btn(row, "↺  RESET", YELLOW,
                                        self._do_reset, width=12)
        self.reset_btn.pack(side="right")

    def _build_status_bar(self, parent):
        bar = tk.Frame(parent, bg="#0a0c0f",
                       highlightbackground=BORDER, highlightthickness=1)
        bar.pack(fill="x")
        inner = tk.Frame(bar, bg="#0a0c0f", padx=14, pady=8)
        inner.pack(fill="x")

        tk.Label(inner, text="⬡", font=("Courier New", 11),
                 fg=ACCENT_BLUE, bg="#0a0c0f").pack(side="left", padx=(0, 6))
        tk.Label(inner, text="MICROCHIP STATUS:", font=FONT_STATUS,
                 fg=TEXT_DIM, bg="#0a0c0f").pack(side="left")

        self.status_led = LED(inner, size=10, bg_color="#0a0c0f")
        self.status_led.pack(side="left", padx=8)

        self.status_var = tk.StringVar(value="NOT INITIALIZED")
        self.status_lbl = tk.Label(inner, textvariable=self.status_var,
                                   font=FONT_STATUS, fg=RED, bg="#0a0c0f")
        self.status_lbl.pack(side="left")

        tk.Label(inner, text="ACTIVE OUTPUTS:", font=FONT_STATUS,
                 fg=TEXT_DIM, bg="#0a0c0f").pack(side="right", padx=(0, 4))
        self.active_ch_var = tk.StringVar(value="0 / 12")
        tk.Label(inner, textvariable=self.active_ch_var,
                 font=FONT_STATUS, fg=TEXT_SEC, bg="#0a0c0f").pack(side="right")

    def _build_separator(self, parent):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=14)

    # ── Button factories ──────────────────────────────────────────────────────
    def _make_btn(self, parent, text, color, cmd, width=10):
        btn = tk.Button(parent, text=text, command=cmd,
                        font=FONT_BTN, width=width,
                        fg=color, bg=CARD,
                        activeforeground=BG, activebackground=color,
                        relief="flat", bd=0,
                        highlightbackground=color, highlightthickness=1,
                        cursor="hand2", pady=6)
        btn.bind("<Enter>", lambda e, b=btn, c=color: b.configure(bg=c, fg=BG))
        btn.bind("<Leave>", lambda e, b=btn, c=color: (
            b.configure(bg=CARD, fg=c) if str(b["state"]) != "disabled" else None))
        return btn

    def _make_toggle_btn(self, parent, cmd):
        btn = tk.Button(parent, text="OFF", command=cmd,
                        font=FONT_MODE, width=5,
                        fg=RED, bg=CARD,
                        activeforeground=BG, activebackground=GREEN,
                        relief="flat", bd=0,
                        highlightbackground=RED, highlightthickness=1,
                        cursor="hand2", pady=2)
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
        self.init_led.set(GREEN, on=True)
        self.init_btn.configure(fg=GREEN, highlightbackground=GREEN)
        self.esp32.send("1")
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
        """Single place that updates connection UI. Main thread only."""
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
        # ── CAMBIO 3: notifica estado de conexión al scope ────────────────────
        self._scope_tab.set_connected(self.connected)

    # ── Reset ─────────────────────────────────────────────────────────────────
    def _do_reset(self):
        if not messagebox.askyesno("Reset",
                                   "Reset all channels and disconnect?",
                                   parent=self):
            return
        self._shutdown_link()

    def _shutdown_link(self):
        """Send *RST, close port, clear all GUI state. Safe to call anytime."""
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

        if ch == 0 and m_idx == 0 and on:
            print("CH1 BATTERY is ON")
            self.esp32.send("BAT1 on")
        if ch == 0 and m_idx == 0 and not on:
            print("CH1 BATTERY is OFF")
            self.esp32.send("BAT1 off")

        if ch == 0 and m_idx == 1 and on:
            print("CH1 ACC is ON")
            self.esp32.send("ACC1 on")
        if ch == 0 and m_idx == 1 and not on:
            self.esp32.send("ACC1 off")

        if ch == 0 and m_idx == 2 and on:
            print("CH1 IGN is ON")
            self.esp32.send("IGN1 on")
        if ch == 0 and m_idx == 2 and not on:
            self.esp32.send("IGN1 off")

        if ch == 1 and m_idx == 0 and on:
            print("CH2 BATTERY is on")
        if ch == 1 and m_idx == 0 and not on:
            print("CH2 BATTERY is off")
            self.esp32.send("IGN1 off")

        if ch == 1 and m_idx == 1 and on:
            print("CH2 ACC is ON")
        if ch == 1 and m_idx == 1 and not on:
            print("CH2 ACC is OFF")

        if ch == 1 and m_idx == 2 and on:
            print("CH2 IGN is ON")
        if ch == 1 and m_idx == 2 and not on:
            print("CH2 IGN is OFF")

        if ch == 2 and m_idx == 0 and on:
            print("CH3 BATTERY is OFF")
        if ch == 2 and m_idx == 0 and not on:
            print("CH3 BATTERY is OFF")

        if ch == 2 and m_idx == 1 and on:
            print("CH3 ACC is ON")
        if ch == 2 and m_idx == 1 and not on:
            print("CH3 ACC is OFF")

        if ch == 2 and m_idx == 2 and on:
            print("CH3 IGN is ON")
        if ch == 2 and m_idx == 2 and not on:
            print("CH3 IGN is OFF")

        if ch == 3 and m_idx == 0 and on:
            print("CH4 BATTERY is ON")
        if ch == 3 and m_idx == 0 and not on:
            print("CH4 BATTERY is OFF")

        if ch == 3 and m_idx == 1 and on:
            print("CH4 ACC is ON")
        if ch == 3 and m_idx == 1 and not on:
            print("CH4 ACC is OFF")

        if ch == 3 and m_idx == 2 and on:
            print("CH4 IGN is ON")
        if ch == 3 and m_idx == 2 and not on:
            print("CH4 IGN is OFF")

    def _update_ch_led(self, ch):
        ch_colors = [ACCENT_BLUE, ACCENT_CYAN, GREEN, ORANGE]
        any_on = any(self.ch_states[ch])
        self.ch_leds[ch].set(ch_colors[ch] if any_on else None, on=any_on)

    def _update_all_led(self, m_idx):
        _, m_color = MODES[m_idx]
        all_on = all(self.ch_states[ch][m_idx] for ch in range(self.NUM_CHANNELS))
        any_on = any(self.ch_states[ch][m_idx] for ch in range(self.NUM_CHANNELS))
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
                        self.after(0, lambda: self._set_conn_state(self.ST_DISCONNECTED))
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

                if result == "init":
                    misses = 0
                elif result == "noinit":
                    misses = 0
                    if self.initialized:
                        self.after(0, self._on_link_lost)
                        continue
                elif result == "boot":
                    misses = 0
                    if self.initialized:
                        self.after(0, self._on_link_lost)
                        continue
                else:
                    misses += 1
                    if misses >= self.MISS_THRESHOLD:
                        misses = 0
                        self.esp32.disconnect()
                        self.after(0, self._on_link_lost)
                        continue

                for _ in range(int(self.PING_INTERVAL * 10)):
                    if self._worker_stop.is_set():
                        return
                    time.sleep(0.1)
                continue

            if state == self.ST_RECONNECTING:
                if not self._want_connected:
                    self.after(0, lambda: self._set_conn_state(self.ST_DISCONNECTED))
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

    def _on_link_lost(self):
        self.initialized = False
        self.init_led.set(on=False)
        self.init_btn.configure(fg=ACCENT_BLUE, highlightbackground=ACCENT_BLUE)
        self._reset_readings()
        self._set_conn_state(self.ST_RECONNECTING)

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
