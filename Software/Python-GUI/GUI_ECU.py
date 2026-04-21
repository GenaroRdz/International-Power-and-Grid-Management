import tkinter as tk
from tkinter import messagebox
import threading
import time
import serial
 
try:
    ser = serial.Serial('COM3', 115200, timeout=1)
    device_connected = True
except:
    print("not connected")
    device_connected = False
 
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
ORANGE      = "#ff8c00"
TEXT_SEC    = "#7a8899"
TEXT_DIM    = "#3d4a5a"
FONT_LABEL  = ("Courier New", 8, "bold")
FONT_TITLE  = ("Courier New", 18, "bold")
FONT_STATUS = ("Courier New", 9)
FONT_CH     = ("Courier New", 11, "bold")
FONT_BTN    = ("Courier New", 8, "bold")
FONT_MODE   = ("Courier New", 8, "bold")
 
# Mode definitions: (label, colour)
MODES = [
    ("BATTERY", ACCENT_BLUE),
    ("ACC",     ACCENT_CYAN),
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
 
 
# ── Main App ──────────────────────────────────────────────────────────────────
class ECUSupplyController(tk.Tk):
    NUM_CHANNELS = 4
 
    def __init__(self):
        super().__init__()
        self.title("ECU Supply Controller")
        self.configure(bg=BG)
        self.resizable(False, False)
 
        self.initialized  = False
        self.connected    = False
        # ch_states[ch][mode] = True/False
        self.ch_states    = [[False]*len(MODES) for _ in range(self.NUM_CHANNELS)]
        self._blink_after = None
        self._conn_blink  = False
 
        self._build_ui()
        self._update_status_bar()
        self._start_blink_loop()
 
    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        outer = tk.Frame(self, bg=BG, padx=18, pady=18)
        outer.pack()
 
        self._build_title(outer)
 
        main = tk.Frame(outer, bg=PANEL, bd=0,
                        highlightbackground=BORDER, highlightthickness=1)
        main.pack(fill="x", pady=(0, 12))
 
        pad = dict(padx=18, pady=14)
        self._build_init_row(main, pad)
        self._build_separator(main)
        self._build_channels(main, pad)
        self._build_separator(main)
        self._build_global_buttons(main, pad)
        self._build_status_bar(outer)
 
    def _build_title(self, parent):
        hdr = tk.Frame(parent, bg=BG)
        hdr.pack(fill="x", pady=(0, 14))
        tk.Label(hdr, text="⬡ ECU SUPPLY CONTROLLER",
                 font=FONT_TITLE, fg=ACCENT_BLUE, bg=BG).pack(side="left")
        tk.Label(hdr, text=" v1.1 ", font=FONT_LABEL,
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
        self.init_led.pack(side="left", padx=(0, 5))
        tk.Label(row, text="INITIALIZED", font=FONT_LABEL,
                 fg=TEXT_SEC, bg=PANEL).pack(side="left", padx=(0, 24))
 
        self.conn_led = LED(row, size=14, bg_color=PANEL)
        self.conn_led.pack(side="left", padx=(0, 5))
        self.conn_label = tk.Label(row, text="DEVICE NOT CONNECTED",
                                   font=FONT_LABEL, fg=TEXT_SEC, bg=PANEL)
        self.conn_label.pack(side="left")
 
    def _build_channels(self, parent, pad):
        tk.Label(parent, text="CHANNEL OUTPUTS",
                 font=FONT_LABEL, fg=TEXT_DIM, bg=PANEL).pack(
            anchor="w", padx=18, pady=(4, 0))
 
        grid = tk.Frame(parent, bg=PANEL)
        grid.pack(fill="x", **pad)
 
        ch_colors = [ACCENT_BLUE, ACCENT_CYAN, GREEN, ORANGE]
 
        # Storage
        self.ch_leds           = []
        self.mode_leds         = []   # [ch][mode] → LED
        self.mode_toggle_btns  = []   # [ch][mode] → toggle Button
        self.all_mode_leds     = []   # [mode] → LED
        self.all_toggle_btns   = []   # [mode] → toggle Button
 
        # ── ALL column (col 0) ────────────────────────────────────────────────
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
 
            # Status LED
            led = LED(row_f, size=10, bg_color=CARD)
            led.pack(side="left", padx=(0, 4))
            self.all_mode_leds.append(led)
 
            # Mode label
            tk.Label(row_f, text=m_label, font=FONT_MODE,
                     fg=m_color, bg=CARD, width=7,
                     anchor="w").pack(side="left", padx=(0, 6))
 
            # Single toggle button
            btn = self._make_toggle_btn(
                row_f, lambda mi=m_idx: self._all_mode_toggle(mi))
            btn.pack(side="right")
            self.all_toggle_btns.append(btn)
 
        # ── Per-channel columns (col 1..4) ────────────────────────────────────
        for ch in range(self.NUM_CHANNELS):
            col = ch + 1
            card = tk.Frame(grid, bg=CARD,
                            highlightbackground=BORDER, highlightthickness=1,
                            padx=10, pady=10)
            card.grid(row=0, column=col, padx=5, sticky="nsew")
            grid.columnconfigure(col, weight=1)
 
            # Channel header
            hdr = tk.Frame(card, bg=CARD)
            hdr.pack(fill="x", pady=(0, 6))
            ch_led = LED(hdr, size=12, bg_color=CARD)
            ch_led.pack(side="left", padx=(0, 5))
            self.ch_leds.append(ch_led)
            tk.Label(hdr, text=f"ECU {ch+1:02d}", font=FONT_CH,
                     fg=ch_colors[ch], bg=CARD).pack(side="left")
 
            ch_mode_leds      = []
            ch_mode_tog_btns  = []
 
            for m_idx, (m_label, m_color) in enumerate(MODES):
                row_f = tk.Frame(card, bg=CARD)
                row_f.pack(fill="x", pady=3)
 
                # Status LED
                led = LED(row_f, size=10, bg_color=CARD)
                led.pack(side="left", padx=(0, 4))
                ch_mode_leds.append(led)
 
                # Mode label
                tk.Label(row_f, text=m_label, font=FONT_MODE,
                         fg=m_color, bg=CARD, width=7,
                         anchor="w").pack(side="left", padx=(0, 6))
 
                # Single toggle button
                btn = self._make_toggle_btn(
                    row_f, lambda c=ch, mi=m_idx: self._ch_mode_toggle(c, mi))
                btn.pack(side="right")
                ch_mode_tog_btns.append(btn)
 
            self.mode_leds.append(ch_mode_leds)
            self.mode_toggle_btns.append(ch_mode_tog_btns)
 
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
        """Single ON/OFF toggle button. Starts showing OFF state."""
        btn = tk.Button(parent, text="OFF", command=cmd,
                        font=FONT_MODE, width=5,
                        fg=RED, bg=CARD,
                        activeforeground=BG, activebackground=GREEN,
                        relief="flat", bd=0,
                        highlightbackground=RED, highlightthickness=1,
                        cursor="hand2", pady=2)
        btn._is_on = False
 
        def _enter(e, b=btn):
            if str(b["state"]) == "disabled": return
            b.configure(bg=GREEN if b._is_on else RED, fg=BG)
 
        def _leave(e, b=btn):
            if str(b["state"]) == "disabled": return
            if b._is_on:
                b.configure(bg=GREEN, fg=BG)
            else:
                b.configure(bg=CARD, fg=RED)
 
        btn.bind("<Enter>", _enter)
        btn.bind("<Leave>", _leave)
        return btn
 
    def _set_toggle(self, btn, on):
        """Flip a toggle button's visual state."""
        btn._is_on = on
        if on:
            btn.configure(text="ON", fg=BG, bg=GREEN,
                          highlightbackground=GREEN)
        else:
            btn.configure(text="OFF", fg=RED, bg=CARD,
                          highlightbackground=RED)
 
    # ── Guard ─────────────────────────────────────────────────────────────────
    def _guard(self):
        if not self.initialized or not self.connected:
            messagebox.showwarning(
                "Device Not Initialized",
                "⚠  Device is not initialized or not connected.\n\n"
                "Please press INIT first.", parent=self)
            return False
        return True
 
    # ── Actions ───────────────────────────────────────────────────────────────
    def _do_init(self):
        if self.initialized:
            return
        self.init_btn.configure(state="disabled", text="  CONNECTING…")
        self.status_var.set("INITIALIZING…")
        self.status_lbl.configure(fg=YELLOW)
        self.status_led.set(YELLOW, on=True)
 
        def _connect():
            time.sleep(1.2)
            self.initialized = True
            self.connected   = True
            self.after(0, self._on_connected)
            if device_connected:
                ser.write("0\n".encode())
                time.sleep(1.2)
                ser.write("1\n".encode())
 
        threading.Thread(target=_connect, daemon=True).start()
 
    def _on_connected(self):
        self.init_btn.configure(state="normal", text="⚡  INIT",
                                fg=GREEN, highlightbackground=GREEN)
        self.init_led.set(GREEN, on=True)
        self.conn_led.set(GREEN, on=True)
        self.conn_label.configure(text="DEVICE CONNECTED", fg=GREEN)
        self._update_status_bar()
 
    def _do_reset(self):
        if not messagebox.askyesno("Reset", "Reset all channels and disconnect?",
                                   parent=self):
            return
        self.initialized = False
        self.connected   = False
        self.ch_states   = [[False]*len(MODES) for _ in range(self.NUM_CHANNELS)]
 
        for ch in range(self.NUM_CHANNELS):
            for m_idx in range(len(MODES)):
                self._apply_ch_mode_ui(ch, m_idx, False)
            self._update_ch_led(ch)
        for m_idx in range(len(MODES)):
            self._update_all_led(m_idx)
            self._set_toggle(self.all_toggle_btns[m_idx], False)
 
        self.init_led.set(on=False)
        self.conn_led.set(on=False)
        self.conn_label.configure(text="DEVICE NOT CONNECTED", fg=TEXT_SEC)
        self.init_btn.configure(state="normal", text="⚡  INIT",
                                fg=ACCENT_BLUE, highlightbackground=ACCENT_BLUE)
        self._update_status_bar()
 
    # ── Per-channel mode toggle ───────────────────────────────────────────────
    def _ch_mode_toggle(self, ch, m_idx):
        if not self._guard():
            return
        new_state = not self.ch_states[ch][m_idx]
        self.ch_states[ch][m_idx] = new_state
        self._apply_ch_mode_ui(ch, m_idx, new_state)
        self._update_ch_led(ch)
        self._update_all_led(m_idx)
        self._update_status_bar()
 
    def _apply_ch_mode_ui(self, ch, m_idx, on):
        _, m_color = MODES[m_idx]
        self.mode_leds[ch][m_idx].set(m_color if on else None, on=on)
        self._set_toggle(self.mode_toggle_btns[ch][m_idx], on)
 
    def _update_ch_led(self, ch):
        ch_colors = [ACCENT_BLUE, ACCENT_CYAN, GREEN, ORANGE]
        any_on = any(self.ch_states[ch])
        self.ch_leds[ch].set(ch_colors[ch] if any_on else None, on=any_on)
 
    def _update_all_led(self, m_idx):
        """ALL LED: mode-colour if all on, yellow if partial, off if none."""
        _, m_color = MODES[m_idx]
        all_on = all(self.ch_states[ch][m_idx] for ch in range(self.NUM_CHANNELS))
        any_on = any(self.ch_states[ch][m_idx] for ch in range(self.NUM_CHANNELS))
        color  = m_color if all_on else (YELLOW if any_on else None)
        self.all_mode_leds[m_idx].set(color, on=any_on)
 
    # ── ALL column actions ────────────────────────────────────────────────────
    def _all_mode_toggle(self, m_idx):
        if not self._guard():
            return
        # If all are on → turn off; otherwise turn all on
        new_state = not all(self.ch_states[ch][m_idx] for ch in range(self.NUM_CHANNELS))
        for ch in range(self.NUM_CHANNELS):
            self.ch_states[ch][m_idx] = new_state
            self._apply_ch_mode_ui(ch, m_idx, new_state)
            self._update_ch_led(ch)
        self._update_all_led(m_idx)
        self._set_toggle(self.all_toggle_btns[m_idx], new_state)
        self._update_status_bar()
 
    # ── Global ALL ON ─────────────────────────────────────────────────────────
    def _all_on(self):
        if not self._guard():
            return
        for ch in range(self.NUM_CHANNELS):
            for m_idx in range(len(MODES)):
                self.ch_states[ch][m_idx] = True
                self._apply_ch_mode_ui(ch, m_idx, True)
            self._update_ch_led(ch)
        for m_idx in range(len(MODES)):
            self._update_all_led(m_idx)
            self._set_toggle(self.all_toggle_btns[m_idx], True)
        self._update_status_bar()
 
    # ── Global ALL OFF ────────────────────────────────────────────────────────
    def _all_off(self):
        if not self._guard():
            return
        for ch in range(self.NUM_CHANNELS):
            for m_idx in range(len(MODES)):
                self.ch_states[ch][m_idx] = False
                self._apply_ch_mode_ui(ch, m_idx, False)
            self._update_ch_led(ch)
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
 
        if not self.initialized:
            self.status_var.set("NOT INITIALIZED")
            self.status_lbl.configure(fg=RED)
            self.status_led.set(RED, on=True)
        elif self.connected:
            self.status_var.set("CONNECTED  —  OPERATIONAL")
            self.status_lbl.configure(fg=GREEN)
            self.status_led.set(GREEN, on=True)
        else:
            self.status_var.set("DISCONNECTED")
            self.status_lbl.configure(fg=RED)
            self.status_led.set(RED, on=True)
 
    # ── Blink loop ────────────────────────────────────────────────────────────
    def _start_blink_loop(self):
        def _blink():
            if not self.connected:
                self._conn_blink = not self._conn_blink
                self.conn_led.set(YELLOW if self._conn_blink else None,
                                  on=self._conn_blink)
                self.status_led.set(YELLOW if self._conn_blink else None,
                                    on=self._conn_blink)
            self._blink_after = self.after(600, _blink)
        _blink()
 
    def destroy(self):
        if self._blink_after:
            self.after_cancel(self._blink_after)
        super().destroy()
 
 
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = ECUSupplyController()
    app.mainloop()