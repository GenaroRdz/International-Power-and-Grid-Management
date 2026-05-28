import tkinter as tk
import collections

# ── Colours (mismo esquema que el app principal) ──────────────────────────────
BG       = "#0d0f12"
PANEL    = "#161a1f"
CARD     = "#1c2128"
BORDER   = "#2a3040"
GRID_C   = "#1e2530"
TEXT_SEC = "#7a8899"
TEXT_DIM = "#3d4a5a"

CH_COLORS = ["#00aaff", "#00e5cc", "#00e676", "#ff8c00"]
PWR_COLOR = "#B565FC"   # lila — igual al LILA del app principal

FONT_LABEL = ("Courier New", 8,  "bold")
FONT_CH    = ("Courier New", 11, "bold")
FONT_VAL   = ("Courier New", 13, "bold")
FONT_SMALL = ("Courier New", 9)

NUM_CH   = 4
BUF_SIZE = 200   # muestras en el buffer circular por canal


class ScopeTab(tk.Frame):
    """
    Pestaña de osciloscopio en vivo.
    Se instancia como:
        scope = ScopeTab(notebook)
        notebook.add(scope, text="  📡  LIVE SCOPE  ")

    API pública:
        scope.push(ch, voltage, current, power)   ← llamar desde update_reading()
        scope.set_connected(bool)                 ← llamar desde _set_conn_state()
    """

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)

        self._paused    = False
        self._show_v    = True
        self._show_a    = True
        self._show_p    = False   # power oculto por defecto; se puede activar
        self._connected = False

        # Buffers circulares por canal
        self.volt_buf = [collections.deque([0.0] * BUF_SIZE, maxlen=BUF_SIZE)
                         for _ in range(NUM_CH)]
        self.amp_buf  = [collections.deque([0.0] * BUF_SIZE, maxlen=BUF_SIZE)
                         for _ in range(NUM_CH)]
        self.pwr_buf  = [collections.deque([0.0] * BUF_SIZE, maxlen=BUF_SIZE)
                         for _ in range(NUM_CH)]

        self._last_v = [0.0] * NUM_CH
        self._last_a = [0.0] * NUM_CH
        self._last_p = [0.0] * NUM_CH

        self._build_ui()
        self._schedule_draw()

    # ── API pública ───────────────────────────────────────────────────────────
    def push(self, ch: int, voltage: float, current: float, power: float):
        """
        Recibe una nueva muestra para el canal ch.
        Llamar desde update_reading() en el hilo principal de tkinter.
        """
        if self._paused or not (0 <= ch < NUM_CH):
            return
        self.volt_buf[ch].append(voltage)
        self.amp_buf[ch].append(current)
        self.pwr_buf[ch].append(power)
        self._last_v[ch] = voltage
        self._last_a[ch] = current
        self._last_p[ch] = power

    def set_connected(self, connected: bool):
        """
        Actualiza el indicador STREAMING / OFFLINE.
        Llamar desde _set_conn_state() en el app principal.
        """
        self._connected = connected
        if not self._paused:
            color = "#00e676" if connected else "#ff1744"
            text  = "● STREAMING" if connected else "● OFFLINE"
            self._live_lbl.configure(text=text, fg=color)

    # ── Construcción de UI ────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Toolbar ──────────────────────────────────────────────────────────
        bar = tk.Frame(self, bg=BG, pady=8, padx=14)
        bar.pack(fill="x")

        tk.Label(bar, text="LIVE SCOPE", font=FONT_LABEL,
                 fg="#00aaff", bg=BG).pack(side="left", padx=(0, 16))

        self._btn_v = self._pill(bar, "VOLTAGE", self._toggle_v,
                                 active=True, color="#00aaff")
        self._btn_v.pack(side="left", padx=3)

        self._btn_a = self._pill(bar, "CURRENT", self._toggle_a,
                                 active=True, color="#00e5cc")
        self._btn_a.pack(side="left", padx=3)

        self._btn_p = self._pill(bar, "POWER", self._toggle_p,
                                 active=False, color=PWR_COLOR)
        self._btn_p.pack(side="left", padx=3)

        self._btn_pause = self._pill(bar, "⏸  PAUSE", self._toggle_pause,
                                     active=False, color="#ffd600")
        self._btn_pause.pack(side="left", padx=(12, 3))

        tk.Frame(bar, bg=BG).pack(side="left", expand=True)

        self._live_lbl = tk.Label(bar, text="● OFFLINE",
                                  font=FONT_LABEL, fg="#ff1744", bg=BG)
        self._live_lbl.pack(side="right", padx=4)
        self._blink_live()

        # ── Grid 2×2 de canales ───────────────────────────────────────────────
        grid = tk.Frame(self, bg=BG)
        grid.pack(fill="both", expand=True, padx=14, pady=(0, 6))

        self._canvases = []
        self._lbl_v    = []
        self._lbl_a    = []
        self._lbl_p    = []

        for ch in range(NUM_CH):
            row, col = divmod(ch, 2)
            card = tk.Frame(grid, bg=PANEL,
                            highlightbackground=BORDER, highlightthickness=1)
            card.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            grid.columnconfigure(col, weight=1)
            grid.rowconfigure(row, weight=1)

            # Header del canal
            hdr = tk.Frame(card, bg=PANEL, pady=5, padx=10)
            hdr.pack(fill="x")

            tk.Label(hdr, text=f"CH {ch+1:02d}", font=FONT_CH,
                     fg=CH_COLORS[ch], bg=PANEL).pack(side="left")

            rf = tk.Frame(hdr, bg=PANEL)
            rf.pack(side="right")

            lv = self._stat_label(rf, "V", "#00aaff")
            la = self._stat_label(rf, "A", "#00e5cc")
            lp = self._stat_label(rf, "W", PWR_COLOR)

            self._lbl_v.append(lv)
            self._lbl_a.append(la)
            self._lbl_p.append(lp)

            # Canvas del osciloscopio
            cv = tk.Canvas(card, bg=CARD, highlightthickness=0, height=110)
            cv.pack(fill="x", padx=2, pady=(0, 4))
            self._canvases.append(cv)

        # ── Leyenda inferior ──────────────────────────────────────────────────
        leg = tk.Frame(self, bg=BG, padx=14, pady=4)
        leg.pack(fill="x")

        self._legend_item(leg, "#00aaff", "voltage (V)  —— solid")
        self._legend_item(leg, "#00e5cc", "current (A)  - - dashed")
        self._legend_item(leg, PWR_COLOR,  "power (W)  ···· dotted")

        tk.Label(leg, text=f"BUFFER  {BUF_SIZE} pts",
                 font=FONT_LABEL, fg=TEXT_DIM, bg=BG).pack(side="right")

    def _stat_label(self, parent, unit, color):
        """Crea un par 'unidad + valor' en el header del canal."""
        tk.Label(parent, text=unit, font=FONT_LABEL,
                 fg=TEXT_DIM, bg=PANEL).pack(side="left", padx=(0, 2))
        lbl = tk.Label(parent, text=" 0.00", font=FONT_VAL,
                       fg=color, bg=PANEL, width=6, anchor="e")
        lbl.pack(side="left", padx=(0, 10))
        return lbl

    def _legend_item(self, parent, color, label):
        f = tk.Frame(parent, bg=BG)
        f.pack(side="left", padx=(0, 16))
        tk.Canvas(f, width=20, height=2, bg=color,
                  highlightthickness=0).pack(side="left", padx=(0, 5))
        tk.Label(f, text=label, font=FONT_SMALL,
                 fg=TEXT_SEC, bg=BG).pack(side="left")

    # ── Pill button ───────────────────────────────────────────────────────────
    def _pill(self, parent, text, cmd, active=False, color="#00aaff"):
        btn = tk.Button(parent, text=text,
                        font=FONT_LABEL, pady=3, padx=8,
                        relief="flat", bd=0, cursor="hand2",
                        highlightbackground=color, highlightthickness=1)
        btn._active = active
        btn._color  = color
        btn._cmd    = cmd

        def _refresh(b=btn):
            b.configure(fg=BG if b._active else b._color,
                        bg=b._color if b._active else CARD)

        def _click(b=btn):
            b._cmd()
            _refresh()

        btn._refresh = _refresh
        btn.configure(command=_click)
        _refresh()
        return btn

    # ── Controles toolbar ─────────────────────────────────────────────────────
    def _toggle_v(self):
        self._show_v = not self._show_v
        self._btn_v._active = self._show_v

    def _toggle_a(self):
        self._show_a = not self._show_a
        self._btn_a._active = self._show_a

    def _toggle_p(self):
        self._show_p = not self._show_p
        self._btn_p._active = self._show_p

    def _toggle_pause(self):
        self._paused = not self._paused
        self._btn_pause._active = self._paused
        self._btn_pause.configure(
            text="▶  RESUME" if self._paused else "⏸  PAUSE")
        self._btn_pause._refresh()

        if self._paused:
            self._live_lbl.configure(text="● PAUSED", fg="#ffd600")
        else:
            self.set_connected(self._connected)   # restaura el label correcto

    # ── Blink del indicador STREAMING ─────────────────────────────────────────
    def _blink_live(self):
        if self._connected and not self._paused:
            cur = self._live_lbl.cget("fg")
            self._live_lbl.configure(
                fg="#00e676" if cur != "#00e676" else BG)
        self.after(600, self._blink_live)

    # ── Loop de dibujado (~20 fps) ────────────────────────────────────────────
    def _schedule_draw(self):
        self._redraw_all()
        self.after(50, self._schedule_draw)

    def _redraw_all(self):
        for ch in range(NUM_CH):
            self._draw_channel(ch)
            self._lbl_v[ch].configure(text=f"{self._last_v[ch]:6.2f}")
            self._lbl_a[ch].configure(text=f"{self._last_a[ch]:6.2f}")
            self._lbl_p[ch].configure(text=f"{self._last_p[ch]:6.2f}")

    def _draw_channel(self, ch):
        cv = self._canvases[ch]
        W  = cv.winfo_width()
        H  = cv.winfo_height()
        if W < 2 or H < 2:
            return

        cv.delete("all")

        # Grid de fondo
        for i in range(1, 6):
            cv.create_line(int(W * i / 6), 0, int(W * i / 6), H,
                           fill=GRID_C, width=1)
        for i in range(1, 3):
            cv.create_line(0, int(H * i / 3), W, int(H * i / 3),
                           fill=GRID_C, width=1)

        # Señales
        if self._show_v:
            self._draw_signal(cv, list(self.volt_buf[ch]),
                              CH_COLORS[ch], W, H, style="solid")
        if self._show_a:
            self._draw_signal(cv, list(self.amp_buf[ch]),
                              "#00e5cc", W, H, style="dashed")
        if self._show_p:
            self._draw_signal(cv, list(self.pwr_buf[ch]),
                              PWR_COLOR, W, H, style="dotted")

    def _draw_signal(self, cv, data, color, W, H, style="solid"):
        if len(data) < 2:
            return

        lo   = min(data)
        hi   = max(data)
        span = hi - lo or 0.01
        pad  = H * 0.10
        eff  = H - 2 * pad
        n    = len(data)

        # Pre-calcular puntos
        pts = []
        for i, v in enumerate(data):
            pts.append(int(i / (n - 1) * W))
            pts.append(int(pad + eff * (1.0 - (v - lo) / span)))

        if style == "solid":
            cv.create_line(*pts, fill=color, width=2, smooth=True)

        elif style == "dashed":
            # Segmentos alternos cada 6 px
            for i in range(0, len(pts) - 2, 4):
                cv.create_line(pts[i], pts[i + 1],
                               pts[i + 2], pts[i + 3],
                               fill=color, width=1)

        elif style == "dotted":
            # Puntos cada ~8 px
            for i in range(0, len(pts) - 2, 8):
                x, y = pts[i], pts[i + 1]
                cv.create_oval(x - 1, y - 1, x + 1, y + 1,
                               fill=color, outline="")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    def destroy(self):
        super().destroy()
