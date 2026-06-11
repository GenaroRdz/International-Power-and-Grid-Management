import tkinter as tk
from tkinter import filedialog
import datetime
import threading
import time

# ── Paleta (igual al app principal) ──────────────────────────────────────────
BG       = "#0d0f12"
PANEL    = "#161a1f"
CARD     = "#1c2128"
BORDER   = "#2a3040"
GREEN    = "#00e676"
RED      = "#ff1744"
YELLOW   = "#ffd600"
CYAN     = "#00e5cc"
BLUE     = "#00aaff"
LILA     = "#B565FC"
ORANGE   = "#ff8c00"
TEXT_SEC = "#7a8899"
TEXT_DIM = "#3d4a5a"

FONT_TERM  = ("Courier New", 9)
FONT_LABEL = ("Courier New", 8,  "bold")
FONT_CH    = ("Courier New", 11, "bold")

# ── Comandos conocidos (para autocompletado con Tab) ──────────────────────────
KNOWN_COMMANDS = [
    "*INIT", "*RST", "*IDN?",
    "ping",
    "INA_START", "INA_STOP", "INA?",
    "ON ALL", "OFF ALL",
    "CANTEST", "CAN?",
    "ECU1_BAT ON", "ECU1_BAT OFF",
    "ECU1_ACC ON", "ECU1_ACC OFF",
    "ECU1_IGN ON", "ECU1_IGN OFF",
    "ECU2_BAT ON", "ECU2_BAT OFF",
    "ECU2_ACC ON", "ECU2_ACC OFF",
    "ECU2_IGN ON", "ECU2_IGN OFF",
    "ECU3_BAT ON", "ECU3_BAT OFF",
    "ECU3_ACC ON", "ECU3_ACC OFF",
    "ECU3_IGN ON", "ECU3_IGN OFF",
    "ECU4_BAT ON", "ECU4_BAT OFF",
    "ECU4_ACC ON", "ECU4_ACC OFF",
    "ECU4_IGN ON", "ECU4_IGN OFF",
]

# ── Colores por tipo de línea ─────────────────────────────────────────────────
def _line_color(line: str) -> str:
    """Devuelve el color del texto según el contenido de la línea."""
    lo = line.lower()
    if line.startswith(">>>"):          # comando enviado por el usuario
        return BLUE
    if line.startswith("INA,"):         # streaming de sensores
        return TEXT_DIM
    if line.startswith("CANTX") or line.startswith("CANRX"):
        return CYAN
    if "error" in lo:
        return RED
    if "pong" in lo or "initialized" in lo or "listo" in lo:
        return GREEN
    if "warn" in lo or "boot" in lo:
        return YELLOW
    return TEXT_SEC                     # respuesta genérica


class TerminalTab(tk.Frame):
    MAX_LINES = 500     # líneas máximas en el historial antes de limpiar las viejas

    def __init__(self, parent, esp32, **kw):
        kw.setdefault("bg", BG)
        super().__init__(parent, **kw)

        self._esp32      = esp32
        self._connected  = False
        self._history    = []       # historial de comandos enviados (para ↑↓)
        self._hist_idx   = -1       # posición actual en el historial
        self._ac_matches = []       # candidatos de autocompletado actuales
        self._ac_idx     = -1       # posición en el ciclo de autocompletado

        # ── Script runner (PC-side sequencer) ────────────────────────────────
        self._seq_thread = None
        self._seq_stop   = threading.Event()

        self._build_ui()

    # ── API pública ───────────────────────────────────────────────────────────
    def feed_response(self, line: str):
        """
        Muestra una línea recibida del ESP32 en el historial.
        Se puede llamar desde cualquier hilo — usa after() internamente.
        """
        self.after(0, lambda l=line: self._append(l, source="rx"))

    def set_connected(self, connected: bool):
        self._connected = connected
        # If the link drops while a script is running, abort it.
        if not connected:
            self._seq_stop.set()
        color  = GREEN  if connected else RED
        symbol = "●"    if connected else "○"
        self._prompt_lbl.configure(fg=color,
                                   text=f"{symbol} ESP32 > ")
        self._entry.configure(
            state="normal" if connected else "disabled")

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Header ───────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=BG, pady=8, padx=14)
        hdr.pack(fill="x")

        tk.Label(hdr, text="SERIAL TERMINAL", font=FONT_CH,
                 fg=BLUE, bg=BG).pack(side="left")

        self._clear_btn = tk.Button(
            hdr, text="✕  CLEAR", font=FONT_LABEL,
            fg=TEXT_SEC, bg=CARD, relief="flat", bd=0,
            highlightbackground=BORDER, highlightthickness=1,
            cursor="hand2", padx=8, pady=3,
            command=self._clear_history)
        self._clear_btn.pack(side="right", padx=(4, 0))

        tk.Label(hdr, text="Tab: autocomplete   ↑↓: history",
                 font=FONT_LABEL, fg=TEXT_DIM, bg=BG).pack(side="right", padx=12)

        # ── Área de historial ────────────────────────────────────────────────
        text_frame = tk.Frame(self, bg=BORDER, padx=1, pady=1)
        text_frame.pack(fill="both", expand=True, padx=14, pady=(0, 8))

        self._text = tk.Text(
            text_frame,
            bg=CARD, fg=TEXT_SEC,
            font=FONT_TERM,
            relief="flat", bd=0,
            state="disabled",
            wrap="word",
            insertbackground=GREEN,
            selectbackground=BORDER,
            cursor="arrow",
            padx=10, pady=8,
        )
        sb = tk.Scrollbar(text_frame, command=self._text.yview,
                          bg=PANEL, troughcolor=CARD)
        self._text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._text.pack(fill="both", expand=True)

        # Configurar tags de color (uno por color posible)
        for color in [BLUE, CYAN, GREEN, RED, YELLOW, TEXT_SEC, TEXT_DIM, ORANGE, LILA]:
            self._text.tag_configure(color, foreground=color)
        # Tag para timestamps
        self._text.tag_configure("ts", foreground=TEXT_DIM)

        # Mensaje de bienvenida
        self._append_raw(
            "─" * 60 + "\n"
            "  ECU SUPPLY CONTROLLER — Serial Terminal\n"
            "  Comandos directos al ESP32. Tab = autocomplete.\n"
            "  Conecta el dispositivo para habilitar el input.\n"
            "─" * 60 + "\n",
            color=TEXT_DIM
        )

        # ── Input row ────────────────────────────────────────────────────────
        inp = tk.Frame(self, bg=PANEL,
                       highlightbackground=BORDER, highlightthickness=1,
                       padx=8, pady=6)
        inp.pack(fill="x", padx=14, pady=(0, 14))

        self._prompt_lbl = tk.Label(
            inp, text="○ ESP32 > ",
            font=FONT_TERM, fg=RED, bg=PANEL)
        self._prompt_lbl.pack(side="left")

        self._entry_var = tk.StringVar()
        self._entry = tk.Entry(
            inp,
            textvariable=self._entry_var,
            font=FONT_TERM,
            fg=GREEN, bg=PANEL,
            insertbackground=GREEN,
            relief="flat", bd=0,
            disabledbackground=PANEL,
            state="disabled",
        )
        self._entry.pack(side="left", fill="x", expand=True)

        send_btn = tk.Button(
            inp, text="SEND ↵",
            font=FONT_LABEL, fg=BLUE, bg=CARD,
            relief="flat", bd=0,
            highlightbackground=BLUE, highlightthickness=1,
            cursor="hand2", padx=8, pady=2,
            command=self._send_current)
        send_btn.pack(side="right", padx=(8, 0))

        # ── Script runner controls ───────────────────────────────────────────
        self._run_btn = tk.Button(
            inp, text="▶ RUN SCRIPT",
            font=FONT_LABEL, fg=GREEN, bg=CARD,
            relief="flat", bd=0,
            highlightbackground=GREEN, highlightthickness=1,
            cursor="hand2", padx=8, pady=2,
            command=self._run_script_dialog)
        self._run_btn.pack(side="right", padx=(8, 0))

        self._stop_btn = tk.Button(
            inp, text="■ STOP",
            font=FONT_LABEL, fg=RED, bg=CARD,
            relief="flat", bd=0,
            highlightbackground=RED, highlightthickness=1,
            cursor="hand2", padx=8, pady=2,
            state="disabled",
            command=self._stop_script)
        self._stop_btn.pack(side="right", padx=(8, 0))

        # Bindings
        self._entry.bind("<Return>",   lambda e: self._send_current())
        self._entry.bind("<Up>",       lambda e: self._history_prev())
        self._entry.bind("<Down>",     lambda e: self._history_next())
        self._entry.bind("<Tab>",      self._autocomplete)
        self._entry.bind("<KeyPress>", self._reset_ac)   # cualquier otra tecla cancela el ciclo

    # ── Envío de comandos ─────────────────────────────────────────────────────
    def _send_current(self):
        cmd = self._entry_var.get().strip()
        if not cmd:
            return

        # Guardar en historial
        if not self._history or self._history[-1] != cmd:
            self._history.append(cmd)
        self._hist_idx = -1

        self._entry_var.set("")
        self._ac_matches = []
        self._ac_idx = -1

        # Mostrar en el log
        self._append(f">>> {cmd}", source="tx")

        # Enviar al ESP32
        if not self._connected:
            self._append("  ⚠  No conectado — comando no enviado.", source="warn")
            return

        ok = self._esp32.send(cmd)
        if not ok:
            self._append("  ✕  Error al enviar (puerto cerrado).", source="err")

    # ── Historial ↑ / ↓ ──────────────────────────────────────────────────────
    def _history_prev(self):
        if not self._history:
            return
        if self._hist_idx == -1:
            self._hist_idx = len(self._history) - 1
        elif self._hist_idx > 0:
            self._hist_idx -= 1
        self._entry_var.set(self._history[self._hist_idx])
        self._entry.icursor(tk.END)

    def _history_next(self):
        if self._hist_idx == -1:
            return
        if self._hist_idx < len(self._history) - 1:
            self._hist_idx += 1
            self._entry_var.set(self._history[self._hist_idx])
        else:
            self._hist_idx = -1
            self._entry_var.set("")
        self._entry.icursor(tk.END)

    # ── Autocompletado con Tab ────────────────────────────────────────────────
    def _autocomplete(self, event):
        typed = self._entry_var.get()

        # Si ya estamos ciclando, avanzar al siguiente
        if self._ac_matches:
            self._ac_idx = (self._ac_idx + 1) % len(self._ac_matches)
            self._entry_var.set(self._ac_matches[self._ac_idx])
            self._entry.icursor(tk.END)
            return "break"

        # Buscar candidatos
        up = typed.upper()
        self._ac_matches = [c for c in KNOWN_COMMANDS
                            if c.upper().startswith(up) and c.upper() != up]
        if not self._ac_matches:
            return "break"

        self._ac_idx = 0
        self._entry_var.set(self._ac_matches[0])
        self._entry.icursor(tk.END)
        return "break"

    def _reset_ac(self, event):
        # Tab se maneja aparte; cualquier otra tecla resetea el ciclo
        if event.keysym not in ("Tab",):
            self._ac_matches = []
            self._ac_idx = -1

    # ── Script runner (PC-side sequencer) ─────────────────────────────────────
    #
    # An ESP32 does not execute a .py file -- its firmware only understands the
    # serial commands (ECU1_BAT ON, etc.). So "uploading a script" really means:
    # parse a tiny text script HERE and stream the commands over serial with the
    # right timing. The firmware stays untouched.
    #
    # Script format (one instruction per line, '#' starts a comment):
    #     LOOP 10            # repeat the block 10 times
    #         ECU1_BAT ON
    #         DELAY 50       # wait 50 ms
    #         ECU1_BAT OFF
    #         DELAY 50
    #     ENDLOOP
    #
    # As a convenience the inline form is also accepted, so the requested
    # behaviour can be written on one line:
    #     ECU1_BAT ON for 50ms
    # ('for <n>ms' is turned into the command + a DELAY automatically.)

    def _run_script_dialog(self):
        if self._seq_thread is not None and self._seq_thread.is_alive():
            return
        if not self._connected:
            self._append("  ⚠  Conecta el dispositivo antes de correr un script.",
                         source="warn")
            return
        path = filedialog.askopenfilename(
            title="Select sequence script",
            filetypes=[("Script", "*.py *.txt *.seq"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            self._append(f"  ✕  No se pudo abrir el archivo: {e}", source="err")
            return

        try:
            program = self._compile_script(text)
        except ValueError as e:
            self._append(f"  ✕  Error de sintaxis: {e}", source="err")
            return

        if not program:
            self._append("  ⚠  El script no contiene comandos.", source="warn")
            return

        self._start_sequence(program, path)

    def _compile_script(self, text):
        """Turn script text into a flat list of ('CMD', str) / ('DELAY', ms)
        steps. LOOP/ENDLOOP blocks are expanded here so the runner thread stays
        trivial. Raises ValueError on malformed input."""
        steps = []
        loop_stack = []     # list of (repeat_count, start_index_in_steps)

        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            up = line.upper()

            if up.startswith("LOOP"):
                parts = line.split()
                if len(parts) != 2 or not parts[1].isdigit():
                    raise ValueError(f"LOOP necesita un número: '{line}'")
                loop_stack.append((int(parts[1]), len(steps)))
                continue

            if up == "ENDLOOP":
                if not loop_stack:
                    raise ValueError("ENDLOOP sin LOOP")
                count, start = loop_stack.pop()
                block = steps[start:]
                del steps[start:]
                for _ in range(count):
                    steps.extend(block)
                continue

            if up.startswith("DELAY"):
                parts = line.split()
                if len(parts) != 2 or not parts[1].isdigit():
                    raise ValueError(f"DELAY necesita ms: '{line}'")
                steps.append(("DELAY", int(parts[1])))
                continue

            # Inline 'CMD for <n>ms' -> command + delay.
            low = line.lower()
            if " for " in low and low.rstrip().endswith("ms"):
                head, tail = line.rsplit(" for ", 1)
                ms_txt = tail.strip()[:-2].strip()
                if not ms_txt.isdigit():
                    raise ValueError(f"Tiempo inválido: '{line}'")
                steps.append(("CMD", head.strip()))
                steps.append(("DELAY", int(ms_txt)))
                continue

            steps.append(("CMD", line))

        if loop_stack:
            raise ValueError("LOOP sin ENDLOOP")
        return steps

    def _start_sequence(self, program, path):
        self._seq_stop.clear()
        self._run_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        name = path.replace("\\", "/").split("/")[-1]
        n_cmds = sum(1 for k, _ in program if k == "CMD")
        self._append(f"  ▶  RUN '{name}'  ({n_cmds} comandos)…", source="tx")

        self._seq_thread = threading.Thread(
            target=self._sequence_worker, args=(program,), daemon=True)
        self._seq_thread.start()

    def _sequence_worker(self, program):
        """Runs on its own thread so the GUI never freezes. Streams commands
        over serial and sleeps for DELAY steps. Stops promptly on STOP."""
        finished = True
        for kind, val in program:
            if self._seq_stop.is_set():
                finished = False
                break
            if kind == "DELAY":
                # Sleep in small slices so STOP reacts within ~20 ms.
                end = time.time() + val / 1000.0
                while time.time() < end:
                    if self._seq_stop.is_set():
                        break
                    time.sleep(0.02)
            else:  # CMD
                self.feed_response(f">>> {val}")
                ok = self._esp32.send(val)
                if not ok:
                    self.feed_response("  ✕  Error al enviar (puerto cerrado).")
                    finished = False
                    break
        self.after(0, lambda: self._sequence_done(finished))

    def _sequence_done(self, finished):
        self._run_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        if finished:
            self._append("  ✓  Script finalizado.", source="rx")
        else:
            self._append("  ■  Script detenido.", source="warn")

    def _stop_script(self):
        self._seq_stop.set()

    # ── Escritura en el Text widget ───────────────────────────────────────────
    def _append(self, line: str, source: str = "rx"):
        """Agrega una línea al historial con timestamp y color automático."""
        if source == "tx":
            color = BLUE
        elif source == "warn":
            color = YELLOW
        elif source == "err":
            color = RED
        else:
            color = _line_color(line)

        # No mostrar líneas INA en la terminal (son ruidosas y ya las grafica el scope)
        if line.startswith("INA,"):
            return

        self._append_raw(line + "\n", color=color, timestamp=True)

    def _append_raw(self, text: str, color: str = TEXT_SEC, timestamp: bool = False):
        self._text.configure(state="normal")

        if timestamp:
            ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self._text.insert(tk.END, f"[{ts}] ", "ts")

        self._text.insert(tk.END, text, color)

        # Recortar si hay demasiadas líneas
        lines = int(self._text.index("end-1c").split(".")[0])
        if lines > self.MAX_LINES:
            self._text.delete("1.0", f"{lines - self.MAX_LINES}.0")

        self._text.configure(state="disabled")
        self._text.see(tk.END)

    def _clear_history(self):
        self._text.configure(state="normal")
        self._text.delete("1.0", tk.END)
        self._text.configure(state="disabled")