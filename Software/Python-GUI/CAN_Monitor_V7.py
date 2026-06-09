import tkinter as tk
from tkinter import ttk
import time
import struct
import threading
import queue
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Optional

# ============================================================================
# CONFIGURATION
# ============================================================================
# The monitor is fed ONLY by REAL CAN traffic coming from the MAIN ESP32 over
# the serial link (see GUI_ECU._on_serial_line -> CanMonitor.feed_line).
# There is no simulated/demo bus: every row you see is a frame the device
# actually put on the wire, so the route always reflects the real target.

# ── MATCHED THEME COLORS FROM GUI_ECU ───────────────────────────────────────
COLORS = {
    "bg":          "#0d0f12",   # BG
    "panel":       "#161a1f",   # PANEL
    "panel_alt":   "#1c2128",   # CARD
    "border":      "#2a3040",   # BORDER
    "text":        "#cdd6e0",   # Clean UI Text
    "text_dim":    "#7a8899",   # TEXT_SEC
    "text_mono":   "#00e5cc",   # ACCENT_CYAN
    "tx":          "#00aaff",   # ACCENT_BLUE
    "rx":          "#00e676",   # GREEN
    "err":         "#ff1744",   # RED
    "row_alt":     "#121519",
    "highlight":   "#00aaff",
    "err_row":     "#3a171c",
}

# ============================================================================
# PROTOCOL DEFINITION  (mirrors the firmware in setup.py / functions.py)
# ============================================================================
# A real CAN frame on this bus = 1 target-address byte + short ASCII text,
# e.g. b'\x01BAT ON'. WHO sent it is encoded in the CAN id, so:
#   id 0x100 = MAIN, 0x101 = ECU1 ... 0x104 = ECU4
# and the first payload byte is the target address index (MAIN=0, ECU1=1, ...).
#
# Keep these two maps in sync with setup.py if you ever renumber the devices.
NODE_IDS = {
    "MAIN": 0x100,
    "ECU1": 0x101,
    "ECU2": 0x102,
    "ECU3": 0x103,
    "ECU4": 0x104,
}

NODE_NAMES  = {cid: name for name, cid in NODE_IDS.items()}        # id   -> name
ADDR_INDEX  = {name: i for i, name in enumerate(NODE_IDS)}         # name -> byte
ADDR_NAMES  = {i: name for name, i in ADDR_INDEX.items()}          # byte -> name
MAIN_ID     = NODE_IDS["MAIN"]

# Per-arbitration-id description, used by the supervision table + filter popup.
# Each node owns exactly one TX id, so "name" and "sender" coincide here.
MESSAGES = {cid: {"name": name, "sender": name}
            for cid, name in NODE_NAMES.items()}

# Source nodes offered in the filter popup.
DEVICES = {name: f"node 0x{cid:03X}" for name, cid in NODE_IDS.items()}


def format_timestamp(ts: float) -> str:
    """Converts unix timestamp float to a clean human readable HH:MM:SS.mmm format."""
    if ts == 0.0: return "never"
    struct_time = time.localtime(ts)
    millis = int((ts - int(ts)) * 1000)
    return f"{time.strftime('%H:%M:%S', struct_time)}.{millis:03d}"

# ============================================================================
# DATA MODEL
# ============================================================================
@dataclass
class CanFrame:
    timestamp: float
    can_id: int
    data: bytes
    direction: str = "TX"          # "TX" = MAIN sent it, "RX" = MAIN received it
    is_error: bool = False
    error_reason: str = ""

    @property
    def source(self) -> str:
        """Sending node (decoded from the CAN id)."""
        return NODE_NAMES.get(self.can_id, f"0x{self.can_id:03X}")

    @property
    def target(self) -> str:
        """Addressed node (first payload byte)."""
        if self.data:
            return ADDR_NAMES.get(self.data[0], "?")
        return "?"

    @property
    def destinations(self) -> list:
        return [self.target]

    @property
    def text(self) -> str:
        """ASCII command text carried after the target byte."""
        if len(self.data) > 1:
            try:
                return bytes(self.data[1:]).decode(errors="replace").strip()
            except Exception:
                return ""
        return ""

    @property
    def name(self) -> str:
        """Symbolic name shown in the log: 'MAIN→ECU1' etc."""
        tgt = self.target
        if tgt and tgt != "?":
            return f"{self.source}\u2192{tgt}"
        return self.source

    def decode(self) -> list:
        """Return [(field, value, unit), ...] for the detail panel / search."""
        if self.is_error:
            return []
        return [
            ("dir",    self.direction,        ""),
            ("from",   self.source,           ""),
            ("to",     self.target,           ""),
            ("text",   self.text or "\u2014", ""),
        ]

    def summary(self) -> str:
        if self.is_error:
            return self.error_reason or "ERROR"
        arrow = "\u2192" if self.direction == "TX" else "\u2190"
        return f"{arrow} {self.target} : {self.text or '\u2014'}"

class MessageStats:
    def __init__(self):
        self.count = 0
        self.last_time = 0.0

# ============================================================================
# MAIN MONITOR FRAME
# ============================================================================
class CanMonitor(tk.Frame):
    def __init__(self, parent, **kwargs):
        kwargs.setdefault("bg", COLORS["bg"])
        super().__init__(parent, **kwargs)

        self.frame_queue = queue.Queue()
        self.log = deque(maxlen=2000)
        self.msg_stats = defaultdict(MessageStats)
        self.total_messages = 0
        self.paused = False
        self.autoscroll = True

        # Connection / CAN health (updated from the serial reader thread via
        # plain attribute writes; the Tk side reads them in _refresh_ui).
        self._connected  = False
        self._can_health = ("CAN \u2014", COLORS["text_dim"])

        # Advanced Filter State
        self.filter_state = {
            "ids":        set(),
            "sources":    set(),
            "frame_type": "all",
            "text":       "",
        }

        self._setup_styles()
        self._build_layout()

        self.after(20, self._drain_queue)
        self.after(100, self._refresh_ui)

    # ------------------------------------------------------------------
    # PUBLIC API  (called by GUI_ECU)
    # ------------------------------------------------------------------
    def feed_line(self, line: str):
        """Parse one serial line from the MAIN ESP32 and queue any CAN frame.

        Runs on the serial reader thread, so it must stay thread-safe: it only
        touches the (thread-safe) frame_queue and a couple of plain attributes,
        never a Tk widget.

        Understood lines:
            CANTX,<id-hex>,<data-hex>[,ERR]   frame transmitted by MAIN
            CANRX,<id-hex>,<data-hex>         frame received by MAIN
            CAN_OK = True/False               CAN controller health
            CANTEST: PASS/FAIL ...            loopback self-test result
            CAN-> ECU1 : BAT ON  [sent]       legacy debug line (older firmware)
        Anything else is ignored.
        """
        if not line:
            return
        line = line.strip()

        # 1) Structured trace emitted by the firmware.
        if line.startswith("CANTX,") or line.startswith("CANRX,"):
            kind = "TX" if line[3:5] == "TX" else "RX"
            parts = line.split(",")
            if len(parts) < 3:
                return
            try:
                can_id = int(parts[1], 16)
                data = bytes.fromhex(parts[2]) if parts[2] else b""
            except ValueError:
                return
            is_err = (len(parts) >= 4 and parts[3].strip().upper() == "ERR")
            reason = "TX failed (no ACK / bus error)" if is_err else ""
            self.frame_queue.put(CanFrame(time.time(), can_id, data,
                                          direction=kind, is_error=is_err,
                                          error_reason=reason))
            return

        # 2) CAN health / status lines.
        if line.startswith("CAN_OK"):
            ok = line.rstrip().endswith("True")
            self._can_health = ("CAN OK" if ok else "CAN DOWN",
                                COLORS["rx"] if ok else COLORS["err"])
            return
        if line.startswith("CANTEST:"):
            good = "PASS" in line
            self._can_health = ("CANTEST PASS" if good else "CANTEST FAIL",
                                COLORS["rx"] if good else COLORS["err"])
            return

        # 3) Legacy human-readable debug line (firmware without CANTX).
        if line.startswith("CAN->"):
            frame = self._parse_legacy_can(line)
            if frame is not None:
                self.frame_queue.put(frame)
            return
        # anything else is not ours -> ignore

    def set_connected(self, connected: bool):
        """Reflect the serial link state in the toolbar. Called on the Tk thread."""
        self._connected = bool(connected)
        color = COLORS["rx"] if connected else COLORS["err"]
        txt   = "BUS LIVE" if connected else "BUS OFFLINE"
        try:
            self._status_dot.itemconfig(self._status_oval, fill=color)
            self._status_lbl.config(text=txt, fg=color)
        except Exception:
            pass

    @staticmethod
    def _parse_legacy_can(line: str) -> Optional["CanFrame"]:
        """Reconstruct a frame from 'CAN-> ECU1 : BAT ON  [sent]'."""
        try:
            body = line[len("CAN->"):]
            ok = "FAILED" not in body.upper()
            if "[" in body:
                body = body.split("[", 1)[0]
            if ":" not in body:
                return None
            dev, text = body.split(":", 1)
            dev = dev.strip().upper()
            text = text.strip()
            if dev not in ADDR_INDEX:
                return None
            payload = bytes([ADDR_INDEX[dev]]) + text.encode()
            return CanFrame(time.time(), MAIN_ID, payload, direction="TX",
                            is_error=not ok,
                            error_reason="" if ok else "TX failed (no ACK / bus error)")
        except Exception:
            return None

    # ------------------------------------------------------------------
    def _setup_styles(self):
        self.font_mono = ("Courier New", 9)
        self.font_mono_bold = ("Courier New", 9, "bold")
        self.font_header = ("Courier New", 12, "bold")

        style = ttk.Style(self)
        if "clam" in style.theme_names(): style.theme_use("clam")

        style.configure("Treeview", background=COLORS["panel"], foreground=COLORS["text"], fieldbackground=COLORS["panel"], rowheight=24, font=self.font_mono, borderwidth=0)
        style.configure("Treeview.Heading", background=COLORS["border"], foreground=COLORS["text_mono"], font=self.font_mono_bold, relief="flat", borderwidth=0, padding=(6, 4))
        style.map("Treeview.Heading", background=[("active", COLORS["panel_alt"])])
        style.map("Treeview", background=[("selected", COLORS["highlight"])], foreground=[("selected", COLORS["bg"])])

    def _build_layout(self):
        # --- TOOLBAR ---
        bar = tk.Frame(self, bg=COLORS["panel"], highlightbackground=COLORS["border"], highlightthickness=1)
        bar.pack(fill="x", padx=18, pady=(0, 8))

        self._status_dot = tk.Canvas(bar, width=12, height=12, bg=COLORS["panel"], highlightthickness=0)
        self._status_oval = self._status_dot.create_oval(2, 2, 10, 10, fill=COLORS["err"], outline="")
        self._status_dot.pack(side="left", padx=(14, 6), pady=12)

        self._status_lbl = tk.Label(bar, text="BUS OFFLINE", bg=COLORS["panel"], fg=COLORS["err"], font=self.font_mono_bold)
        self._status_lbl.pack(side="left")

        self.lbl_total = tk.Label(bar, text="TOTAL: 0", bg=COLORS["panel"], fg=COLORS["text_dim"], font=self.font_mono)
        self.lbl_total.pack(side="left", padx=20)

        self.can_health_lbl = tk.Label(bar, text="CAN \u2014", bg=COLORS["panel"], fg=COLORS["text_dim"], font=self.font_mono)
        self.can_health_lbl.pack(side="left")

        # Right side controls
        tk.Button(bar, text="CLEAR", command=self._clear_log, bg=COLORS["panel_alt"], fg=COLORS["text"], font=self.font_mono_bold, relief="flat", bd=0, padx=12, pady=4, cursor="hand2").pack(side="right", padx=(0, 10), pady=6)

        self.pause_btn = tk.Button(bar, text="PAUSE", command=self._toggle_pause, bg=COLORS["panel_alt"], fg=COLORS["text"], font=self.font_mono_bold, relief="flat", bd=0, padx=12, pady=4, cursor="hand2")
        self.pause_btn.pack(side="right", padx=(0, 4), pady=6)

        # Filter button with popup logic
        self.filter_btn = tk.Button(bar, text="FILTER \u25be", command=self._open_filter_popup, bg=COLORS["panel_alt"], fg=COLORS["text"], font=self.font_mono_bold, relief="flat", bd=0, padx=12, pady=4, cursor="hand2")
        self.filter_btn.pack(side="right", padx=(0, 4), pady=6)

        self.filter_indicator = tk.Canvas(bar, width=10, height=10, bg=COLORS["panel"], highlightthickness=0)
        self.filter_indicator_oval = self.filter_indicator.create_oval(1, 1, 9, 9, fill=COLORS["panel"], outline="")
        self.filter_indicator.pack(side="right", padx=(0, 4))

        # --- SUPERVISION NODE ---
        sup_frame = tk.Frame(self, bg=COLORS["panel"], highlightbackground=COLORS["border"], highlightthickness=1)
        sup_frame.pack(fill="x", padx=18, pady=(0, 8))
        tk.Label(sup_frame, text="SUPERVISION NODE", bg=COLORS["panel"], fg=COLORS["text"], font=self.font_header).pack(anchor="w", padx=14, pady=(6, 4))

        self.sup_tree = ttk.Treeview(sup_frame, columns=("id", "name", "src", "count", "last"), show="headings", height=5)
        for col, lbl, w, anch in [("id", "ID", 70, "w"), ("name", "NODE", 130, "w"), ("src", "ROLE", 120, "w"), ("count", "FRAMES", 80, "e"), ("last", "LAST SEEN", 140, "e")]:
            self.sup_tree.heading(col, text=lbl)
            self.sup_tree.column(col, width=w, anchor=anch, stretch=False)
        self.sup_tree.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        for cid in MESSAGES:
            role = "MAIN (controller)" if cid == MAIN_ID else "ECU node"
            self.sup_tree.insert("", "end", iid=f"sup_{cid}", values=(f"0x{cid:03X}", MESSAGES[cid]["name"], role, "0", "never"))

        # --- PANED BODY CONTAINER ---
        body = tk.PanedWindow(self, orient="horizontal", bg=COLORS["bg"], sashwidth=6, bd=0)
        body.pack(fill="both", expand=True, padx=18, pady=(0, 18))

        # --- MESSAGE LOG (LEFT PANEL) ---
        log_panel = tk.Frame(body, bg=COLORS["panel"], highlightbackground=COLORS["border"], highlightthickness=1)
        body.add(log_panel, minsize=500, stretch="always")

        log_hdr = tk.Frame(log_panel, bg=COLORS["panel"])
        log_hdr.pack(fill="x", padx=14, pady=(6, 4))
        tk.Label(log_hdr, text="MESSAGE LOG", bg=COLORS["panel"], fg=COLORS["text"], font=self.font_header).pack(side="left")
        self.log_count_label = tk.Label(log_hdr, text="", bg=COLORS["panel"], fg=COLORS["text_dim"], font=self.font_mono)
        self.log_count_label.pack(side="right")

        tree_frame = tk.Frame(log_panel, bg=COLORS["panel"])
        tree_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self.tree = ttk.Treeview(tree_frame, columns=("time", "id", "name", "data", "decoded"), show="headings")
        self.tree.heading("time", text="TIME")
        self.tree.column("time", width=120, stretch=False)
        self.tree.heading("id", text="ID")
        self.tree.column("id", width=70, stretch=False)
        self.tree.heading("name", text="ROUTE")
        self.tree.column("name", width=120, stretch=False)
        self.tree.heading("data", text="DATA (HEX)")
        self.tree.column("data", width=200, stretch=False)
        self.tree.heading("decoded", text="DECODED")
        self.tree.column("decoded", width=250, stretch=True)

        self.tree.tag_configure("alt", background=COLORS["row_alt"])
        self.tree.tag_configure("err", background=COLORS["err_row"])
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self.tree.bind("<Button-1>", lambda e: setattr(self, 'autoscroll', False))
        self.tree.bind("<<TreeviewSelect>>", self._on_frame_selected)

        # --- FRAME DETAIL (RIGHT PANEL) ---
        det_panel = tk.Frame(body, bg=COLORS["panel"], highlightbackground=COLORS["border"], highlightthickness=1)
        body.add(det_panel, minsize=260, stretch="never", width=340)
        tk.Label(det_panel, text="FRAME DETAIL", bg=COLORS["panel"], fg=COLORS["text"], font=self.font_header).pack(anchor="w", padx=14, pady=(6, 4))

        self.detail_text = tk.Text(det_panel, bg=COLORS["panel_alt"], fg=COLORS["text"], font=self.font_mono, relief="flat", padx=12, pady=10, wrap="word", bd=0, highlightthickness=1, highlightbackground=COLORS["border"])
        self.detail_text.pack(fill="both", expand=True, padx=14, pady=(0, 14))

    # ------------------------------------------------------------------
    # POPUP FILTER LOGIC
    # ------------------------------------------------------------------
    def _open_filter_popup(self):
        if hasattr(self, "_filter_popup") and self._filter_popup.winfo_exists():
            self._filter_popup.lift()
            self._filter_popup.focus_force()
            return

        popup = tk.Toplevel(self)
        self._filter_popup = popup
        popup.title("Filter Messages")
        popup.configure(bg=COLORS["panel"])
        popup.transient(self.winfo_toplevel())
        popup.resizable(False, False)

        self.update_idletasks()
        btn_x = self.filter_btn.winfo_rootx()
        btn_y = self.filter_btn.winfo_rooty() + self.filter_btn.winfo_height() + 4
        pw = 360
        right_edge = btn_x + self.filter_btn.winfo_width()
        x = max(20, right_edge - pw)
        popup.geometry(f"{pw}x520+{x}+{btn_y}")

        popup.bind("<Escape>", lambda _e: self._cancel_filter_popup())

        body = tk.Frame(popup, bg=COLORS["panel"])
        body.pack(fill="both", expand=True, padx=14, pady=12)

        self._pf_id_vars     = {}
        self._pf_unknown_var = tk.BooleanVar(value=False)
        self._pf_source_vars = {}
        self._pf_frame_type  = tk.StringVar(value=self.filter_state["frame_type"])
        self._pf_text_var    = tk.StringVar(value=self.filter_state["text"])

        # Section: Message ID
        self._popup_section_header(body, "Message ID")
        for cid, info in MESSAGES.items():
            var = tk.BooleanVar(value=(cid in self.filter_state["ids"]))
            self._pf_id_vars[cid] = var
            self._popup_check(body, f"0x{cid:03X}  {info['name']} ({info['sender']})", var)
        self._pf_unknown_var.set(-1 in self.filter_state["ids"])
        self._popup_check(body, "Unknown / undefined IDs", self._pf_unknown_var)

        # Section: Source node
        self._popup_section_header(body, "Source node")
        for name, desc in DEVICES.items():
            var = tk.BooleanVar(value=(name in self.filter_state["sources"]))
            self._pf_source_vars[name] = var
            self._popup_check(body, f"{name} ({desc})", var)

        # Section: Frame type
        self._popup_section_header(body, "Frame type")
        for value, label in [("all", "All frames"), ("data", "Data frames only"), ("error", "Error frames only")]:
            self._popup_radio(body, label, self._pf_frame_type, value)

        # Section: Text search
        self._popup_section_header(body, "Text search (optional)")
        entry = tk.Entry(body, textvariable=self._pf_text_var, bg=COLORS["panel_alt"], fg=COLORS["text"], insertbackground=COLORS["text"], relief="flat", font=self.font_mono)
        entry.pack(fill="x", padx=(20, 0), ipady=5, pady=(2, 4))

        tk.Label(body, text="Matches ID, route, source, and decoded text.", bg=COLORS["panel"], fg=COLORS["text_dim"], font=("Courier New", 8), anchor="w").pack(fill="x", padx=(20, 0))

        # Buttons
        btn_row = tk.Frame(popup, bg=COLORS["panel"])
        btn_row.pack(fill="x", padx=14, pady=(0, 14))
        tk.Button(btn_row, text="Clear all", command=self._clear_filter_popup, bg=COLORS["panel_alt"], fg=COLORS["text"], font=self.font_mono, relief="flat", padx=14).pack(side="left")
        tk.Button(btn_row, text="Apply", command=self._apply_filter_popup, bg=COLORS["tx"], fg=COLORS["bg"], font=self.font_mono_bold, relief="flat", padx=14).pack(side="right")
        tk.Button(btn_row, text="Cancel", command=self._cancel_filter_popup, bg=COLORS["panel_alt"], fg=COLORS["text"], font=self.font_mono, relief="flat", padx=14).pack(side="right", padx=(0,6))

        popup.bind("<Return>", lambda _e: self._apply_filter_popup())

    def _popup_section_header(self, parent, text):
        tk.Label(parent, text=text, bg=COLORS["panel"], fg=COLORS["text"], font=self.font_mono_bold, anchor="w").pack(fill="x", pady=(8, 2))

    def _popup_check(self, parent, label, var):
        tk.Checkbutton(parent, text=label, variable=var, bg=COLORS["panel"], fg=COLORS["text"], selectcolor=COLORS["panel_alt"], activebackground=COLORS["panel"], activeforeground=COLORS["text"], font=self.font_mono, anchor="w", bd=0).pack(fill="x", padx=(20, 0), pady=1)

    def _popup_radio(self, parent, label, var, value):
        tk.Radiobutton(parent, text=label, variable=var, value=value, bg=COLORS["panel"], fg=COLORS["text"], selectcolor=COLORS["panel_alt"], activebackground=COLORS["panel"], activeforeground=COLORS["text"], font=self.font_mono, anchor="w", bd=0).pack(fill="x", padx=(20, 0), pady=1)

    def _clear_filter_popup(self):
        for v in self._pf_id_vars.values(): v.set(False)
        self._pf_unknown_var.set(False)
        for v in self._pf_source_vars.values(): v.set(False)
        self._pf_frame_type.set("all")
        self._pf_text_var.set("")

    def _cancel_filter_popup(self):
        if hasattr(self, "_filter_popup") and self._filter_popup.winfo_exists():
            self._filter_popup.destroy()

    def _apply_filter_popup(self):
        ids = {cid for cid, v in self._pf_id_vars.items() if v.get()}
        if self._pf_unknown_var.get(): ids.add(-1)
        sources = {n for n, v in self._pf_source_vars.items() if v.get()}

        self.filter_state = {
            "ids":        ids,
            "sources":    sources,
            "frame_type": self._pf_frame_type.get(),
            "text":       self._pf_text_var.get().strip().lower(),
        }
        self._update_filter_indicator()
        self._rebuild_log_treeview()
        self._cancel_filter_popup()

    def _update_filter_indicator(self):
        active = self._filter_active()
        self.filter_indicator.itemconfig(self.filter_indicator_oval, fill=COLORS["tx"] if active else COLORS["panel"])
        if active:
            n = sum([1 for k, v in self.filter_state.items() if v and v != "all"])
            self.filter_btn.config(text=f"FILTER ({n}) \u25be", fg=COLORS["tx"])
        else:
            self.filter_btn.config(text="FILTER \u25be", fg=COLORS["text"])

    def _filter_active(self) -> bool:
        fs = self.filter_state
        return bool(fs["ids"]) or bool(fs["sources"]) or fs["frame_type"] != "all" or bool(fs["text"])

    def _matches_filter(self, f: CanFrame) -> bool:
        fs = self.filter_state
        if fs["ids"]:
            id_ok = (f.can_id in fs["ids"]) or (-1 in fs["ids"] and f.can_id not in MESSAGES)
            if not id_ok: return False
        if fs["sources"] and f.source not in fs["sources"]: return False
        if fs["frame_type"] == "error" and not f.is_error: return False
        if fs["frame_type"] == "data" and f.is_error: return False
        if fs["text"]:
            decoded_text = " ".join(f"{name}={val}{unit}" for name, val, unit in f.decode())
            hay = (f"0x{f.can_id:03X} {f.name} {f.source} {','.join(f.destinations)} {decoded_text} {f.error_reason}").lower()
            if fs["text"] not in hay: return False
        return True

    # ------------------------------------------------------------------
    # CORE LOGIC
    # ------------------------------------------------------------------
    def _drain_queue(self):
        try:
            for _ in range(60):
                frame = self.frame_queue.get_nowait()
                self.log.append(frame)
                self.total_messages += 1
                stats = self.msg_stats[frame.can_id]
                stats.count += 1
                stats.last_time = frame.timestamp
        except queue.Empty:
            pass
        self.after(20, self._drain_queue)

    def _refresh_ui(self):
        self.lbl_total.config(text=f"TOTAL: {self.total_messages:,}")

        txt, color = self._can_health
        self.can_health_lbl.config(text=txt, fg=color)

        for cid, stats in self.msg_stats.items():
            iid = f"sup_{cid}"
            if self.sup_tree.exists(iid):
                role = "MAIN (controller)" if cid == MAIN_ID else "ECU node"
                self.sup_tree.item(iid, values=(f"0x{cid:03X}", MESSAGES[cid]["name"], role, str(stats.count), format_timestamp(stats.last_time)))

        if not self.paused:
            self._rebuild_log_treeview()

        self.after(100, self._refresh_ui)

    def _rebuild_log_treeview(self):
        active = self._filter_active()
        all_frames = list(self.log)
        visible = [f for f in all_frames if self._matches_filter(f)] if active else all_frames
        display_list = visible[-100:]

        self.tree.delete(*self.tree.get_children())

        for i, f in enumerate(display_list):
            clean_id = f"0x{f.can_id:03X}"
            hex_data = " ".join(f"{b:02X}" for b in f.data)

            tag = "err" if f.is_error else ("alt" if i % 2 == 0 else "")

            self.tree.insert("", "end", values=(format_timestamp(f.timestamp), clean_id, f.name, hex_data, f.summary()), tags=(tag,))

        if self.autoscroll:
            self.tree.yview_moveto(1)

        self.log_count_label.config(text=f"Showing {len(display_list)} of {len(all_frames):,} {'(filtered)' if active else ''}")

    def _on_frame_selected(self, event):
        selection = self.tree.selection()
        if not selection: return
        item = self.tree.item(selection[0])
        values = item['values']
        if not values: return

        try: target_id = int(values[1], 16)
        except ValueError: return

        matched_frame = None
        for f in reversed(self.log):
            if f.can_id == target_id:
                matched_frame = f
                break

        self.detail_text.delete("1.0", tk.END)
        if not matched_frame: return

        lines = [
            f"FRAME DETAILS",
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
            f"Timestamp : {format_timestamp(matched_frame.timestamp)}",
            f"Arb ID    : 0x{matched_frame.can_id:03X}",
            f"Route     : {matched_frame.name}",
            f"Direction : {matched_frame.direction}",
            f"Source    : {matched_frame.source}",
            f"Target    : {matched_frame.target}",
            f"Length    : {len(matched_frame.data)} bytes",
            f"Raw Hex   : {' '.join(f'{b:02X}' for b in matched_frame.data)}",
            f"\nDECODED:",
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
        ]

        if matched_frame.is_error:
            lines.append(f"!! ERROR FRAME: {matched_frame.error_reason} !!")
        else:
            signals = matched_frame.decode()
            if signals:
                for name, val, unit in signals: lines.append(f" \u2022 {name:<8} : {val} {unit}".strip())
            else:
                lines.append(" No payload to decode.")

        self.detail_text.insert("1.0", "\n".join(lines))

    def _toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.pause_btn.config(text="RESUME", fg=COLORS["bg"], bg=COLORS["highlight"])
            self.autoscroll = False
        else:
            self.pause_btn.config(text="PAUSE", fg=COLORS["text"], bg=COLORS["panel_alt"])
            self.autoscroll = True

    def _clear_log(self):
        self.log.clear()
        self.tree.delete(*self.tree.get_children())
        self.total_messages = 0
        self.msg_stats.clear()
        self.autoscroll = True
        self.detail_text.delete("1.0", tk.END)
        for cid in MESSAGES:
            iid = f"sup_{cid}"
            if self.sup_tree.exists(iid):
                role = "MAIN (controller)" if cid == MAIN_ID else "ECU node"
                self.sup_tree.item(iid, values=(f"0x{cid:03X}", MESSAGES[cid]["name"], role, "0", "never"))