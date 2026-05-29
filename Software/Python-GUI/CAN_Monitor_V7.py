"""
CAN Bus Monitor — Tkinter GUI
=============================
"""
import tkinter as tk
from tkinter import ttk, font
from tkinter import messagebox
import time
import random
import struct
import threading
import queue
import os
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Optional, Callable

# ============================================================================
# CONFIGURATION
# ============================================================================
USE_REAL_BUS = False   
BUS_BITRATE = 500_000

TIMEOUT_FACTOR = 2.0
JITTER_WARN_FRACTION = 0.20   
EVENT_ACTIVE_WINDOW_S = 30.0

# ============================================================================
# PROTOCOL DEFINITION 
# ============================================================================
DEVICES = {
    "Controller": "Master controller node",
    "Device":     "Motor / sensor node",
}

MESSAGES = {
    0x100: {
        "name":      "MotorCmd",
        "sender":    "Controller",
        "receivers": ["Device"],
        "cycle_ms":  None,          
        "format":    "<BBHB3x",     
        "signals": [
            ("Command",   "B",  1.0,    0,    ""),     
            ("Direction", "B",  1.0,    0,    ""),     
            ("TargetRPM", "H",  1.0,    0,    "rpm"),
            ("Accel",     "B",  1.0,    0,    ""),
        ],
    },
    0x200: {
        "name":      "SensorData",
        "sender":    "Device",
        "receivers": ["Controller"],
        "cycle_ms":  100,
        "format":    "<HhBB2x",     
        "signals": [
            ("ActualRPM", "H",  1.0,    0,    "rpm"),
            ("Temp",      "h",  0.1,    0,    "°C"),
            ("Current",   "B",  0.1,    0,    "A"),
            ("Flags",     "B",  1.0,    0,    "bits"),
        ],
    },
    0x300: {
        "name":      "Heartbeat",
        "sender":    "Device",
        "receivers": ["Controller"],
        "cycle_ms":  1000,
        "format":    "<BH5x",       
        "signals": [
            ("Counter", "B",  1.0,    0,    ""),
            ("Uptime",  "H",  1.0,    0,    "s"),
        ],
    },
}

COMMAND_LABELS = {1: "START", 2: "STOP", 3: "SET_SPEED"}
DIRECTION_LABELS = {0: "stop", 1: "fwd", 2: "rev"}

SENSOR_FLAGS = [
    (0b00000001, "RUN"),
    (0b00000010, "OVERHEAT"),
    (0b00000100, "OVERCURRENT"),
]

COLORS = {
    "bg":          "#1a1a1f",
    "panel":       "#23232a",
    "panel_alt":   "#2a2a32",
    "border":      "#3a3a45",
    "text":        "#e6e6ea",
    "text_dim":    "#8a8a94",
    "text_mono":   "#c8c8d2",
    "tx":          "#3B82F6",   
    "rx":          "#10B981",   
    "err":         "#F59E0B",   
    "idle":        "#52525b",   
    "row_alt":     "#26262d",
    "highlight":   "#2c4060",
    "err_row":     "#5a3a1a",
    "tx_row":      "#1a3050",
}

# ============================================================================
# DATA MODEL
# ============================================================================
@dataclass
class CanFrame:
    timestamp: float
    can_id: int
    data: bytes
    is_error: bool = False
    error_reason: str = ""

    @property
    def msg_info(self) -> dict:
        return MESSAGES.get(self.can_id, {})

    @property
    def source(self) -> str:
        return self.msg_info.get("sender", "?")

    @property
    def destinations(self) -> list:
        return self.msg_info.get("receivers", ["?"])

    @property
    def name(self) -> str:
        return self.msg_info.get("name", f"Unknown_0x{self.can_id:03X}")

    def decode(self) -> list:
        info = self.msg_info
        if not info or self.is_error:
            return []
        try:
            fmt = info["format"]
            values = struct.unpack(fmt, self.data)
        except struct.error:
            return []

        out = []
        for (sig_name, _stype, scale, offset, unit), raw in zip(info["signals"], values):
            value = raw * scale + offset
            if self.can_id == 0x100 and sig_name == "Command":
                out.append((sig_name, COMMAND_LABELS.get(int(value), str(int(value))), ""))
            elif self.can_id == 0x100 and sig_name == "Direction":
                out.append((sig_name, DIRECTION_LABELS.get(int(value), str(int(value))), ""))
            elif self.can_id == 0x200 and sig_name == "Flags":
                bits = [name for mask, name in SENSOR_FLAGS if int(value) & mask]
                out.append((sig_name, "|".join(bits) if bits else "—", ""))
            elif unit == "°C" or unit == "A":
                out.append((sig_name, f"{value:.1f}", unit))
            else:
                out.append((sig_name, f"{int(value)}" if scale == 1.0 else f"{value:.2f}", unit))
        return out

    def summary(self) -> str:
        decoded = self.decode()
        if not decoded:
            return self.error_reason if self.is_error else ""
        parts = []
        for name, val, unit in decoded[:2]:
            parts.append(f"{name}={val}{unit}" if unit else f"{name}={val}")
        return ", ".join(parts)

@dataclass
class NodeStats:
    tx_total: int = 0
    rx_total: int = 0
    errors:   int = 0
    last_tx_time: float = 0.0
    last_rx_time: float = 0.0

@dataclass
class MessageStats:
    count: int = 0
    errors: int = 0
    last_time: float = 0.0
    intervals: deque = field(default_factory=lambda: deque(maxlen=64))

    def record(self, ts: float, is_error: bool) -> None:
        if self.last_time > 0:
            self.intervals.append(ts - self.last_time)
        self.last_time = ts
        self.count += 1
        if is_error:
            self.errors += 1

    @property
    def actual_cycle_ms(self) -> Optional[float]:
        if not self.intervals:
            return None
        return (sum(self.intervals) / len(self.intervals)) * 1000.0

    @property
    def jitter_ms(self) -> Optional[float]:
        if len(self.intervals) < 2:
            return None
        mean = sum(self.intervals) / len(self.intervals)
        max_dev = max(abs(i - mean) for i in self.intervals)
        return max_dev * 1000.0

# ============================================================================
# BUS BACKENDS
# ============================================================================
class SimulatedBus:
    def __init__(self, out_queue: queue.Queue):
        self.q = out_queue
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.target_rpm = 1500
        self.actual_rpm = 0
        self.temp_c = 25.0
        self.running_motor = False
        self.heartbeat_counter = 0
        self.uptime_start = time.time()
        self.next_send = {cid: time.time() for cid in MESSAGES}
        self.next_send[0x100] = time.time() + 2.0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running:
            now = time.time()
            if now >= self.next_send[0x100]:
                self._emit_motor_cmd(now)
                self.next_send[0x100] = now + random.uniform(3.0, 6.0)
            if now >= self.next_send[0x200]:
                self._emit_sensor(now)
                self.next_send[0x200] = now + 0.1
            if now >= self.next_send[0x300]:
                self._emit_heartbeat(now)
                self.next_send[0x300] = now + 1.0
            time.sleep(0.005)

    def _emit_motor_cmd(self, ts):
        choice = random.choice([
            (1, 1, random.choice([1200, 1800, 2400, 3000]), 50),  
            (3, 1, random.choice([800, 1500, 2200]), 80),         
            (2, 0, 0, 200),                                       
        ])
        cmd, direction, rpm, accel = choice
        if cmd == 1:
            self.running_motor = True
            self.target_rpm = rpm
        elif cmd == 2:
            self.running_motor = False
            self.target_rpm = 0
        elif cmd == 3:
            self.target_rpm = rpm
        try:
            data = struct.pack("<BBHB3x", cmd, direction, rpm, accel)
            self.q.put(CanFrame(ts, 0x100, data))
        except struct.error:
            pass

    def _emit_sensor(self, ts):
        if self.running_motor:
            if self.actual_rpm < self.target_rpm:
                self.actual_rpm = min(self.actual_rpm + 80, self.target_rpm)
            elif self.actual_rpm > self.target_rpm:
                self.actual_rpm = max(self.actual_rpm - 80, self.target_rpm)
            self.temp_c = min(self.temp_c + 0.15, 85.0)
        else:
            self.actual_rpm = max(self.actual_rpm - 150, 0)
            self.temp_c = max(self.temp_c - 0.08, 22.0)

        current_a = self.actual_rpm / 1000.0
        flags = 0
        if self.running_motor:        flags |= 0b001
        if self.temp_c   > 75:        flags |= 0b010
        if current_a     > 4.5:       flags |= 0b100

        if random.random() < 0.01:
            bad = bytes([random.randint(0, 255) for _ in range(8)])
            self.q.put(CanFrame(ts, 0x200, bad, is_error=True,
                                error_reason="CRC mismatch"))
            return

        try:
            data = struct.pack("<HhBB2x",
                               int(self.actual_rpm),
                               int(self.temp_c * 10),
                               int(current_a * 10),
                               flags)
            self.q.put(CanFrame(ts, 0x200, data))
        except struct.error:
            pass

    def _emit_heartbeat(self, ts):
        self.heartbeat_counter = (self.heartbeat_counter + 1) % 256
        uptime = int(ts - self.uptime_start)
        try:
            data = struct.pack("<BH5x", self.heartbeat_counter, uptime & 0xFFFF)
            self.q.put(CanFrame(ts, 0x300, data))
        except struct.error:
            pass

def connect_real_bus(out_queue: queue.Queue):
    import can  
    bus = can.interface.Bus(bustype="socketcan", channel="can0", bitrate=BUS_BITRATE)
    def reader():
        for msg in bus:
            out_queue.put(CanFrame(
                timestamp=msg.timestamp,
                can_id=msg.arbitration_id,
                data=bytes(msg.data),
                is_error=msg.is_error_frame,
                error_reason="error frame" if msg.is_error_frame else "",
            ))
    t = threading.Thread(target=reader, daemon=True)
    t.start()
    return bus

# ============================================================================
# GUI - AHORA COMO FRAME INCRUSTABLE
# ============================================================================
class CanMonitor(tk.Frame):
    def __init__(self, parent, **kwargs):
        kwargs.setdefault("bg", COLORS["bg"])
        super().__init__(parent, **kwargs)
        
        self._last_width = 1240
        self.bind("<Configure>", self._on_window_resize)

        self.frame_queue: queue.Queue = queue.Queue()
        self.log: deque = deque(maxlen=5000)
        self.stats: dict = defaultdict(NodeStats)
        self.msg_stats: dict = defaultdict(MessageStats)
        self.total_messages = 0
        self.error_count = 0
        self.rate_window = deque(maxlen=500)
        self.paused = False
        self.autoscroll = True
        self.filter_state = {
            "ids":        set(),    
            "sources":    set(),    
            "frame_type": "all",
            "text":       "",
        }
        self.selected_frame: Optional[CanFrame] = None
        self.dirty = True
        self.tree_iid_by_frame: dict = {}

        self._setup_styles()
        self._build_layout()

        if USE_REAL_BUS:
            try:
                self.bus = connect_real_bus(self.frame_queue)
                self.bus_label_text = "Real bus"
            except Exception as e:
                print(f"Could not open real bus:\n{e}\n\nFalling back to simulator.")
                self.bus = SimulatedBus(self.frame_queue)
                self.bus.start()
                self.bus_label_text = "Simulator (fallback)"
        else:
            self.bus = SimulatedBus(self.frame_queue)
            self.bus.start()
            self.bus_label_text = "Simulator"
        self.mode_label.config(text=self.bus_label_text)

        self.after(20,  self._drain_queue)
        self.after(150, self._refresh_ui)
        self.after(200, self._refresh_supervision)
        self.after(50, lambda: self._apply_responsive_columns(self.winfo_width()))

    def _setup_styles(self):
        self.font_mono = font.Font(family="Courier", size=10)
        self.font_mono_bold = font.Font(family="Courier", size=10, weight="bold")
        self.font_ui = font.Font(family="Helvetica", size=10)
        self.font_ui_bold = font.Font(family="Helvetica", size=10, weight="bold")
        self.font_header = font.Font(family="Helvetica", size=11, weight="bold")
        self.font_small = font.Font(family="Helvetica", size=9)

        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        
        style.configure("Treeview",
                        background=COLORS["panel"],
                        foreground=COLORS["text"],
                        fieldbackground=COLORS["panel"],
                        rowheight=22,
                        font=self.font_mono,
                        borderwidth=0)
        style.configure("Treeview.Heading",
                        background=COLORS["panel_alt"],
                        foreground=COLORS["text_dim"],
                        font=self.font_ui_bold,
                        relief="flat",
                        borderwidth=0,
                        padding=(6, 4))
        style.map("Treeview.Heading", background=[("active", COLORS["panel_alt"])])
        style.map("Treeview",
                  background=[("selected", COLORS["highlight"])],
                  foreground=[("selected", COLORS["text"])])

    def _build_layout(self):
        self._build_toolbar(self)
        self._build_supervision_panel(self)
        
        self.body = tk.PanedWindow(self, orient="horizontal",
                                   bg=COLORS["bg"], sashwidth=6,
                                   sashrelief="flat",
                                   bd=0, opaqueresize=True)
        self.body.pack(fill="both", expand=True, padx=8, pady=4)
        self._build_log(self.body)
        self._build_sidebar(self.body)

    def _build_toolbar(self, parent):
        bar = tk.Frame(parent, bg=COLORS["panel"], height=48)
        bar.pack(fill="x", padx=8, pady=(8, 4))
        bar.pack_propagate(False)

        dot = tk.Canvas(bar, width=14, height=14, bg=COLORS["panel"],
                        highlightthickness=0)
        dot.create_oval(2, 2, 12, 12, fill=COLORS["rx"], outline="")
        dot.pack(side="left", padx=(14, 6))
        tk.Label(bar, text="Bus online", bg=COLORS["panel"],
                 fg=COLORS["text"], font=self.font_ui_bold).pack(side="left")

        self.mode_label = tk.Label(bar, text="Simulator",
                                   bg=COLORS["panel"], fg=COLORS["tx"],
                                   font=self.font_small)
        self.mode_label.pack(side="left", padx=(8, 0))

        def stat_label(text):
            lbl = tk.Label(bar, text=text, bg=COLORS["panel"],
                           fg=COLORS["text_dim"], font=self.font_ui)
            lbl.pack(side="left", padx=(20, 0))
            return lbl

        self.lbl_bitrate = stat_label(f"{BUS_BITRATE//1000} kbit/s")
        self.lbl_rate    = stat_label("0 msg/s")
        self.lbl_total   = stat_label("Total: 0")
        self.lbl_errors  = stat_label("Errors: 0")

        def make_btn(p, text, cmd, color=None):
            return tk.Button(p, text=text, command=cmd,
                             bg=COLORS["panel_alt"],
                             fg=color or COLORS["text"],
                             font=self.font_ui, relief="flat", padx=12, pady=2,
                             activebackground=COLORS["border"],
                             activeforeground=COLORS["text"])

        make_btn(bar, "Clear", self._clear_log).pack(side="right", padx=(0, 8))
        self.pause_btn = make_btn(bar, "Pause", self._toggle_pause)
        self.pause_btn.pack(side="right", padx=(0, 4))

        self.filter_btn = tk.Button(bar, text="Filter ▾",
                                    command=self._open_filter_popup,
                                    bg=COLORS["panel_alt"], fg=COLORS["text"],
                                    font=self.font_ui, relief="flat",
                                    padx=12, pady=2,
                                    activebackground=COLORS["border"],
                                    activeforeground=COLORS["text"])
        self.filter_btn.pack(side="right", padx=(0, 4))
        self.filter_indicator = tk.Canvas(bar, width=10, height=10,
                                          bg=COLORS["panel"],
                                          highlightthickness=0)
        self.filter_indicator_oval = self.filter_indicator.create_oval(
            1, 1, 9, 9, fill=COLORS["panel"], outline="")
        self.filter_indicator.pack(side="right", padx=(0, 4))

    def _build_supervision_panel(self, parent):
        frame = tk.Frame(parent, bg=COLORS["panel"])
        frame.pack(fill="x", padx=8, pady=4)

        header = tk.Frame(frame, bg=COLORS["panel"])
        header.pack(fill="x", padx=14, pady=(8, 4))
        tk.Label(header, text="Message supervision",
                 bg=COLORS["panel"], fg=COLORS["text"],
                 font=self.font_header).pack(side="left")

        tk.Label(header,
                 text=f"timeout = {TIMEOUT_FACTOR:g}× nominal cycle",
                 bg=COLORS["panel"], fg=COLORS["text_dim"],
                 font=self.font_small).pack(side="right")

        self.node_state_frame = tk.Frame(frame, bg=COLORS["panel"])
        self.node_state_frame.pack(fill="x", padx=14, pady=(0, 6))
        self.node_state_labels = {}
        for name in DEVICES:
            row = tk.Frame(self.node_state_frame, bg=COLORS["panel"])
            row.pack(side="left", padx=(0, 24))
            tk.Label(row, text=f"{name}:", bg=COLORS["panel"],
                     fg=COLORS["text_dim"], font=self.font_ui
                     ).pack(side="left", padx=(0, 6))
            badge = tk.Label(row, text="UNCOMMISSIONED",
                             bg=COLORS["panel"], fg=COLORS["idle"],
                             font=self.font_ui_bold)
            badge.pack(side="left")
            self.node_state_labels[name] = badge

        table_frame = tk.Frame(frame, bg=COLORS["panel"])
        table_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        cols = ("state", "id", "name", "src",
                "nom", "act", "jit", "count", "last")
        self.sup_tree = ttk.Treeview(table_frame, columns=cols,
                                     show="headings", height=max(3, len(MESSAGES)))
        col_specs = [
            ("state",  "State",            110, "w"),
            ("id",     "ID",                60, "w"),
            ("name",   "Name",              95, "w"),
            ("src",    "Source",            85, "w"),
            ("nom",    "Nominal",           75, "e"),
            ("act",    "Actual cycle",      90, "e"),
            ("jit",    "Jitter",            70, "e"),
            ("count",  "Count",             70, "e"),
            ("last",   "Last seen",         85, "e"),
        ]
        for col, label, width, anchor in col_specs:
            self.sup_tree.heading(col, text=label)
            self.sup_tree.column(col, width=width, anchor=anchor, stretch=False)

        self.sup_tree.tag_configure("nominal",         foreground=COLORS["rx"])
        self.sup_tree.tag_configure("sporadic",        foreground=COLORS["rx"])
        self.sup_tree.tag_configure("unstable",        foreground=COLORS["err"])
        self.sup_tree.tag_configure("overdue",         foreground=COLORS["err"],
                                    background=COLORS["err_row"])
        self.sup_tree.tag_configure("fault",           foreground=COLORS["err"],
                                    background=COLORS["err_row"])
        self.sup_tree.tag_configure("absent",          foreground=COLORS["text_dim"])
        self.sup_tree.tag_configure("dormant",         foreground=COLORS["text_dim"])
        self.sup_tree.tag_configure("uncommissioned",  foreground=COLORS["text_dim"])
        self.sup_tree.pack(fill="both", expand=True)

        self.sup_iids = {}
        for cid in MESSAGES:
            iid = f"id_{cid:03X}"
            self.sup_iids[cid] = iid
            initial = "ABSENT" if MESSAGES[cid].get("cycle_ms") else "UNCOMMISSIONED"
            tag = "absent" if MESSAGES[cid].get("cycle_ms") else "uncommissioned"
            self.sup_tree.insert("", "end", iid=iid,
                                 values=(initial, f"0x{cid:03X}",
                                         MESSAGES[cid]["name"],
                                         MESSAGES[cid]["sender"],
                                         "—", "—", "—", "0", "never"),
                                 tags=(tag,))

    def _build_log(self, parent):
        left = tk.Frame(parent, bg=COLORS["panel"])
        parent.add(left, minsize=320, stretch="always")

        header = tk.Frame(left, bg=COLORS["panel"])
        header.pack(fill="x", padx=12, pady=(8, 4))
        tk.Label(header, text="Message log",
                 bg=COLORS["panel"], fg=COLORS["text"],
                 font=self.font_header).pack(side="left")
        self.log_count_label = tk.Label(header, text="",
                                         bg=COLORS["panel"],
                                         fg=COLORS["text_dim"],
                                         font=self.font_small)
        self.log_count_label.pack(side="right")

        log_frame = tk.Frame(left, bg=COLORS["panel"])
        log_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        columns = ("time", "id", "name", "src", "dst", "dlc", "data", "decoded")
        self.tree = ttk.Treeview(log_frame, columns=columns,
                                 show="headings")
        self.tree_col_specs = [
            ("time",    "Timestamp",    105,  0),
            ("id",      "ID",            62,  0),
            ("name",    "Name",          90,  0),
            ("src",     "From",          85,  0),
            ("dst",     "To",            95,  0),
            ("dlc",     "DLC",           38,  0),
            ("data",    "Data (hex)",   195,  1),
            ("decoded", "Decoded",      170,  2),
        ]
        for col, label, width, _stretch in self.tree_col_specs:
            self.tree.heading(col, text=label)
            anchor = "e" if col == "dlc" else "w"
            self.tree.column(col, width=width, minwidth=40, anchor=anchor,
                             stretch=(col == "decoded"))

        self.tree.tag_configure("tx_cmd", background=COLORS["tx_row"])
        self.tree.tag_configure("err",    background=COLORS["err_row"])
        self.tree.tag_configure("alt",    background=COLORS["row_alt"])
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<MouseWheel>", self._on_scroll)
        self.tree.bind("<Button-4>",   self._on_scroll)
        self.tree.bind("<Button-5>",   self._on_scroll)

        vsb = ttk.Scrollbar(log_frame, orient="vertical",
                            command=self._on_scroll_cmd)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

    def _on_scroll(self, _event):
        self.autoscroll = False

    def _on_scroll_cmd(self, *args):
        self.autoscroll = False
        self.tree.yview(*args)

    def _build_sidebar(self, parent):
        right = tk.Frame(parent, bg=COLORS["bg"])
        parent.add(right, minsize=300, stretch="never", width=400)

        det = tk.Frame(right, bg=COLORS["panel"])
        det.pack(fill="both", expand=True, pady=(0, 8))
        tk.Label(det, text="Frame detail", bg=COLORS["panel"],
                 fg=COLORS["text"], font=self.font_header
                 ).pack(anchor="w", padx=14, pady=(8, 4))
        self.detail_text = tk.Text(det, bg=COLORS["panel"],
                                   fg=COLORS["text_mono"], font=self.font_mono,
                                   relief="flat", padx=14, pady=4, height=12,
                                   wrap="word", state="disabled",
                                   insertbackground=COLORS["text"])
        self.detail_text.tag_configure("h",       foreground=COLORS["text"],
                                       font=self.font_mono_bold)
        self.detail_text.tag_configure("dim",     foreground=COLORS["text_dim"])
        self.detail_text.tag_configure("hex_rpm", background="#1e3a5f")
        self.detail_text.tag_configure("hex_thr", background="#1e4a3a")
        self.detail_text.tag_configure("hex_tmp", background="#5a3a1a")
        self.detail_text.tag_configure("err",     foreground=COLORS["err"])
        self.detail_text.pack(fill="both", expand=True, padx=4, pady=(0, 8))

        stats = tk.Frame(right, bg=COLORS["panel"])
        stats.pack(fill="x")
        tk.Label(stats, text="Per-device stats", bg=COLORS["panel"],
                 fg=COLORS["text"], font=self.font_header
                 ).pack(anchor="w", padx=14, pady=(8, 4))

        cols = ("node", "tx", "rx", "err", "last")
        self.stats_tree = ttk.Treeview(stats, columns=cols, show="headings",
                                       height=max(2, len(DEVICES)))
        for col, label, width, anchor, stretch in [
            ("node", "Node",      95, "w", True),
            ("tx",   "TX/s",      55, "e", False),
            ("rx",   "RX/s",      55, "e", False),
            ("err",  "Errors",    60, "e", False),
            ("last", "Last seen", 80, "e", False),
        ]:
            self.stats_tree.heading(col, text=label)
            self.stats_tree.column(col, width=width, minwidth=40,
                                   anchor=anchor, stretch=stretch)
        self.stats_tree.tag_configure("err_node", foreground=COLORS["err"])
        self.stats_tree.tag_configure("ok_node",  foreground=COLORS["rx"])
        self.stats_tree.pack(fill="x", padx=4, pady=(0, 8))

        ids = tk.Frame(right, bg=COLORS["panel"])
        ids.pack(fill="both", expand=False, pady=(8, 0))
        tk.Label(ids, text="Known message IDs", bg=COLORS["panel"],
                 fg=COLORS["text"], font=self.font_header
                 ).pack(anchor="w", padx=14, pady=(8, 4))
        for cid, info in MESSAGES.items():
            row = tk.Frame(ids, bg=COLORS["panel"])
            row.pack(fill="x", padx=14, pady=2)
            tk.Label(row, text=f"0x{cid:03X}", bg=COLORS["panel"],
                     fg=COLORS["tx"], font=self.font_mono_bold, width=7,
                     anchor="w").pack(side="left")
            tk.Label(row, text=info["name"], bg=COLORS["panel"],
                     fg=COLORS["text"], font=self.font_mono, width=12,
                     anchor="w").pack(side="left")
            tk.Label(row,
                     text=f"{info['sender']} → {', '.join(info['receivers'])}",
                     bg=COLORS["panel"], fg=COLORS["text_dim"],
                     font=self.font_small, anchor="w"
                     ).pack(side="left", fill="x", expand=True)
        tk.Frame(ids, bg=COLORS["panel"], height=8).pack(fill="x")

    def _on_window_resize(self, event):
        if event.widget is not self:
            return
        w = event.width
        if abs(w - self._last_width) < 30:
            return
        self._last_width = w
        self._apply_responsive_columns(w)

    def _apply_responsive_columns(self, window_width: int):
        if window_width >= 1100:
            cols = ("time", "id", "name", "src", "dst", "dlc", "data", "decoded")
        elif window_width >= 850:
            cols = ("time", "id", "name", "src", "dst", "dlc", "data")
        elif window_width >= 700:
            cols = ("time", "id", "src", "dst", "dlc", "data")
        else:
            cols = ("time", "id", "src", "data")

        self.tree["displaycolumns"] = cols
        for col, _label, _w, _s in self.tree_col_specs:
            self.tree.column(col, stretch=(col == cols[-1]))

    def _toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.config(text="Resume" if self.paused else "Pause",
                              fg=COLORS["err"] if self.paused else COLORS["text"])

    def _clear_log(self):
        self.log.clear()
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        self.tree_iid_by_frame.clear()
        self.total_messages = 0
        self.error_count = 0
        self.rate_window.clear()
        self.autoscroll = True

    def _open_filter_popup(self):
        if hasattr(self, "_filter_popup") and self._filter_popup.winfo_exists():
            self._filter_popup.lift()
            self._filter_popup.focus_force()
            return

        popup = tk.Toplevel(self)
        self._filter_popup = popup
        popup.title("Filter messages")
        popup.configure(bg=COLORS["panel"])
        # Cambiamos transient a winfo_toplevel porque ahora esto es un Frame
        popup.transient(self.winfo_toplevel())
        popup.resizable(False, False)

        self.update_idletasks()
        btn_x = self.filter_btn.winfo_rootx()
        btn_y = self.filter_btn.winfo_rooty() + self.filter_btn.winfo_height() + 4
        pw = 340
        right_edge = btn_x + self.filter_btn.winfo_width()
        x = max(20, right_edge - pw)
        popup.geometry(f"{pw}x520+{x}+{btn_y}")

        popup.bind("<Escape>", lambda _e: self._cancel_filter_popup())
        popup.protocol("WM_DELETE_WINDOW", self._cancel_filter_popup)

        body = tk.Frame(popup, bg=COLORS["panel"])
        body.pack(fill="both", expand=True, padx=14, pady=12)

        self._pf_id_vars     = {}
        self._pf_unknown_var = tk.BooleanVar(value=False)
        self._pf_source_vars = {}
        self._pf_frame_type  = tk.StringVar(value=self.filter_state["frame_type"])
        self._pf_text_var    = tk.StringVar(value=self.filter_state["text"])

        self._popup_section_header(body, "Message ID")
        for cid, info in MESSAGES.items():
            var = tk.BooleanVar(value=(cid in self.filter_state["ids"]))
            self._pf_id_vars[cid] = var
            label = f"0x{cid:03X}    {info['name']}   ({info['sender']} → " \
                    f"{', '.join(info['receivers'])})"
            self._popup_check(body, label, var)
        self._pf_unknown_var.set(-1 in self.filter_state["ids"])
        self._popup_check(body, "Unknown / undefined IDs", self._pf_unknown_var)

        self._popup_section_header(body, "Source node")
        for name, desc in DEVICES.items():
            var = tk.BooleanVar(value=(name in self.filter_state["sources"]))
            self._pf_source_vars[name] = var
            self._popup_check(body, f"{name}   ({desc})", var)

        self._popup_section_header(body, "Frame type")
        for value, label in [
            ("all",   "All frames"),
            ("data",  "Data frames only"),
            ("error", "Error frames only"),
        ]:
            self._popup_radio(body, label, self._pf_frame_type, value)

        self._popup_section_header(body, "Text search (optional)")
        entry = tk.Entry(body, textvariable=self._pf_text_var,
                         bg=COLORS["panel_alt"], fg=COLORS["text"],
                         insertbackground=COLORS["text"],
                         relief="flat", font=self.font_ui)
        entry.pack(fill="x", padx=(20, 0), ipady=5, pady=(2, 4))
        tk.Label(body,
                 text="Matches against ID, name, source, destination, and decoded signals.",
                 bg=COLORS["panel"], fg=COLORS["text_dim"],
                 font=self.font_small, anchor="w", justify="left",
                 wraplength=280
                 ).pack(fill="x", padx=(20, 0))

        btn_row = tk.Frame(popup, bg=COLORS["panel"])
        btn_row.pack(fill="x", padx=14, pady=(0, 14))
        tk.Button(btn_row, text="Clear all", command=self._clear_filter_popup,
                  bg=COLORS["panel_alt"], fg=COLORS["text"],
                  font=self.font_ui, relief="flat", padx=14, pady=4,
                  activebackground=COLORS["border"]
                  ).pack(side="left")
        tk.Button(btn_row, text="Cancel", command=self._cancel_filter_popup,
                  bg=COLORS["panel_alt"], fg=COLORS["text"],
                  font=self.font_ui, relief="flat", padx=14, pady=4,
                  activebackground=COLORS["border"]
                  ).pack(side="right", padx=(6, 0))
        tk.Button(btn_row, text="Apply", command=self._apply_filter_popup,
                  bg=COLORS["tx"], fg="white",
                  font=self.font_ui_bold, relief="flat", padx=14, pady=4,
                  activebackground=COLORS["highlight"],
                  activeforeground="white"
                  ).pack(side="right")

        popup.bind("<Return>", lambda _e: self._apply_filter_popup())

    def _popup_section_header(self, parent, text):
        tk.Label(parent, text=text, bg=COLORS["panel"],
                 fg=COLORS["text"], font=self.font_ui_bold, anchor="w"
                 ).pack(fill="x", pady=(8, 2))

    def _popup_check(self, parent, label, var):
        cb = tk.Checkbutton(parent, text=label, variable=var,
                            bg=COLORS["panel"], fg=COLORS["text"],
                            selectcolor=COLORS["panel_alt"],
                            activebackground=COLORS["panel"],
                            activeforeground=COLORS["text"],
                            font=self.font_ui, anchor="w",
                            highlightthickness=0, bd=0)
        cb.pack(fill="x", padx=(20, 0), pady=1)

    def _popup_radio(self, parent, label, var, value):
        rb = tk.Radiobutton(parent, text=label, variable=var, value=value,
                            bg=COLORS["panel"], fg=COLORS["text"],
                            selectcolor=COLORS["panel_alt"],
                            activebackground=COLORS["panel"],
                            activeforeground=COLORS["text"],
                            font=self.font_ui, anchor="w",
                            highlightthickness=0, bd=0)
        rb.pack(fill="x", padx=(20, 0), pady=1)

    def _clear_filter_popup(self):
        for v in self._pf_id_vars.values():
            v.set(False)
        self._pf_unknown_var.set(False)
        for v in self._pf_source_vars.values():
            v.set(False)
        self._pf_frame_type.set("all")
        self._pf_text_var.set("")

    def _cancel_filter_popup(self):
        if hasattr(self, "_filter_popup") and self._filter_popup.winfo_exists():
            self._filter_popup.destroy()

    def _apply_filter_popup(self):
        ids = {cid for cid, v in self._pf_id_vars.items() if v.get()}
        if self._pf_unknown_var.get():
            ids.add(-1)  
        sources = {n for n, v in self._pf_source_vars.items() if v.get()}

        self.filter_state = {
            "ids":        ids,
            "sources":    sources,
            "frame_type": self._pf_frame_type.get(),
            "text":       self._pf_text_var.get().strip().lower(),
        }

        self._update_filter_indicator()
        self.dirty = True
        self._cancel_filter_popup()

    def _update_filter_indicator(self):
        active = bool(self.filter_state["ids"]) or bool(self.filter_state["sources"]) or self.filter_state["frame_type"] != "all" or bool(self.filter_state["text"])
        self.filter_indicator.itemconfig(
            self.filter_indicator_oval,
            fill=COLORS["tx"] if active else COLORS["panel"])
        if active:
            fs = self.filter_state
            n = 0
            if fs["ids"]:        n += 1
            if fs["sources"]:    n += 1
            if fs["frame_type"] != "all": n += 1
            if fs["text"]:       n += 1
            self.filter_btn.config(text=f"Filter ({n}) ▾", fg=COLORS["tx"])
        else:
            self.filter_btn.config(text="Filter ▾", fg=COLORS["text"])

    def _matches_unknown(self, f: CanFrame) -> bool:
        return f.can_id not in MESSAGES

    def _matches_filter(self, f: CanFrame) -> bool:
        fs = self.filter_state
        if fs["ids"] and not ((f.can_id in fs["ids"]) or (-1 in fs["ids"] and f.can_id not in MESSAGES)): return False
        if fs["sources"] and f.source not in fs["sources"]: return False
        if fs["frame_type"] == "error" and not f.is_error: return False
        if fs["frame_type"] == "data" and f.is_error: return False
        if fs["text"]:
            hay = (f"0x{f.can_id:03X} {f.name} {f.source} {','.join(f.destinations)} {' '.join(f'{name}={val}{unit}' for name, val, unit in f.decode())} {f.error_reason}").lower()
            if fs["text"] not in hay: return False
        return True

    def _row_values(self, f: CanFrame):
        ts = time.strftime("%H:%M:%S", time.localtime(f.timestamp))
        ts += f".{int((f.timestamp % 1) * 1000):03d}"
        id_str = f"0x{f.can_id:03X}"
        data_hex = " ".join(f"{b:02X}" for b in f.data)
        return (ts, id_str, f.name, f.source, ", ".join(f.destinations),
                len(f.data), data_hex, f.summary())

    def _row_tags(self, f: CanFrame, alt: bool):
        if f.is_error:
            return ("err",)
        if f.can_id == 0x100:
            return ("tx_cmd",)
        return ("alt",) if alt else ()

    def _on_select(self, _event):
        sel = self.tree.selection()
        if not sel:
            return
        iid = sel[0]
        for f in self.log:
            if str(id(f)) == iid:
                self.selected_frame = f
                self._render_detail(f)
                break

    def destroy(self):
        try:
            if hasattr(self.bus, "stop"):
                self.bus.stop()
        except Exception:
            pass
        super().destroy()

    def _render_detail(self, f: CanFrame):
        self.detail_text.config(state="normal")
        self.detail_text.delete("1.0", "end")

        info = f.msg_info

        self.detail_text.insert("end", f"ID         ", "dim")
        self.detail_text.insert("end", f"0x{f.can_id:03X}\n", "h")
        self.detail_text.insert("end", f"Name       ", "dim")
        self.detail_text.insert("end", f"{f.name}\n", "h")
        self.detail_text.insert("end", f"From       ", "dim")
        self.detail_text.insert("end", f"{f.source}\n")
        self.detail_text.insert("end", f"To         ", "dim")
        self.detail_text.insert("end", f"{', '.join(f.destinations)}\n")
        self.detail_text.insert("end", f"DLC        ", "dim")
        self.detail_text.insert("end", f"{len(f.data)} bytes\n")
        cycle = info.get("cycle_ms", "—")
        self.detail_text.insert("end", f"Cycle      ", "dim")
        self.detail_text.insert("end", f"{cycle} ms\n")

        if f.is_error:
            self.detail_text.insert("end", f"\n!! ERROR FRAME — {f.error_reason}\n", "err")

        self.detail_text.insert("end", "\nRaw bytes\n", "h")
        self.detail_text.insert("end", "  Idx  ", "dim")
        self.detail_text.insert("end",
            "  ".join(f"{i}" for i in range(len(f.data))) + "\n", "dim")
        self.detail_text.insert("end", "  Hex  ", "dim")
        self.detail_text.insert("end",
            " ".join(f"{b:02X}" for b in f.data) + "\n")

        self.detail_text.insert("end", "\nDecoded signals\n", "h")
        decoded = f.decode()
        if decoded:
            for name, val, unit in decoded:
                line = f"  {name:<12} = {val}"
                if unit:
                    line += f" {unit}"
                self.detail_text.insert("end", line + "\n")
        else:
            self.detail_text.insert("end", "  (no signal map / undecodable)\n", "dim")

        if info.get("format"):
            self.detail_text.insert("end", "\nStruct format\n", "h")
            self.detail_text.insert("end", f"  {info['format']}\n", "dim")

        self.detail_text.config(state="disabled")

    def _drain_queue(self):
        processed = 0
        while processed < 200:
            try:
                f = self.frame_queue.get_nowait()
            except queue.Empty:
                break
            processed += 1
            if self.paused:
                continue

            self.log.append(f)
            self.total_messages += 1
            self.rate_window.append(f.timestamp)
            if f.is_error:
                self.error_count += 1

            s = self.stats[f.source]
            s.tx_total += 1
            s.last_tx_time = f.timestamp
            if f.is_error:
                s.errors += 1
            for dst in f.destinations:
                if dst in DEVICES:
                    rs = self.stats[dst]
                    rs.rx_total += 1
                    rs.last_rx_time = f.timestamp

            if f.can_id in MESSAGES:
                self.msg_stats[f.can_id].record(f.timestamp, f.is_error)

            self.dirty = True
        self.after(20, self._drain_queue)

    def _refresh_ui(self):
        now = time.time()
        recent = [t for t in self.rate_window if now - t < 1.0]
        self.lbl_rate.config(text=f"{len(recent)} msg/s")
        self.lbl_total.config(text=f"Total: {self.total_messages:,}")
        self.lbl_errors.config(
            text=f"Errors: {self.error_count}",
            fg=COLORS["err"] if self.error_count > 0 else COLORS["text_dim"])

        if self.dirty:
            self._refresh_log()
            self._refresh_stats()
            self.dirty = False

        self.after(150, self._refresh_ui)

    def _refresh_log(self):
        all_frames = list(self.log)
        if self.filter_state["ids"] or self.filter_state["sources"] or self.filter_state["frame_type"] != "all" or self.filter_state["text"]:
            visible = [f for f in all_frames if self._matches_filter(f)][-300:]
        else:
            visible = all_frames[-300:]

        new_ids = set()
        alt = False
        for f in visible:
            iid = str(id(f))
            new_ids.add(iid)
            values = self._row_values(f)
            tags = self._row_tags(f, alt)
            alt = not alt
            if iid in self.tree_iid_by_frame:
                self.tree.item(iid, values=values, tags=tags)
            else:
                self.tree.insert("", "end", iid=iid, values=values, tags=tags)
                self.tree_iid_by_frame[iid] = f

        to_remove = [iid for iid in self.tree_iid_by_frame if iid not in new_ids]
        for iid in to_remove:
            try:
                self.tree.delete(iid)
            except tk.TclError:
                pass
            self.tree_iid_by_frame.pop(iid, None)

        if self.autoscroll:
            kids = self.tree.get_children()
            if kids:
                self.tree.see(kids[-1])

        active = bool(self.filter_state["ids"]) or bool(self.filter_state["sources"]) or self.filter_state["frame_type"] != "all" or bool(self.filter_state["text"])
        self.log_count_label.config(
            text=f"Showing {len(visible)} of {len(all_frames):,} "
                 f"{'(filtered)' if active else ''}")

    def _refresh_stats(self):
        now = time.time()
        tx_rate = defaultdict(int)
        rx_rate = defaultdict(int)
        for f in self.log:
            if now - f.timestamp <= 1.0:
                tx_rate[f.source] += 1
                for d in f.destinations:
                    if d in DEVICES:
                        rx_rate[d] += 1

        for iid in self.stats_tree.get_children():
            self.stats_tree.delete(iid)

        for name in DEVICES:
            s = self.stats[name]
            tag = ("err_node",) if s.errors > 0 else ("ok_node",) if s.tx_total > 0 else ()
            last_seen = "—"
            if s.last_tx_time > 0:
                age = now - s.last_tx_time
                last_seen = f"{age:.1f}s" if age < 99 else "old"
            self.stats_tree.insert("", "end",
                values=(name, tx_rate[name], rx_rate[name], s.errors, last_seen),
                tags=tag)

    @staticmethod
    def _format_age(age: float) -> str:
        if age < 1.0:    return f"{int(age * 1000)} ms"
        if age < 60.0:   return f"{age:.1f} s"
        if age < 3600.0: return f"{int(age / 60)} min"
        return f"{int(age / 3600)} h"

    def _evaluate_message(self, can_id: int, now: float) -> dict:
        info = MESSAGES[can_id]
        ms = self.msg_stats.get(can_id)
        nominal_ms = info.get("cycle_ms")  

        if ms is None or ms.last_time == 0:
            return {
                "state": "ABSENT" if nominal_ms else "UNCOMMISSIONED",
                "tag":   "absent" if nominal_ms else "uncommissioned",
                "nominal":  f"{nominal_ms:.0f} ms" if nominal_ms else "event",
                "actual":   "—",
                "jitter":   "—",
                "count":    "0",
                "last":     "never",
            }

        age = now - ms.last_time
        actual_ms = ms.actual_cycle_ms
        jitter_ms = ms.jitter_ms

        if nominal_ms is None:
            state = "SPORADIC" if age < EVENT_ACTIVE_WINDOW_S else "DORMANT"
            tag   = "sporadic" if state == "SPORADIC" else "dormant"
            return {
                "state":   state,
                "tag":     tag,
                "nominal": "event",
                "actual":  f"{actual_ms:.0f} ms" if actual_ms else "—",
                "jitter":  f"±{jitter_ms:.1f} ms" if jitter_ms else "—",
                "count":   str(ms.count),
                "last":    self._format_age(age),
            }

        timeout_s = (nominal_ms / 1000.0) * TIMEOUT_FACTOR

        if ms.errors > 0 and age < timeout_s * 2:
            state, tag = "FAULT", "fault"
        elif age > timeout_s:
            state, tag = "OVERDUE", "overdue"
        elif (actual_ms is not None
              and abs(actual_ms - nominal_ms) / nominal_ms > JITTER_WARN_FRACTION):
            state, tag = "UNSTABLE", "unstable"
        else:
            state, tag = "NOMINAL", "nominal"

        return {
            "state":   state,
            "tag":     tag,
            "nominal": f"{nominal_ms:.0f} ms",
            "actual":  f"{actual_ms:.1f} ms" if actual_ms else "—",
            "jitter":  f"±{jitter_ms:.1f} ms" if jitter_ms else "—",
            "count":   str(ms.count),
            "last":    self._format_age(age),
        }

    def _aggregate_node_state(self, node: str,
                              per_msg: dict) -> tuple:
        my_ids = [cid for cid, info in MESSAGES.items() if info["sender"] == node]
        if not my_ids:
            return "—", COLORS["text_dim"]

        states = [per_msg[cid]["state"] for cid in my_ids]

        if all(s in ("ABSENT", "UNCOMMISSIONED") for s in states):
            return "UNCOMMISSIONED", COLORS["idle"]
        if all(s in ("OVERDUE", "ABSENT", "DORMANT") for s in states):
            return "OFFLINE", COLORS["err"]
        if any(s in ("OVERDUE", "FAULT", "UNSTABLE", "ABSENT") for s in states):
            return "DEGRADED", COLORS["err"]
        return "OPERATIONAL", COLORS["rx"]

    def _refresh_supervision(self):
        now = time.time()
        per_msg = {cid: self._evaluate_message(cid, now) for cid in MESSAGES}

        for cid, info in MESSAGES.items():
            r = per_msg[cid]
            iid = self.sup_iids[cid]
            self.sup_tree.item(iid,
                values=(r["state"], f"0x{cid:03X}", info["name"], info["sender"],
                        r["nominal"], r["actual"], r["jitter"],
                        r["count"], r["last"]),
                tags=(r["tag"],))

        for node, badge in self.node_state_labels.items():
            state, color = self._aggregate_node_state(node, per_msg)
            badge.config(text=state, fg=color)

        self.after(200, self._refresh_supervision)

# Este bloque te permite probar el monitor CAN solo (sin el GUI principal) 
# si ejecutas el archivo CAN_Monitor_V7.py directamente.
if __name__ == "__main__":
    root = tk.Tk()
    root.title("CAN Bus Monitor (Standalone)")
    root.geometry("1240x820")
    root.configure(bg=COLORS["bg"])
    app = CanMonitor(root)
    app.pack(fill="both", expand=True)
    root.mainloop()