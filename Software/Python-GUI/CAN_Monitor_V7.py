import tkinter as tk
from tkinter import ttk
import time
import random
import struct
import threading
import queue
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Optional

# ============================================================================
# CONFIGURATION
# ============================================================================
USE_REAL_BUS = False   
BUS_BITRATE = 500_000

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
        "format":    "<BH5x",       
        "signals": [
            ("Counter", "B",  1.0,    0,    ""),
            ("Uptime",  "H",  1.0,    0,    "s"),
        ],
    },
}

COMMAND_LABELS = {1: "START", 2: "STOP", 3: "SET_SPEED"}
DIRECTION_LABELS = {0: "stop", 1: "fwd", 2: "rev"}
SENSOR_FLAGS = [(0b001, "RUN"), (0b010, "OVERHEAT"), (0b100, "OVERCURRENT")]

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
    is_error: bool = False
    error_reason: str = ""

    @property
    def msg_info(self) -> dict: return MESSAGES.get(self.can_id, {})

    @property
    def source(self) -> str: return self.msg_info.get("sender", "?")

    @property
    def destinations(self) -> list: return self.msg_info.get("receivers", ["?"])

    @property
    def name(self) -> str:
        return self.msg_info.get("name", f"Unknown_0x{self.can_id:03X}")

    def decode(self) -> list:
        info = self.msg_info
        if not info or self.is_error: return []
        try:
            values = struct.unpack(info["format"], self.data)
        except struct.error: return []

        out = []
        for (sig_name, _, scale, offset, unit), raw in zip(info["signals"], values):
            value = raw * scale + offset
            if self.can_id == 0x100 and sig_name == "Command":
                out.append((sig_name, COMMAND_LABELS.get(int(value), str(int(value))), ""))
            elif self.can_id == 0x100 and sig_name == "Direction":
                out.append((sig_name, DIRECTION_LABELS.get(int(value), str(int(value))), ""))
            elif self.can_id == 0x200 and sig_name == "Flags":
                bits = [name for mask, name in SENSOR_FLAGS if int(value) & mask]
                out.append((sig_name, "|".join(bits) if bits else "—", ""))
            elif unit in ["°C", "A"]: out.append((sig_name, f"{value:.1f}", unit))
            else: out.append((sig_name, f"{int(value)}" if scale == 1.0 else f"{value:.2f}", unit))
        return out

    def summary(self) -> str:
        decoded = self.decode()
        if not decoded: return self.error_reason if self.is_error else ""
        return ", ".join(f"{n}={v}{u}" for n, v, u in decoded[:2])

class MessageStats:
    def __init__(self):
        self.count = 0
        self.last_time = 0.0

# ============================================================================
# SIMULATED BUS 
# ============================================================================
class SimulatedBus:
    def __init__(self, out_queue: queue.Queue):
        self.q = out_queue
        self.running = False
        self.target_rpm, self.actual_rpm = 1500, 0
        self.temp_c = 25.0
        self.running_motor = False
        self.heartbeat_counter = 0

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        next_100 = next_200 = next_300 = time.time()
        while self.running:
            now = time.time()
            if now >= next_100:
                choice = random.choice([(1, 1, 1800, 50), (2, 0, 0, 200)])
                self.running_motor = (choice[0] == 1)
                self.target_rpm = choice[2]
                self.q.put(CanFrame(now, 0x100, struct.pack("<BBHB3x", *choice)))
                next_100 = now + random.uniform(4.0, 7.0)

            if now >= next_200:
                if self.running_motor:
                    self.actual_rpm = min(self.actual_rpm + 120, self.target_rpm)
                    self.temp_c = min(self.temp_c + 0.2, 78.8)
                else:
                    self.actual_rpm = max(self.actual_rpm - 200, 0)
                    self.temp_c = max(self.temp_c - 0.1, 24.5)
                
                flags = 1 if self.running_motor else 0
                if self.temp_c > 75: flags |= 2
                
                # Occasional Error Frame simulation for the filter to catch
                if random.random() < 0.02:
                    self.q.put(CanFrame(now, 0x200, b'\x00\xFF'*4, is_error=True, error_reason="CRC mismatch"))
                else:
                    data = struct.pack("<HhBB2x", int(self.actual_rpm), int(self.temp_c * 10), int(self.actual_rpm/1200*10), flags)
                    self.q.put(CanFrame(now, 0x200, data))
                next_200 = now + 0.1

            if now >= next_300:
                self.heartbeat_counter = (self.heartbeat_counter + 1) % 256
                self.q.put(CanFrame(now, 0x300, struct.pack("<BH5x", self.heartbeat_counter, self.heartbeat_counter * 3)))
                next_300 = now + 1.0
            time.sleep(0.01)

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
        
        # Advanced Filter State
        self.filter_state = {
            "ids":        set(),    
            "sources":    set(),    
            "frame_type": "all",
            "text":       "",
        }

        self._setup_styles()
        self._build_layout()

        self.bus = SimulatedBus(self.frame_queue)
        self.bus.start()

        self.after(20, self._drain_queue)
        self.after(100, self._refresh_ui)

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

        dot = tk.Canvas(bar, width=12, height=12, bg=COLORS["panel"], highlightthickness=0)
        dot.create_oval(2, 2, 10, 10, fill=COLORS["rx"], outline="")
        dot.pack(side="left", padx=(14, 6), pady=12)
        
        tk.Label(bar, text="BUS ACTIVE (SIMULATED)", bg=COLORS["panel"], fg=COLORS["text"], font=self.font_mono_bold).pack(side="left")
        self.lbl_total = tk.Label(bar, text="TOTAL: 0", bg=COLORS["panel"], fg=COLORS["text_dim"], font=self.font_mono)
        self.lbl_total.pack(side="left", padx=20)

        # Right side controls
        tk.Button(bar, text="CLEAR", command=self._clear_log, bg=COLORS["panel_alt"], fg=COLORS["text"], font=self.font_mono_bold, relief="flat", bd=0, padx=12, pady=4, cursor="hand2").pack(side="right", padx=(0, 10), pady=6)
        
        self.pause_btn = tk.Button(bar, text="PAUSE", command=self._toggle_pause, bg=COLORS["panel_alt"], fg=COLORS["text"], font=self.font_mono_bold, relief="flat", bd=0, padx=12, pady=4, cursor="hand2")
        self.pause_btn.pack(side="right", padx=(0, 4), pady=6)
        
        # Filter button with popup logic
        self.filter_btn = tk.Button(bar, text="FILTER ▾", command=self._open_filter_popup, bg=COLORS["panel_alt"], fg=COLORS["text"], font=self.font_mono_bold, relief="flat", bd=0, padx=12, pady=4, cursor="hand2")
        self.filter_btn.pack(side="right", padx=(0, 4), pady=6)
        
        self.filter_indicator = tk.Canvas(bar, width=10, height=10, bg=COLORS["panel"], highlightthickness=0)
        self.filter_indicator_oval = self.filter_indicator.create_oval(1, 1, 9, 9, fill=COLORS["panel"], outline="")
        self.filter_indicator.pack(side="right", padx=(0, 4))

        # --- SUPERVISION NODE ---
        sup_frame = tk.Frame(self, bg=COLORS["panel"], highlightbackground=COLORS["border"], highlightthickness=1)
        sup_frame.pack(fill="x", padx=18, pady=(0, 8))
        tk.Label(sup_frame, text="SUPERVISION NODE", bg=COLORS["panel"], fg=COLORS["text"], font=self.font_header).pack(anchor="w", padx=14, pady=(6, 4))

        self.sup_tree = ttk.Treeview(sup_frame, columns=("id", "name", "src", "count", "last"), show="headings", height=3)
        for col, lbl, w, anch in [("id", "ID", 70, "w"), ("name", "NAME", 130, "w"), ("src", "SOURCE", 120, "w"), ("count", "COUNT", 80, "e"), ("last", "LAST SEEN", 140, "e")]:
            self.sup_tree.heading(col, text=lbl)
            self.sup_tree.column(col, width=w, anchor=anch, stretch=False)
        self.sup_tree.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        
        for cid in MESSAGES:
            self.sup_tree.insert("", "end", iid=f"sup_{cid}", values=(f"0x{cid:03X}", MESSAGES[cid]["name"], MESSAGES[cid]["sender"], "0", "never"))

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
        self.tree.heading("name", text="NAME")
        self.tree.column("name", width=110, stretch=False)
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
        
        tk.Label(body, text="Matches ID, name, source, and decoded signals.", bg=COLORS["panel"], fg=COLORS["text_dim"], font=("Courier New", 8), anchor="w").pack(fill="x", padx=(20, 0))

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
            self.filter_btn.config(text=f"FILTER ({n}) ▾", fg=COLORS["tx"])
        else:
            self.filter_btn.config(text="FILTER ▾", fg=COLORS["text"])

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

        for cid, stats in self.msg_stats.items():
            iid = f"sup_{cid}"
            if self.sup_tree.exists(iid):
                self.sup_tree.item(iid, values=(f"0x{cid:03X}", MESSAGES[cid]["name"], MESSAGES[cid]["sender"], str(stats.count), format_timestamp(stats.last_time)))

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
            f"───────────────────────────────",
            f"Timestamp : {format_timestamp(matched_frame.timestamp)}",
            f"Arb ID    : 0x{matched_frame.can_id:03X}",
            f"Msg Name  : {matched_frame.name}",
            f"Source    : {matched_frame.source}",
            f"Length    : {len(matched_frame.data)} bytes",
            f"Raw Hex   : {' '.join(f'{b:02X}' for b in matched_frame.data)}",
            f"\nDECODED SIGNALS:",
            f"───────────────────────────────"
        ]
        
        if matched_frame.is_error:
            lines.append(f"!! ERROR FRAME: {matched_frame.error_reason} !!")
        else:
            signals = matched_frame.decode()
            if signals:
                for name, val, unit in signals: lines.append(f" • {name:<12} : {val} {unit}".strip())
            else:
                lines.append(" No signal specification matching.")

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
        self.autoscroll = True
        self.detail_text.delete("1.0", tk.END)