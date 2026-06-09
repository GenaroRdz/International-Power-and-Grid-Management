import tkinter as tk
from tkinter import ttk
import time
import queue
from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Optional

# ============================================================================
# CONFIGURATION
# ============================================================================
# The monitor is fed ONLY by REAL CAN traffic coming from the MAIN ESP32 over
# the serial link (GUI_ECU._on_serial_line -> CanMonitor.feed_line). Every row
# you see is a frame the device actually put on the wire.
#
# HOW WE KNOW AN ECU IS CONNECTED
# -------------------------------
# A CAN bus gives one shared ACK bit, driven by ANY listening node, so a frame
# sent to a MISSING ECU still looks "delivered" if some other node is on the
# bus. The only trustworthy proof that a SPECIFIC ECU is alive is that it
# answers back: the receiver firmware replies "OK" to every command meant for
# it. So we wait a short time for that reply:
#     reply arrives in time  -> ECU is ONLINE,  command shown normally
#     no reply in time       -> ECU is OFFLINE, command shown as a RED error
#
# If you ever see false "NO REPLY" errors on a healthy bus (e.g. the serial
# link is very busy), raise this timeout a little.
ACK_TIMEOUT_S = 0.8

# ── MATCHED THEME COLORS FROM GUI_ECU ───────────────────────────────────────
COLORS = {
    "bg":          "#0d0f12",
    "panel":       "#161a1f",
    "panel_alt":   "#1c2128",
    "border":      "#2a3040",
    "text":        "#cdd6e0",
    "text_dim":    "#7a8899",
    "text_mono":   "#00e5cc",
    "tx":          "#00aaff",
    "rx":          "#00e676",
    "err":         "#ff1744",
    "err_text":    "#ff8a80",   # readable red on the dark-red error row
    "warn":        "#ffd600",   # "waiting for reply" rows
    "row_alt":     "#121519",
    "highlight":   "#00aaff",
    "err_row":     "#3a171c",
}

# ============================================================================
# PROTOCOL  (mirrors setup.py / functions.py on the boards)
# ============================================================================
# A frame = 1 target-address byte + short ASCII text, e.g. b'\x01BAT ON'.
# WHO sent it is the CAN id:  0x100 = MAIN, 0x101 = ECU1 ... 0x104 = ECU4.
# The first payload byte is the target index: MAIN=0, ECU1=1, ... ECU4=4.
# Keep this identical to the firmware if you ever renumber the devices.
NODE_IDS = {
    "MAIN": 0x100,
    "ECU1": 0x101,
    "ECU2": 0x102,
    "ECU3": 0x103,
    "ECU4": 0x104,
}
NODE_NAMES = {cid: name for name, cid in NODE_IDS.items()}   # id   -> name
ADDR_NAMES = {i: name for i, name in enumerate(NODE_IDS)}    # byte -> name
MAIN_ID    = NODE_IDS["MAIN"]

# Everything that is not MAIN is an ECU we can command and expect a reply from.
ECU_NAMES = {name for name in NODE_IDS if name != "MAIN"}


def format_timestamp(ts: float) -> str:
    """unix time -> 'HH:MM:SS.mmm' (or 'never' for 0)."""
    if ts == 0.0:
        return "never"
    millis = int((ts - int(ts)) * 1000)
    return f"{time.strftime('%H:%M:%S', time.localtime(ts))}.{millis:03d}"


# ============================================================================
# DATA MODEL
# ============================================================================
@dataclass
class CanFrame:
    timestamp: float
    can_id: int
    data: bytes
    direction: str = "TX"          # "TX" = MAIN sent it, "RX" = MAIN received it
    is_error: bool = False         # hard failure (firmware ERR flag, or NO REPLY)
    error_reason: str = ""

    # Delivery tracking. Only meaningful for a TX command addressed to an ECU.
    awaiting_ack: bool = False     # True while we wait for that ECU's reply
    ack_deadline: float = 0.0      # time.time() by which the reply must arrive
    acked: bool = False            # True once the ECU replied in time

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
    def text(self) -> str:
        """ASCII command text after the target byte."""
        if len(self.data) > 1:
            try:
                return bytes(self.data[1:]).decode(errors="replace").strip()
            except Exception:
                return ""
        return ""

    @property
    def name(self) -> str:
        """Route shown in the log, e.g. 'MAIN->ECU1'."""
        tgt = self.target
        if tgt and tgt != "?":
            return f"{self.source}\u2192{tgt}"
        return self.source

    def delivery_state(self) -> str:
        """'failed' | 'pending' | 'ok' | 'plain' -- drives row colour + summary."""
        if self.is_error:
            return "failed"
        if self.acked:
            return "ok"
        if self.awaiting_ack:
            return "pending"
        return "plain"

    def summary(self) -> str:
        if self.is_error:
            return self.error_reason or "communication error"
        arrow = "\u2192" if self.direction == "TX" else "\u2190"
        base = f"{arrow} {self.target} : {self.text or '\u2014'}"
        state = self.delivery_state()
        if state == "ok":
            return base + "  \u2713"        # delivered
        if state == "pending":
            return base + "  \u2026"        # waiting for reply
        return base


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

        # Connection / CAN health (written from the reader thread via plain
        # attribute writes; the Tk side reads them in _refresh_ui).
        self._connected  = False
        self._can_health = ("CAN \u2014", COLORS["text_dim"])

        # Commands still waiting for an ECU reply, and each node's live status.
        # Touched only on the Tk thread (_drain_queue / _refresh_ui / set_connected).
        self._pending = []
        self.node_status = {name: "unknown" for name in NODE_IDS}   # online/offline/unknown

        # Filter state (simple + friendly):
        #   show  : "all" | "tx" | "rx" | "error"
        #   nodes : set of node names to keep (empty = all)
        #   text  : substring to match
        self.filter_state = {"show": "all", "nodes": set(), "text": ""}

        self._setup_styles()
        self._build_layout()

        self.after(20, self._drain_queue)
        self.after(100, self._refresh_ui)

    # ------------------------------------------------------------------
    # PUBLIC API  (called by GUI_ECU)
    # ------------------------------------------------------------------
    def feed_line(self, line: str):
        """Parse one serial line from MAIN and queue any CAN frame.

        Runs on the serial reader thread, so it stays thread-safe: it only
        touches the thread-safe frame_queue and a couple of plain attributes,
        never a Tk widget. All correlation/colour logic happens later on the
        Tk thread.

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
        """Reflect the serial link state. Called on the Tk thread."""
        self._connected = bool(connected)
        color = COLORS["rx"] if connected else COLORS["err"]
        txt   = "BUS LIVE" if connected else "BUS OFFLINE"
        try:
            self._status_dot.itemconfig(self._status_oval, fill=color)
            self._status_lbl.config(text=txt, fg=color)
        except Exception:
            pass

        if connected:
            self.node_status["MAIN"] = "online"
        else:
            # Link down: we no longer know anything; drop pending waits.
            for name in self.node_status:
                self.node_status[name] = "unknown"
            self._pending = []

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
            if dev not in NODE_IDS:
                return None
            payload = bytes([ADDR_NAMES_INV(dev)]) + text.encode()
            return CanFrame(time.time(), MAIN_ID, payload, direction="TX",
                            is_error=not ok,
                            error_reason="" if ok else "TX failed (no ACK / bus error)")
        except Exception:
            return None

    # ------------------------------------------------------------------
    # STYLES + LAYOUT
    # ------------------------------------------------------------------
    def _setup_styles(self):
        self.font_mono = ("Courier New", 9)
        self.font_mono_bold = ("Courier New", 9, "bold")
        self.font_header = ("Courier New", 12, "bold")

        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Treeview", background=COLORS["panel"], foreground=COLORS["text"],
                        fieldbackground=COLORS["panel"], rowheight=24, font=self.font_mono, borderwidth=0)
        style.configure("Treeview.Heading", background=COLORS["border"], foreground=COLORS["text_mono"],
                        font=self.font_mono_bold, relief="flat", borderwidth=0, padding=(6, 4))
        style.map("Treeview.Heading", background=[("active", COLORS["panel_alt"])])
        style.map("Treeview", background=[("selected", COLORS["highlight"])],
                  foreground=[("selected", COLORS["bg"])])

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
        tk.Button(bar, text="CLEAR", command=self._clear_log, bg=COLORS["panel_alt"], fg=COLORS["text"],
                  font=self.font_mono_bold, relief="flat", bd=0, padx=12, pady=4, cursor="hand2").pack(side="right", padx=(0, 10), pady=6)

        self.pause_btn = tk.Button(bar, text="PAUSE", command=self._toggle_pause, bg=COLORS["panel_alt"], fg=COLORS["text"],
                                   font=self.font_mono_bold, relief="flat", bd=0, padx=12, pady=4, cursor="hand2")
        self.pause_btn.pack(side="right", padx=(0, 4), pady=6)

        self.filter_btn = tk.Button(bar, text="FILTER \u25be", command=self._open_filter_popup, bg=COLORS["panel_alt"], fg=COLORS["text"],
                                    font=self.font_mono_bold, relief="flat", bd=0, padx=12, pady=4, cursor="hand2")
        self.filter_btn.pack(side="right", padx=(0, 4), pady=6)

        self.filter_indicator = tk.Canvas(bar, width=10, height=10, bg=COLORS["panel"], highlightthickness=0)
        self.filter_indicator_oval = self.filter_indicator.create_oval(1, 1, 9, 9, fill=COLORS["panel"], outline="")
        self.filter_indicator.pack(side="right", padx=(0, 4))

        # --- SUPERVISION NODE (who is connected) ---
        sup_frame = tk.Frame(self, bg=COLORS["panel"], highlightbackground=COLORS["border"], highlightthickness=1)
        sup_frame.pack(fill="x", padx=18, pady=(0, 8))
        tk.Label(sup_frame, text="NODES", bg=COLORS["panel"], fg=COLORS["text"], font=self.font_header).pack(anchor="w", padx=14, pady=(6, 4))

        self.sup_tree = ttk.Treeview(sup_frame, columns=("name", "id", "status", "count", "last"), show="headings", height=5)
        for col, lbl, w, anch in [("name", "NODE", 110, "w"), ("id", "ID", 70, "w"),
                                  ("status", "STATUS", 130, "w"), ("count", "FRAMES", 80, "e"),
                                  ("last", "LAST SEEN", 140, "e")]:
            self.sup_tree.heading(col, text=lbl)
            self.sup_tree.column(col, width=w, anchor=anch, stretch=False)
        self.sup_tree.tag_configure("online",  foreground=COLORS["rx"])
        self.sup_tree.tag_configure("offline", foreground=COLORS["err"])
        self.sup_tree.tag_configure("unknown", foreground=COLORS["text_dim"])
        self.sup_tree.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        for name, cid in NODE_IDS.items():
            self.sup_tree.insert("", "end", iid=f"sup_{cid}",
                                 values=(name, f"0x{cid:03X}", "\u2014", "0", "never"), tags=("unknown",))

        # --- PANED BODY ---
        body = tk.PanedWindow(self, orient="horizontal", bg=COLORS["bg"], sashwidth=6, bd=0)
        body.pack(fill="both", expand=True, padx=18, pady=(0, 18))

        # --- MESSAGE LOG (LEFT) ---
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
        for col, lbl, w, stretch in [("time", "TIME", 120, False), ("id", "ID", 70, False),
                                     ("name", "ROUTE", 120, False), ("data", "DATA (HEX)", 200, False),
                                     ("decoded", "DECODED", 250, True)]:
            self.tree.heading(col, text=lbl)
            self.tree.column(col, width=w, stretch=stretch)

        self.tree.tag_configure("alt",     background=COLORS["row_alt"])
        self.tree.tag_configure("err",     background=COLORS["err_row"], foreground=COLORS["err_text"])
        self.tree.tag_configure("pending", foreground=COLORS["warn"])
        self.tree.tag_configure("ok",      foreground=COLORS["rx"])

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self.tree.bind("<Button-1>", lambda e: setattr(self, 'autoscroll', False))
        self.tree.bind("<<TreeviewSelect>>", self._on_frame_selected)

        # --- FRAME DETAIL (RIGHT) ---
        det_panel = tk.Frame(body, bg=COLORS["panel"], highlightbackground=COLORS["border"], highlightthickness=1)
        body.add(det_panel, minsize=260, stretch="never", width=340)
        tk.Label(det_panel, text="FRAME DETAIL", bg=COLORS["panel"], fg=COLORS["text"], font=self.font_header).pack(anchor="w", padx=14, pady=(6, 4))

        self.detail_text = tk.Text(det_panel, bg=COLORS["panel_alt"], fg=COLORS["text"], font=self.font_mono,
                                   relief="flat", padx=12, pady=10, wrap="word", bd=0,
                                   highlightthickness=1, highlightbackground=COLORS["border"])
        self.detail_text.pack(fill="both", expand=True, padx=14, pady=(0, 14))

    # ------------------------------------------------------------------
    # FRIENDLY FILTER POPUP
    # ------------------------------------------------------------------
    def _open_filter_popup(self):
        if hasattr(self, "_filter_popup") and self._filter_popup.winfo_exists():
            self._filter_popup.lift()
            self._filter_popup.focus_force()
            return

        popup = tk.Toplevel(self)
        self._filter_popup = popup
        popup.title("Filter messages")
        popup.configure(bg=COLORS["panel"])
        popup.transient(self.winfo_toplevel())
        popup.resizable(False, False)

        self.update_idletasks()
        bx = self.filter_btn.winfo_rootx()
        by = self.filter_btn.winfo_rooty() + self.filter_btn.winfo_height() + 4
        pw = 320
        x = max(20, bx + self.filter_btn.winfo_width() - pw)
        popup.geometry(f"{pw}x430+{x}+{by}")

        popup.bind("<Escape>", lambda _e: self._cancel_filter_popup())
        popup.bind("<Return>", lambda _e: self._apply_filter_popup())

        body = tk.Frame(popup, bg=COLORS["panel"])
        body.pack(fill="both", expand=True, padx=16, pady=14)

        self._pf_show      = tk.StringVar(value=self.filter_state["show"])
        self._pf_node_vars = {}
        self._pf_text_var  = tk.StringVar(value=self.filter_state["text"])

        # Show: one easy choice covering direction + errors.
        self._popup_section_header(body, "Show")
        for value, label in [("all",   "All messages"),
                             ("tx",    "Sent by MAIN  (TX)"),
                             ("rx",    "Received by MAIN  (RX)"),
                             ("error", "Errors only")]:
            self._popup_radio(body, label, self._pf_show, value)

        # Nodes: a single list (the CAN id and the source node are the same
        # thing, so there is no longer a duplicate "Message ID" + "Source" pair).
        self._popup_section_header(body, "Only from these nodes")
        for name in NODE_IDS:
            var = tk.BooleanVar(value=(name in self.filter_state["nodes"]))
            self._pf_node_vars[name] = var
            self._popup_check(body, name, var)

        # Free-text search.
        self._popup_section_header(body, "Search text")
        entry = tk.Entry(body, textvariable=self._pf_text_var, bg=COLORS["panel_alt"], fg=COLORS["text"],
                         insertbackground=COLORS["text"], relief="flat", font=self.font_mono)
        entry.pack(fill="x", padx=(20, 0), ipady=5, pady=(2, 2))
        tk.Label(body, text="Matches id, route, text, or error.", bg=COLORS["panel"],
                 fg=COLORS["text_dim"], font=("Courier New", 8), anchor="w").pack(fill="x", padx=(20, 0))

        # Buttons -- big, obvious APPLY FILTER on the right.
        btn_row = tk.Frame(popup, bg=COLORS["panel"])
        btn_row.pack(fill="x", padx=16, pady=(4, 16))
        tk.Button(btn_row, text="APPLY FILTER", command=self._apply_filter_popup,
                  bg=COLORS["tx"], fg=COLORS["bg"], font=self.font_mono_bold,
                  relief="flat", bd=0, padx=18, pady=9, cursor="hand2").pack(side="right")
        tk.Button(btn_row, text="Clear", command=self._clear_filter_popup,
                  bg=COLORS["panel_alt"], fg=COLORS["text"], font=self.font_mono,
                  relief="flat", bd=0, padx=14, pady=9, cursor="hand2").pack(side="right", padx=(0, 8))

        entry.focus_set()

    def _popup_section_header(self, parent, text):
        tk.Label(parent, text=text, bg=COLORS["panel"], fg=COLORS["text_mono"],
                 font=self.font_mono_bold, anchor="w").pack(fill="x", pady=(10, 2))

    def _popup_check(self, parent, label, var):
        tk.Checkbutton(parent, text=label, variable=var, bg=COLORS["panel"], fg=COLORS["text"],
                       selectcolor=COLORS["panel_alt"], activebackground=COLORS["panel"],
                       activeforeground=COLORS["text"], font=self.font_mono, anchor="w", bd=0).pack(fill="x", padx=(20, 0), pady=1)

    def _popup_radio(self, parent, label, var, value):
        tk.Radiobutton(parent, text=label, variable=var, value=value, bg=COLORS["panel"], fg=COLORS["text"],
                       selectcolor=COLORS["panel_alt"], activebackground=COLORS["panel"],
                       activeforeground=COLORS["text"], font=self.font_mono, anchor="w", bd=0).pack(fill="x", padx=(20, 0), pady=1)

    def _clear_filter_popup(self):
        """Reset the choices inside the popup (does not apply until APPLY)."""
        self._pf_show.set("all")
        for v in self._pf_node_vars.values():
            v.set(False)
        self._pf_text_var.set("")

    def _cancel_filter_popup(self):
        if hasattr(self, "_filter_popup") and self._filter_popup.winfo_exists():
            self._filter_popup.destroy()

    def _apply_filter_popup(self):
        nodes = {n for n, v in self._pf_node_vars.items() if v.get()}
        self.filter_state = {
            "show":  self._pf_show.get(),
            "nodes": nodes,
            "text":  self._pf_text_var.get().strip().lower(),
        }
        self._update_filter_indicator()
        self._rebuild_log_treeview()
        self._cancel_filter_popup()

    def _update_filter_indicator(self):
        active = self._filter_active()
        self.filter_indicator.itemconfig(self.filter_indicator_oval,
                                         fill=COLORS["tx"] if active else COLORS["panel"])
        self.filter_btn.config(text="FILTER \u25cf \u25be" if active else "FILTER \u25be",
                               fg=COLORS["tx"] if active else COLORS["text"])

    def _filter_active(self) -> bool:
        fs = self.filter_state
        return fs["show"] != "all" or bool(fs["nodes"]) or bool(fs["text"])

    def _matches_filter(self, f: CanFrame) -> bool:
        fs = self.filter_state
        if fs["show"] == "tx" and f.direction != "TX":
            return False
        if fs["show"] == "rx" and f.direction != "RX":
            return False
        if fs["show"] == "error" and not f.is_error:
            return False
        if fs["nodes"] and f.source not in fs["nodes"]:
            return False
        if fs["text"]:
            hay = f"0x{f.can_id:03X} {f.name} {f.source} {f.target} {f.text} {f.error_reason}".lower()
            if fs["text"] not in hay:
                return False
        return True

    # ------------------------------------------------------------------
    # CORE LOGIC
    # ------------------------------------------------------------------
    def _drain_queue(self):
        """Move queued frames into the log and run reply-correlation (Tk thread)."""
        try:
            for _ in range(80):
                frame = self.frame_queue.get_nowait()
                self.log.append(frame)
                self.total_messages += 1
                stats = self.msg_stats[frame.can_id]
                stats.count += 1
                stats.last_time = frame.timestamp
                self._correlate(frame)
        except queue.Empty:
            pass
        self.after(20, self._drain_queue)

    def _correlate(self, frame: CanFrame):
        """Match each command we send to an ECU with that ECU's reply.

        - A TX command addressed to an ECU starts waiting for a reply.
        - Any frame coming FROM an ECU is its reply: it proves the ECU is alive
          and clears whatever we were waiting on for that ECU.
        Frames flow through the queue in order (the firmware prints the TX
        trace before the reply arrives), so the TX is always seen first.
        """
        # Any traffic at all means MAIN is talking to us over serial.
        self.node_status["MAIN"] = "online"

        if (frame.direction == "TX" and not frame.is_error and frame.target in ECU_NAMES):
            frame.awaiting_ack = True
            frame.ack_deadline = time.time() + ACK_TIMEOUT_S
            self._pending.append(frame)

        if frame.direction == "RX" and frame.source in ECU_NAMES:
            self.node_status[frame.source] = "online"
            kept = []
            for pend in self._pending:
                if pend.target == frame.source:
                    pend.awaiting_ack = False
                    pend.acked = True
                else:
                    kept.append(pend)
            self._pending = kept

    def _expire_pending(self):
        """Commands with no reply in time -> RED error + that ECU is OFFLINE."""
        now = time.time()
        kept = []
        for pend in self._pending:
            if now >= pend.ack_deadline:
                pend.awaiting_ack = False
                pend.is_error = True
                pend.error_reason = (
                    "NO REPLY from {} \u2014 not connected (communication error)".format(pend.target))
                self.node_status[pend.target] = "offline"
            else:
                kept.append(pend)
        self._pending = kept

    def _refresh_ui(self):
        self._expire_pending()

        self.lbl_total.config(text=f"TOTAL: {self.total_messages:,}")
        txt, color = self._can_health
        self.can_health_lbl.config(text=txt, fg=color)

        self._update_supervision()

        if not self.paused:
            self._rebuild_log_treeview()

        self.after(100, self._refresh_ui)

    def _update_supervision(self):
        """Refresh the per-node table: STATUS column + green/red colouring."""
        status_txt = {"online": "\u25cf ONLINE", "offline": "\u25cf OFFLINE", "unknown": "\u2014"}
        for name, cid in NODE_IDS.items():
            iid = f"sup_{cid}"
            if not self.sup_tree.exists(iid):
                continue
            st = self.node_status.get(name, "unknown")
            stats = self.msg_stats.get(cid)
            count = stats.count if stats else 0
            last = format_timestamp(stats.last_time) if stats else "never"
            self.sup_tree.item(iid, values=(name, f"0x{cid:03X}", status_txt[st], str(count), last), tags=(st,))

    def _rebuild_log_treeview(self):
        active = self._filter_active()
        all_frames = list(self.log)
        visible = [f for f in all_frames if self._matches_filter(f)] if active else all_frames
        display_list = visible[-100:]

        self.tree.delete(*self.tree.get_children())

        for i, f in enumerate(display_list):
            clean_id = f"0x{f.can_id:03X}"
            hex_data = " ".join(f"{b:02X}" for b in f.data)

            state = f.delivery_state()
            if f.is_error:
                tag = "err"
            elif state == "pending":
                tag = "pending"
            elif state == "ok":
                tag = "ok"
            else:
                tag = "alt" if i % 2 == 0 else ""

            self.tree.insert("", "end",
                             values=(format_timestamp(f.timestamp), clean_id, f.name, hex_data, f.summary()),
                             tags=(tag,))

        if self.autoscroll:
            self.tree.yview_moveto(1)

        self.log_count_label.config(
            text=f"Showing {len(display_list)} of {len(all_frames):,} {'(filtered)' if active else ''}")

    def _on_frame_selected(self, event):
        selection = self.tree.selection()
        if not selection:
            return
        item = self.tree.item(selection[0])
        values = item['values']
        if not values:
            return
        try:
            target_id = int(values[1], 16)
        except (ValueError, IndexError):
            return

        matched = None
        for f in reversed(self.log):
            if f.can_id == target_id:
                matched = f
                break

        self.detail_text.delete("1.0", tk.END)
        if not matched:
            return

        state_label = {
            "ok":      "delivered (reply received)",
            "pending": "waiting for reply\u2026",
            "failed":  matched.error_reason or "no reply",
            "plain":   "\u2014",
        }[matched.delivery_state()]

        line_sep = "\u2500" * 30
        lines = [
            "FRAME DETAILS",
            line_sep,
            f"Timestamp : {format_timestamp(matched.timestamp)}",
            f"Arb ID    : 0x{matched.can_id:03X}",
            f"Route     : {matched.name}",
            f"Direction : {matched.direction}",
            f"Source    : {matched.source}",
            f"Target    : {matched.target}",
            f"Delivery  : {state_label}",
            f"Length    : {len(matched.data)} bytes",
            f"Raw Hex   : {' '.join(f'{b:02X}' for b in matched.data)}",
            "",
            "TEXT:",
            line_sep,
            f"  {matched.text or '\u2014'}",
        ]
        if matched.is_error:
            lines += ["", f"!! {matched.error_reason} !!"]

        self.detail_text.insert("1.0", "\n".join(lines))

    # ------------------------------------------------------------------
    # TOOLBAR ACTIONS
    # ------------------------------------------------------------------
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
        self._pending = []
        # Forget per-ECU verdicts; keep MAIN as-is (driven by the serial link).
        for name in self.node_status:
            if name != "MAIN":
                self.node_status[name] = "unknown"
        self.autoscroll = True
        self.detail_text.delete("1.0", tk.END)
        self._update_supervision()


# Tiny helper used by the legacy parser: node name -> target byte.
def ADDR_NAMES_INV(name: str) -> int:
    for byte, n in ADDR_NAMES.items():
        if n == name:
            return byte
    return 0