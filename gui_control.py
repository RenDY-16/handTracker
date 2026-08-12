"""
Retrolens Control Panel GUI - Dark Sci-Fi Management Window
Built with Tkinter for seamless real-time control of Retrolens Handtrack App.
"""
import tkinter as tk
from tkinter import ttk
import threading
import time

try:
    from pygrabber.dshow_graph import FilterGraph
    HAS_PYGRABBER = True
except ImportError:
    HAS_PYGRABBER = False

class ControlPanelGUI:
    def __init__(self, app_context=None):
        self.app_context = app_context or {}
        self.root = None
        self.is_running = False

    def start_in_thread(self):
        """Starts the Tkinter main loop in a dedicated daemon thread."""
        thread = threading.Thread(target=self._run_gui, daemon=True)
        thread.start()

    def _run_gui(self):
        self.root = tk.Tk()
        self.root.title("🕷️ Retrolens Control Panel")
        self.root.geometry("460x680")
        self.root.configure(bg="#0d0b18")
        self.root.resizable(False, False)

        # Style Configuration
        style = ttk.Style()
        style.theme_use("clam")

        # Custom Colors
        BG_DARK = "#0d0b18"
        CARD_BG = "#161327"
        TEXT_MAIN = "#f0f0f8"
        TEXT_MUTED = "#8e8a9f"
        ACCENT_CYAN = "#00e5ff"
        ACCENT_RED = "#ff0055"

        style.configure(".", background=BG_DARK, foreground=TEXT_MAIN, font=("Segoe UI", 9))
        style.configure("TLabel", background=BG_DARK, foreground=TEXT_MAIN)
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"), foreground=ACCENT_CYAN)
        style.configure("SubHeader.TLabel", font=("Segoe UI", 10, "bold"), foreground=TEXT_MUTED)
        style.configure("Card.TFrame", background=CARD_BG, relief="flat")
        style.configure("TButton", font=("Segoe UI", 9, "bold"), background="#221c3d", foreground=TEXT_MAIN, borderwidth=0)
        style.map("TButton", background=[("active", "#312a57")])

        style.configure("ActionRed.TButton", font=("Segoe UI", 9, "bold"), background="#800020", foreground="#ffffff")
        style.map("ActionRed.TButton", background=[("active", "#b3002d")])
        
        style.configure("ActionCyan.TButton", font=("Segoe UI", 9, "bold"), background="#006680", foreground="#ffffff")
        style.map("ActionCyan.TButton", background=[("active", "#0099bf")])

        # Scrollable Main Frame
        main_container = tk.Frame(self.root, bg=BG_DARK)
        main_container.pack(fill="both", expand=True, padx=12, pady=12)

        # TITLE BANNER
        banner_frame = tk.Frame(main_container, bg=CARD_BG, padx=10, pady=8)
        banner_frame.pack(fill="x", pady=(0, 10))

        title_lbl = tk.Label(banner_frame, text="🕷️ RETROLENS CONTROL PANEL", font=("Segoe UI", 13, "bold"), fg=ACCENT_CYAN, bg=CARD_BG)
        title_lbl.pack(anchor="w")
        sub_lbl = tk.Label(banner_frame, text="Venom Symbiote Edition • Live Visual & Camera Manager", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=CARD_BG)
        sub_lbl.pack(anchor="w")

        # 1. CAMERA SELECTION CARD
        cam_card = tk.LabelFrame(main_container, text=" 📹 CAMERA SELECTION ", font=("Segoe UI", 9, "bold"), fg=ACCENT_CYAN, bg=CARD_BG, bd=1, relief="solid")
        cam_card.pack(fill="x", pady=(0, 8), padx=2, ipadx=6, ipady=6)

        cams = self._get_camera_list()
        self.cam_var = tk.StringVar(value=cams[0] if cams else "Default Camera")
        cam_dropdown = ttk.Combobox(cam_card, textvariable=self.cam_var, values=cams, state="readonly", font=("Segoe UI", 9))
        cam_dropdown.pack(fill="x", padx=8, pady=4)
        cam_dropdown.bind("<<ComboboxSelected>>", self._on_camera_selected)

        # 2. SHADER FILTERS CARD
        filter_card = tk.LabelFrame(main_container, text=" 🎨 VISUAL SHADER FILTERS (24) ", font=("Segoe UI", 9, "bold"), fg=ACCENT_CYAN, bg=CARD_BG, bd=1, relief="solid")
        filter_card.pack(fill="x", pady=(0, 8), padx=2, ipadx=6, ipady=6)

        filters_list = self.app_context.get("filters", [
            "VENOM-VISION", "SPIDER-MAN", "VENOM-CARNAGE", "SPIDER-2099", "SYMBIOTE-VORTEX",
            "MULTIVERSE-GLITCH", "SYMBIOTE-RED", "SPIDER-NOIR", "TOXIC-SYMBIOTE", "ANTI-VENOM",
            "GLITCH", "NEON", "MONO", "PIXELATE", "INVERT"
        ])
        
        self.filter_var = tk.StringVar(value=filters_list[0])
        filter_dropdown = ttk.Combobox(filter_card, textvariable=self.filter_var, values=filters_list, state="readonly", font=("Segoe UI", 9))
        filter_dropdown.pack(fill="x", padx=8, pady=4)
        filter_dropdown.bind("<<ComboboxSelected>>", self._on_filter_selected)

        # Quick Filter Preset Buttons (Row 1 & 2)
        btn_frame1 = tk.Frame(filter_card, bg=CARD_BG)
        btn_frame1.pack(fill="x", padx=8, pady=2)
        
        ttk.Button(btn_frame1, text="🕷️ Venom", command=lambda: self.set_filter("VENOM-VISION")).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Button(btn_frame1, text="🕸️ Spidey", command=lambda: self.set_filter("SPIDER-MAN")).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Button(btn_frame1, text="🩸 Carnage", command=lambda: self.set_filter("VENOM-CARNAGE")).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Button(btn_frame1, text="⚡ 2099", command=lambda: self.set_filter("SPIDER-2099")).pack(side="left", expand=True, fill="x", padx=1)

        btn_frame2 = tk.Frame(filter_card, bg=CARD_BG)
        btn_frame2.pack(fill="x", padx=8, pady=2)
        ttk.Button(btn_frame2, text="💥 Glitch", command=lambda: self.set_filter("MULTIVERSE-GLITCH")).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Button(btn_frame2, text="🔭 Night", command=lambda: self.set_filter("NIGHT-VISION")).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Button(btn_frame2, text="🔣 Hologram", command=lambda: self.set_filter("HOLOGRAM")).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Button(btn_frame2, text="🧪 Anti-Venom", command=lambda: self.set_filter("ANTI-VENOM")).pack(side="left", expand=True, fill="x", padx=1)

        # 3. OVERLAYS & FEATURE TOGGLES CARD
        toggle_card = tk.LabelFrame(main_container, text=" ⚙️ OVERLAYS & TOGGLES ", font=("Segoe UI", 9, "bold"), fg=ACCENT_CYAN, bg=CARD_BG, bd=1, relief="solid")
        toggle_card.pack(fill="x", pady=(0, 8), padx=2, ipadx=6, ipady=6)

        self.telemetry_var = tk.BooleanVar(value=self.app_context.get("show_telemetry_hud", False))
        self.help_var = tk.BooleanVar(value=self.app_context.get("show_help_overlay", False))
        self.hud_var = tk.BooleanVar(value=self.app_context.get("show_hud_buttons", True))
        self.glow_var = tk.BooleanVar(value=self.app_context.get("sci_fi_glow", True))

        chk1 = tk.Checkbutton(toggle_card, text="📊 AI Telemetry Diagnostics (Key D)", variable=self.telemetry_var, command=self._on_toggles_changed, bg=CARD_BG, fg=TEXT_MAIN, selectcolor=BG_DARK, activebackground=CARD_BG, activeforeground=ACCENT_CYAN)
        chk1.pack(anchor="w", padx=8, pady=2)

        chk2 = tk.Checkbutton(toggle_card, text="📖 Interactive Help Guide (Key H)", variable=self.help_var, command=self._on_toggles_changed, bg=CARD_BG, fg=TEXT_MAIN, selectcolor=BG_DARK, activebackground=CARD_BG, activeforeground=ACCENT_CYAN)
        chk2.pack(anchor="w", padx=8, pady=2)

        chk3 = tk.Checkbutton(toggle_card, text="🔘 Touch HUD Menu Buttons (Key M)", variable=self.hud_var, command=self._on_toggles_changed, bg=CARD_BG, fg=TEXT_MAIN, selectcolor=BG_DARK, activebackground=CARD_BG, activeforeground=ACCENT_CYAN)
        chk3.pack(anchor="w", padx=8, pady=2)

        chk4 = tk.Checkbutton(toggle_card, text="✨ Sci-Fi Glow Particles & FX", variable=self.glow_var, command=self._on_toggles_changed, bg=CARD_BG, fg=TEXT_MAIN, selectcolor=BG_DARK, activebackground=CARD_BG, activeforeground=ACCENT_CYAN)
        chk4.pack(anchor="w", padx=8, pady=2)

        # 4. GESTURE CONFIDENCE SLIDER
        sens_card = tk.LabelFrame(main_container, text=" 🎚️ GESTURE SENSITIVITY ", font=("Segoe UI", 9, "bold"), fg=ACCENT_CYAN, bg=CARD_BG, bd=1, relief="solid")
        sens_card.pack(fill="x", pady=(0, 8), padx=2, ipadx=6, ipady=6)

        self.conf_lbl = tk.Label(sens_card, text="Hand Detection Confidence: 70%", bg=CARD_BG, fg=TEXT_MAIN)
        self.conf_lbl.pack(anchor="w", padx=8, pady=(2, 0))

        self.conf_slider = ttk.Scale(sens_card, from_=30, to=95, value=70, command=self._on_sens_changed)
        self.conf_slider.pack(fill="x", padx=8, pady=4)

        # 5. QUICK ACTIONS CARD
        action_card = tk.LabelFrame(main_container, text=" ⚡ QUICK ACTIONS ", font=("Segoe UI", 9, "bold"), fg=ACCENT_CYAN, bg=CARD_BG, bd=1, relief="solid")
        action_card.pack(fill="x", pady=(0, 4), padx=2, ipadx=6, ipady=6)

        act_frame = tk.Frame(action_card, bg=CARD_BG)
        act_frame.pack(fill="x", padx=8, pady=4)

        ttk.Button(act_frame, text="📸 Air-Snap Photo", style="ActionCyan.TButton", command=self.trigger_snap).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(act_frame, text="🎬 Record 5s Clip", style="ActionRed.TButton", command=self.trigger_rec).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(act_frame, text="🔍 Reset Zoom", command=self.trigger_reset_zoom).pack(side="left", expand=True, fill="x", padx=2)

        # Status Bar
        self.status_lbl = tk.Label(main_container, text="Status: Connected to Retrolens Engine", font=("Segoe UI", 8), fg="#00e5ff", bg=BG_DARK)
        self.status_lbl.pack(anchor="w", pady=(6, 0))

        # Periodic UI sync timer
        self.root.after(200, self._periodic_sync)
        self.is_running = True
        self.root.mainloop()
        self.is_running = False

    def _get_camera_list(self):
        if HAS_PYGRABBER:
            try:
                graph = FilterGraph()
                devices = graph.get_input_devices()
                if devices:
                    return [f"[{i}] {name}" for i, name in enumerate(devices)]
            except Exception:
                pass
        return ["Camera 0", "Camera 1", "Camera 2"]

    def set_filter(self, filter_name):
        self.filter_var.set(filter_name)
        if "set_filter_callback" in self.app_context:
            self.app_context["set_filter_callback"](filter_name)
        self.set_status(f"Filter changed to: {filter_name}")

    def _on_filter_selected(self, event=None):
        f_name = self.filter_var.get()
        if "set_filter_callback" in self.app_context:
            self.app_context["set_filter_callback"](f_name)
        self.set_status(f"Filter: {f_name}")

    def _on_camera_selected(self, event=None):
        cam_str = self.cam_var.get()
        if "set_camera_callback" in self.app_context:
            self.app_context["set_camera_callback"](cam_str)
        self.set_status(f"Camera selected: {cam_str}")

    def _on_toggles_changed(self):
        if "sync_toggles_callback" in self.app_context:
            self.app_context["sync_toggles_callback"]({
                "show_telemetry_hud": self.telemetry_var.get(),
                "show_help_overlay": self.help_var.get(),
                "show_hud_buttons": self.hud_var.get(),
                "sci_fi_glow": self.glow_var.get(),
            })

    def _on_sens_changed(self, val):
        conf = int(float(val))
        self.conf_lbl.config(text=f"Hand Detection Confidence: {conf}%")
        if "set_confidence_callback" in self.app_context:
            self.app_context["set_confidence_callback"](conf / 100.0)

    def trigger_snap(self):
        if "trigger_snap_callback" in self.app_context:
            self.app_context["trigger_snap_callback"]()
        self.set_status("📸 Air-Snap Triggered!")

    def trigger_rec(self):
        if "trigger_rec_callback" in self.app_context:
            self.app_context["trigger_rec_callback"]()
        self.set_status("🎬 5s Recording Started!")

    def trigger_reset_zoom(self):
        if "trigger_reset_zoom_callback" in self.app_context:
            self.app_context["trigger_reset_zoom_callback"]()
        self.set_status("🔍 Zoom Reset to 1.0x")

    def set_status(self, text):
        if hasattr(self, 'status_lbl') and self.status_lbl:
            self.status_lbl.config(text=f"Status: {text}")

    def _periodic_sync(self):
        """Syncs active state from main.py back to GUI if changed externally."""
        if not self.is_running or not self.root:
            return
        
        try:
            if "get_current_filter" in self.app_context:
                curr_f = self.app_context["get_current_filter"]()
                if curr_f and curr_f != self.filter_var.get():
                    self.filter_var.set(curr_f)
            
            if "get_toggles" in self.app_context:
                toggles = self.app_context["get_toggles"]()
                if toggles:
                    self.telemetry_var.set(toggles.get("show_telemetry_hud", False))
                    self.help_var.set(toggles.get("show_help_overlay", False))
                    self.hud_var.set(toggles.get("show_hud_buttons", True))
                    self.glow_var.set(toggles.get("sci_fi_glow", True))
        except Exception:
            pass

        if self.root:
            self.root.after(400, self._periodic_sync)
