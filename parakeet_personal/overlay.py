"""
Tkinter always-on-top overlay window.

Screen-share invisibility (best-effort per platform):
  Windows  – SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)
  macOS    – NSWindowSharingNone via pyobjc (if installed)
  Linux    – _NET_WM_WINDOW_TYPE = TOOLTIP via xprop (compositor-dependent)

The token queue pattern (50 ms poll) keeps UI updates smooth without
flooding Tk's event queue with one after() per streamed token.
"""

import platform
import queue
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, scrolledtext, simpledialog

# ── Colour palette ────────────────────────────────────────────────────────────
BG = "#111318"
BG2 = "#0a0c10"
FG = "#e2e4ea"
ACCENT = "#3ecf8e"
MUTED = "#5a5f70"
ERR = "#f87171"
BTN_BG = "#1e2130"
BTN_HOVER = "#2a2f45"

_SENTINEL = object()  # marks end-of-stream in token queue


class SettingsDialog(tk.Toplevel):
    def __init__(self, parent, config, on_save):
        super().__init__(parent)
        self.title("Settings — Parakeet Personal")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.grab_set()

        self._config = config
        self._on_save = on_save
        self._vars: dict[str, tk.Variable] = {}

        fields = [
            ("AI Provider", "ai_provider", "choice", ["claude", "openai"]),
            ("Claude model", "claude_model", "str", None),
            ("OpenAI model", "openai_model", "str", None),
            ("Anthropic API key", "anthropic_api_key", "secret", None),
            ("OpenAI API key", "openai_api_key", "secret", None),
            ("Whisper mode", "whisper_mode", "choice", ["api", "local"]),
            ("Local Whisper model", "whisper_local_model", "choice",
             ["tiny", "base", "small", "medium"]),
            ("Overlay opacity", "overlay_opacity", "float", None),
            ("Extra system prompt", "system_prompt_extra", "str", None),
        ]

        pad = dict(padx=14, pady=4)
        for i, (label, key, kind, opts) in enumerate(fields):
            tk.Label(self, text=label, bg=BG, fg=MUTED,
                     font=("Helvetica", 10)).grid(row=i, column=0, sticky="e", **pad)

            current = str(getattr(config, key))
            if kind == "choice":
                var = tk.StringVar(value=current)
                w = tk.OptionMenu(self, var, *opts)
                w.config(bg=BTN_BG, fg=FG, activebackground=BTN_HOVER,
                         highlightthickness=0, relief="flat")
                w["menu"].config(bg=BTN_BG, fg=FG)
            elif kind == "secret":
                var = tk.StringVar(value=current)
                w = tk.Entry(self, textvariable=var, show="•", width=40,
                             bg=BG2, fg=FG, insertbackground=FG,
                             relief="flat", bd=6)
            elif kind == "float":
                var = tk.StringVar(value=current)
                w = tk.Entry(self, textvariable=var, width=10,
                             bg=BG2, fg=FG, insertbackground=FG,
                             relief="flat", bd=6)
            else:
                var = tk.StringVar(value=current)
                w = tk.Entry(self, textvariable=var, width=40,
                             bg=BG2, fg=FG, insertbackground=FG,
                             relief="flat", bd=6)

            w.grid(row=i, column=1, sticky="w", **pad)
            self._vars[key] = var

        row = len(fields)
        tk.Button(self, text="Save", bg=ACCENT, fg="#000",
                  font=("Helvetica", 11, "bold"), relief="flat",
                  padx=16, pady=6, cursor="hand2",
                  command=self._save).grid(row=row, column=0, columnspan=2, pady=14)

    def _save(self):
        cfg = self._config
        for key, var in self._vars.items():
            raw = var.get()
            cur = getattr(cfg, key)
            if isinstance(cur, float):
                try:
                    raw = float(raw)
                except ValueError:
                    pass
            setattr(cfg, key, raw)
        cfg.save()
        self._on_save()
        self.destroy()


class ParakeetOverlay:
    def __init__(
        self,
        config,
        on_ask,
        on_screen_capture,
        on_audio_toggle,
        on_provider_change,
        on_resume_load,
        on_settings_save,
    ):
        self.config = config
        self.on_ask = on_ask
        self.on_screen_capture = on_screen_capture
        self.on_audio_toggle = on_audio_toggle
        self.on_provider_change = on_provider_change
        self.on_resume_load = on_resume_load
        self.on_settings_save = on_settings_save

        self._listening = False
        self._token_q: queue.Queue = queue.Queue()
        self._root: tk.Tk | None = None
        self._response: scrolledtext.ScrolledText | None = None
        self._status_var: tk.StringVar | None = None
        self._provider_var: tk.StringVar | None = None
        self._listen_btn: tk.Button | None = None
        self._input: tk.Text | None = None

    # ── Public thread-safe API ────────────────────────────────────────────────

    def set_status(self, text: str):
        if self._root:
            self._root.after(0, lambda: self._status_var.set(text))

    def set_question(self, text: str):
        def _set():
            self._input.delete("1.0", "end")
            self._input.insert("1.0", text)
        if self._root:
            self._root.after(0, _set)

    def start_answer(self, question: str):
        separator = f"\n{'─' * 44}\n"
        for ch in (separator, f"Q: {question}\n\n"):
            self._token_q.put(ch)

    def stream_token(self, token: str):
        self._token_q.put(token)

    def end_answer(self):
        self._token_q.put(_SENTINEL)

    # ── Build & run ──────────────────────────────────────────────────────────

    def run(self):
        root = tk.Tk()
        self._root = root
        root.title("Parakeet Personal")
        root.configure(bg=BG)
        root.attributes("-topmost", True)
        root.attributes("-alpha", self.config.overlay_opacity)
        root.geometry("500x620+60+60")
        root.resizable(True, True)

        self._apply_screen_share_invisibility(root)
        self._build_ui(root)
        root.after(50, self._poll_tokens)
        root.mainloop()

    # ── Screen-share invisibility ─────────────────────────────────────────────

    def _apply_screen_share_invisibility(self, root: tk.Tk):
        system = platform.system()
        root.update_idletasks()

        if system == "Windows":
            try:
                import ctypes
                hwnd = ctypes.windll.user32.FindWindowW(None, "Parakeet Personal")
                if hwnd:
                    ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0x00000011)
            except Exception:
                pass

        elif system == "Darwin":
            try:
                from AppKit import NSApplication, NSWindowSharingNone
                for win in NSApplication.sharedApplication().windows():
                    win.setSharingType_(NSWindowSharingNone)
            except Exception:
                pass  # pyobjc not installed – silently skip

        elif system == "Linux":
            try:
                import subprocess
                wid = hex(root.winfo_id())
                subprocess.run(
                    ["xprop", "-id", wid, "-f", "_NET_WM_WINDOW_TYPE", "32a",
                     "-set", "_NET_WM_WINDOW_TYPE", "_NET_WM_WINDOW_TYPE_TOOLTIP"],
                    capture_output=True, timeout=2,
                )
            except Exception:
                pass

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self, root: tk.Tk):
        # Header bar
        header = tk.Frame(root, bg=BG2, pady=7)
        header.pack(fill="x")

        tk.Label(
            header, text="🦜  Parakeet Personal",
            bg=BG2, fg=ACCENT, font=("Helvetica", 13, "bold"),
        ).pack(side="left", padx=12)

        # Provider radio buttons
        self._provider_var = tk.StringVar(value=self.config.ai_provider)
        for p in ("claude", "openai"):
            tk.Radiobutton(
                header, text=p.capitalize(), variable=self._provider_var, value=p,
                bg=BG2, fg=FG, selectcolor=BG2, activebackground=BG2,
                command=self._on_provider_change,
            ).pack(side="right", padx=4)
        tk.Label(header, text="Model:", bg=BG2, fg=MUTED,
                 font=("Helvetica", 9)).pack(side="right", padx=(8, 0))

        # Status bar
        self._status_var = tk.StringVar(value="Ready")
        tk.Label(
            root, textvariable=self._status_var,
            bg=BG2, fg=MUTED, font=("Helvetica", 9), anchor="w", padx=10,
        ).pack(fill="x")

        # Response pane
        resp_frame = tk.Frame(root, bg=BG, pady=4)
        resp_frame.pack(fill="both", expand=True, padx=8)
        tk.Label(resp_frame, text="Answer", bg=BG, fg=MUTED,
                 font=("Helvetica", 9)).pack(anchor="w")
        mono = tkfont.Font(family="Courier", size=10)
        self._response = scrolledtext.ScrolledText(
            resp_frame, bg=BG2, fg=FG, insertbackground=FG,
            font=mono, wrap="word", relief="flat", bd=6,
            selectbackground="#1e3a5f",
        )
        self._response.pack(fill="both", expand=True)
        self._response.config(state="disabled")

        # Input pane
        inp_frame = tk.Frame(root, bg=BG, pady=2)
        inp_frame.pack(fill="x", padx=8, pady=(0, 2))
        tk.Label(inp_frame, text="Question  (Enter to ask · Shift+Enter for newline)",
                 bg=BG, fg=MUTED, font=("Helvetica", 9)).pack(anchor="w")
        self._input = tk.Text(
            inp_frame, bg=BG2, fg=FG, insertbackground=FG,
            height=3, font=("Helvetica", 11), wrap="word", relief="flat", bd=6,
        )
        self._input.pack(fill="x")
        self._input.bind("<Return>", self._on_enter)
        self._input.bind("<Shift-Return>", lambda _: None)  # allow literal newline

        # Button row
        btn_frame = tk.Frame(root, bg=BG, pady=4)
        btn_frame.pack(fill="x", padx=8, pady=(0, 10))

        def btn(parent, text, cmd, fg=FG, bg=BTN_BG, bold=False):
            font = ("Helvetica", 10, "bold") if bold else ("Helvetica", 10)
            b = tk.Button(parent, text=text, command=cmd,
                          bg=bg, fg=fg, activebackground=BTN_HOVER,
                          font=font, relief="flat", padx=10, pady=5,
                          cursor="hand2", bd=0)
            b.pack(side="left", padx=(0, 4))
            return b

        btn(btn_frame, "Ask ↵", self._on_ask, fg="#000", bg=ACCENT, bold=True)
        self._listen_btn = btn(btn_frame, "🎙 Listen", self._on_listen_toggle)
        btn(btn_frame, "📸 Screen", self._on_screen)
        btn(btn_frame, "📄 Resume", self._on_resume)
        btn(btn_frame, "⚙ Settings", self._on_settings)
        btn(btn_frame, "✕ Clear", self._clear, fg=ERR)

    # ── Button handlers ───────────────────────────────────────────────────────

    def _on_enter(self, event):
        if not (event.state & 1):  # Shift not held
            self._on_ask()
            return "break"

    def _on_ask(self):
        q = self._input.get("1.0", "end").strip()
        if q:
            self._input.delete("1.0", "end")
            self.on_ask(q)

    def _on_listen_toggle(self):
        self._listening = not self._listening
        if self._listening:
            self._listen_btn.config(text="⏹ Stop listening", bg="#7f1d1d", fg=FG)
        else:
            self._listen_btn.config(text="🎙 Listen", bg=BTN_BG, fg=FG)
        self.on_audio_toggle(self._listening)

    def _on_screen(self):
        # Briefly hide so the overlay isn't in its own screenshot
        self._root.withdraw()
        self._root.after(350, self._do_screen)

    def _do_screen(self):
        self.on_screen_capture()
        self._root.deiconify()

    def _on_resume(self):
        path = filedialog.askopenfilename(
            title="Select Resume",
            filetypes=[("PDF", "*.pdf"), ("Text", "*.txt"), ("All", "*.*")],
        )
        if path:
            self.on_resume_load(path)

    def _on_settings(self):
        SettingsDialog(self._root, self.config, self.on_settings_save)

    def _on_provider_change(self):
        self.on_provider_change(self._provider_var.get())

    def _clear(self):
        self._response.config(state="normal")
        self._response.delete("1.0", "end")
        self._response.config(state="disabled")

    # ── Token streaming ───────────────────────────────────────────────────────

    def _poll_tokens(self):
        """Drain the token queue and write to response widget — runs on main thread."""
        self._response.config(state="normal")
        try:
            while True:
                item = self._token_q.get_nowait()
                if item is _SENTINEL:
                    self._response.insert("end", "\n")
                    self.set_status("Done")
                else:
                    self._response.insert("end", item)
        except queue.Empty:
            pass
        self._response.see("end")
        self._response.config(state="disabled")
        self._root.after(50, self._poll_tokens)  # reschedule
