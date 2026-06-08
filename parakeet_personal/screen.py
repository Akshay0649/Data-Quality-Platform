"""
Screenshot + OCR.

Tries three capture methods in order (Pillow ImageGrab → pyautogui → scrot)
so it works on macOS, Windows, and most Linux desktops.
"""

import tempfile
from pathlib import Path


def _grab() -> "Image":  # noqa: F821
    """Return a PIL Image of the full primary screen."""
    # 1. Pillow ImageGrab (works on macOS + Windows; Linux needs Xorg + display)
    try:
        from PIL import ImageGrab
        img = ImageGrab.grab()
        if img:
            return img
    except Exception:
        pass

    # 2. pyautogui (cross-platform, works on Wayland via mss back-end)
    try:
        import pyautogui
        return pyautogui.screenshot()
    except Exception:
        pass

    # 3. scrot (Linux CLI fallback)
    try:
        import subprocess
        from PIL import Image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = f.name
        subprocess.run(["scrot", "-z", tmp], check=True, capture_output=True)
        img = Image.open(tmp)
        Path(tmp).unlink(missing_ok=True)
        return img
    except Exception:
        pass

    raise RuntimeError(
        "Screen capture unavailable. Install pyautogui or scrot, or use X11 display."
    )


def capture_screen_text() -> str:
    """Capture full screen and return OCR'd text."""
    try:
        import pytesseract
    except ImportError:
        raise RuntimeError("pytesseract not installed: pip install pytesseract")

    img = _grab()
    text = pytesseract.image_to_string(img)
    return text.strip()
