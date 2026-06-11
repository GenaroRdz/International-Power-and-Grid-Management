import json
import os

# Store the settings file next to this script so it works regardless of the
# directory the app is launched from.
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "settings.json")

DEFAULTS = {
    "port": "COM7",
    "baudrate": 115200,
    
}


def load_settings():
    """Return saved settings merged over the defaults. Never raises."""
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("settings file is not a dict")
        return {**DEFAULTS, **data}
    except (FileNotFoundError, json.JSONDecodeError, ValueError, OSError):
        return dict(DEFAULTS)


def save_settings(settings):
    """Persist settings to disk. Returns True on success."""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
        return True
    except OSError as e:
        print(f"Could not save settings: {e}")
        return False