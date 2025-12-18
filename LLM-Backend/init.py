import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KOBOLD_PATH = HERE / "koboldcpp.exe"
CFG_PATH = HERE / "Settings.kcpps"

cmd = [
    str(KOBOLD_PATH),
    "--config",
    str(CFG_PATH),
    "--skiplauncher",          # skip the launcher GUI and start server directly
    "--port", "8000",          # must match main.py (OpenAI base_url)
]

if not KOBOLD_PATH.exists():
    raise FileNotFoundError(f"Missing KoboldCpp executable: {KOBOLD_PATH}")
if not CFG_PATH.exists():
    raise FileNotFoundError(f"Missing KoboldCpp config: {CFG_PATH}")

p = subprocess.Popen(cmd)

time.sleep(2)
if p.poll() is not None:
    raise RuntimeError(f"KoboldCpp exited early with code {p.returncode}. Check CLI args/config.")

time.sleep(18)  # wait for load; or poll API
print("KoboldCpp PID:", p.pid)