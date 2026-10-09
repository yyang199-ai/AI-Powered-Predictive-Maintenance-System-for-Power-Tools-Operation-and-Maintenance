#!/usr/bin/env bash
set -euo pipefail
dashboard_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$dashboard_root"
if [ ! -x .venv/bin/python ]; then
  echo "请先执行 bash scripts/setup_codespaces.sh。" >&2
  exit 1
fi
.venv/bin/python - "${1:-8501}" <<'PY'
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

root = Path.cwd()
try:
    port = int(sys.argv[1])
    if not 1024 <= port <= 65535:
        raise ValueError
except ValueError:
    sys.exit("端口必须是1024至65535之间的整数。")
output = root / "artifacts"
output.mkdir(exist_ok=True)
pidfile = output / f"codespaces_streamlit_{port}.pid"
logfile = output / f"codespaces_streamlit_{port}.log"
health = f"http://127.0.0.1:{port}/_stcore/health"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def ready():
    try:
        with opener.open(health, timeout=1) as response:
            return response.status == 200 and response.read().strip() == b"ok"
    except OSError:
        return False

if pidfile.exists():
    try:
        pid = int(pidfile.read_text().strip())
        args = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
        if (args[1:5] == ["-m", "streamlit", "run", str(root / "app.py")]
                and "--server.port" in args and args[args.index("--server.port") + 1] == str(port)
                and ready()):
            print(f"实验平台已经运行。请在 Ports 面板打开 {port}。")
            sys.exit(0)
    except (OSError, ValueError, IndexError):
        pass
    pidfile.unlink(missing_ok=True)

try:
    with socket.socket() as probe:
        probe.bind(("0.0.0.0", port))
except OSError:
    sys.exit(f"端口 {port} 已被其他进程占用；请换端口或在原终端停止旧服务。")

with logfile.open("ab") as log:
    process = subprocess.Popen(
        [str(root / ".venv/bin/python"), "-m", "streamlit", "run", str(root / "app.py"),
         "--server.address", "0.0.0.0", "--server.port", str(port),
         "--server.headless", "true", "--browser.gatherUsageStats", "false"],
        cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
pidfile.write_text(str(process.pid))
deadline = time.monotonic() + 40
while time.monotonic() < deadline:
    if process.poll() is not None:
        pidfile.unlink(missing_ok=True)
        sys.exit(f"启动失败，请查看 {logfile.relative_to(root)}。")
    if ready():
        print(f"实验平台已启动。请在 Codespaces 的 Ports 面板打开端口 {port}。")
        print(f"运行日志：{logfile.relative_to(root)}")
        sys.exit(0)
    time.sleep(.25)
process.terminate()
pidfile.unlink(missing_ok=True)
sys.exit(f"启动超时，请查看 {logfile.relative_to(root)}。")
PY
