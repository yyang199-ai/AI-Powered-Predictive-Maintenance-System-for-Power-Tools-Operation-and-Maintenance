#!/usr/bin/env bash
set -euo pipefail
dashboard_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$dashboard_root"
python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'
python -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check -r requirements.lock.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/verify_phase2.py
echo "依赖、公开数据和模型已就绪。无需重新训练。"
