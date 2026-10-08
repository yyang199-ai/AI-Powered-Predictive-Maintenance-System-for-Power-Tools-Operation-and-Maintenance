#!/usr/bin/env bash
set -euo pipefail
cd /workspace/AI-Powered-Predictive-Maintenance-System-for-Power-Tools-Operation-and-Maintenance
python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'
python -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check --no-cache-dir -r requirements.lock.txt
.venv/bin/python -m pip check
if [ ! -f artifacts/model/manifest.json ]; then
    .venv/bin/python -m predictive.train --epochs 12 --seed 42
fi
.venv/bin/python -c 'from predictive.pipeline import load_bundle; load_bundle()'
.venv/bin/python -m unittest discover -s tests -v
