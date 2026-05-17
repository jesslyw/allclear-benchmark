#!/usr/bin/env python3
"""Usage: python run.py --model-name <model> [benchmark args...]"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENVS = {
    "vpint2": ".venv", "leastcloudy": ".venv", "mosaicing": ".venv",
    "emrdm": "emrdm", "uncrtaints": "uncrtaints",
}

args = sys.argv[1:]
model = next((args[i+1].lower() for i, a in enumerate(args)
             if a == "--model-name" and i+1 < len(args)), None)
if not model or model not in ENVS:
    sys.exit(f"Error: --model-name required. Available: {', '.join(ENVS)}")

python = HERE / ENVS[model] / "bin/python"
if not python.exists():
    sys.exit(f"Error: env '{ENVS[model]}' not found. See README.md for setup.")

sys.exit(subprocess.run(
    [str(python), str(HERE / "benchmark.py")] + args).returncode)
