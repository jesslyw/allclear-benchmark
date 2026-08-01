#!/usr/bin/env python3
"""
Entry point to run the benchmark

Runs benchmark.py with the selected model environment's Python interpreter
and forwards all provided CLI arguments

Usage: python run.py --model-name <model> [benchmark.py args...]

"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENVS = {
    "vpint2": ".venv", "leastcloudy": ".venv", "mosaicing": ".venv",
    "emrdm": "emrdm", "uncrtaints": "uncrtaints",
}

args = sys.argv[1:]
try:
    model = args[args.index("--model-name") + 1].lower()
except (ValueError, IndexError):
    model = None
if not model or model not in ENVS:
    sys.exit(f"Error: --model-name required. Available: {', '.join(ENVS)}")

python = HERE / ENVS[model] / "bin/python"
if not python.exists():
    sys.exit(f"Error: env '{ENVS[model]}' not found. See README.md for setup.")

sys.exit(subprocess.run(
    [str(python), str(HERE / "benchmark.py")] + args).returncode)
