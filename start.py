"""Start the environment prepared by install.py, without manual activation."""
import json
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    manifest = root / "external/installation.json"
    if not manifest.is_file():
        print("Run python install.py first.", file=sys.stderr)
        return 1
    settings = json.loads(manifest.read_text(encoding="utf-8"))
    if not Path(settings["conda"]).is_file() or not Path(settings["prefix"]).is_dir():
        print("Installation paths have changed. Run python install.py again.", file=sys.stderr)
        return 1
    return subprocess.call([settings["conda"], "run", "--no-capture-output", "--prefix", settings["prefix"], "python", str(root / "run.py"), *sys.argv[1:]], cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
