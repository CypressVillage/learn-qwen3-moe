"""Generate, validate, and build the static course website."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
WEB_ROOT = ROOT / "web"


def run(command: list[str], *, cwd: Path = ROOT) -> None:
    print(f"+ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


for step in range(15):
    run([sys.executable, f"scripts/generate_step{step:02d}_assets.py"])

run([sys.executable, "scripts/validate_course_assets.py"])
run(["npm", "ci"], cwd=WEB_ROOT)
run(["npm", "run", "build"], cwd=WEB_ROOT)

print(f"site built at {WEB_ROOT / 'dist'}")
