"""One-command setup.

    python install.py                # main environment (.venv)
    python install.py --geoparser    # also the Irchel Geoparser environment

Creates .venv, installs the pinned requirements (requirements.lock), copies the
.env templates into place if you have no .env yet, and checks that the runner
starts. Safe to run again: an existing environment is reused and an existing
.env is never overwritten.

Standard library only, so it runs with any Python before anything is installed.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTED = (3, 13)
GEOPARSER_VERSION = "0.6.1"   # must equal geoparser.version in configs/geoparser_irchel.yaml
ENV_TEMPLATES = [ROOT / "src" / ".env.example",
                 ROOT / "generation" / "synthetic" / ".env.example"]


def venv_python(env_dir: Path) -> Path:
    if sys.platform == "win32":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def run(*cmd) -> None:
    print("\n>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT)


def make_env(env_dir: Path) -> Path:
    python = venv_python(env_dir)
    if python.exists():
        print(f"{env_dir.name}: already there, reusing it")
    else:
        print(f"{env_dir.name}: creating")
        venv.EnvBuilder(with_pip=True).create(env_dir)
    run(python, "-m", "pip", "install", "--upgrade", "pip")
    return python


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--geoparser", action="store_true",
                    help="also create .venv-geoparser for the geoparser_irchel configuration")
    args = ap.parse_args()

    if sys.version_info[:2] != TESTED:
        print(f"note: the pinned requirements were made with Python "
              f"{TESTED[0]}.{TESTED[1]}; this is {sys.version.split()[0]}. "
              f"If the install fails, use Python {TESTED[0]}.{TESTED[1]}.")
    if shutil.which("git") is None:
        print("error: git is needed (ReFinED is installed from its repository)")
        return 1

    python = make_env(ROOT / ".venv")
    run(python, "-m", "pip", "install", "-r", ROOT / "requirements.lock")

    for template in ENV_TEMPLATES:
        target = template.with_name(".env")
        if not target.exists():
            shutil.copyfile(template, target)
            print(f"wrote {target.relative_to(ROOT)} (add your keys there)")

    if args.geoparser:
        geo = make_env(ROOT / ".venv-geoparser")
        run(geo, "-m", "pip", "install", f"geoparser=={GEOPARSER_VERSION}")
        run(geo, "-m", "spacy", "download", "en_core_web_sm")
        print("\nThe geoparser still needs its GeoNames gazetteer (about 10 GB, once):")
        print(f"  {geo.relative_to(ROOT)} -m geoparser install geonames")

    run(python, ROOT / "src" / "runner.py", "--list")
    rel = python.relative_to(ROOT)
    print("\nInstalled. Try a configuration that needs no model and no key:")
    print(f"  {rel} src/runner.py --config majority_class --dataset synthetic --limit 20")
    print(f"  {rel} scripts/score.py")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as e:
        print(f"\nfailed: {' '.join(e.cmd)} (exit {e.returncode})")
        sys.exit(e.returncode or 1)
