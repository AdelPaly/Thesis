"""One-command setup.

    python install.py                # check the results: small, no key, no model
    python install.py --run          # everything needed to run the configurations
    python install.py --geoparser    # also the Irchel Geoparser environment

Without options it creates .venv with the three packages that scoring needs,
then re-computes the tables in results/summary/ from the committed predictions
and compares them (scripts/check_rescore.py). That takes a few minutes and
needs no key, no model and no network beyond the package download.

--run installs the pinned requirements (requirements.lock, several GB, mostly
PyTorch) so that the configurations themselves can be run, and checks that the
runner starts. --geoparser implies --run.

Safe to run again: an existing environment is reused, an existing .env is
never overwritten.

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
LOCK = ROOT / "requirements.lock"
SCORING_PACKAGES = ("PyYAML", "numpy", "scipy")   # all that scripts/score.py and analyse.py import
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


def works(python: Path) -> bool:
    """An interpreter file can be left behind by an interrupted creation, so
    the environment counts only if its pip answers."""
    if not python.exists():
        return False
    return subprocess.run([str(python), "-m", "pip", "--version"],
                          capture_output=True, cwd=ROOT).returncode == 0


def make_env(env_dir: Path) -> Path:
    python = venv_python(env_dir)
    if works(python):
        print(f"{env_dir.name}: already there, reusing it")
    else:
        if env_dir.exists():
            print(f"{env_dir.name}: incomplete, creating it again")
        else:
            print(f"{env_dir.name}: creating")
        try:
            venv.EnvBuilder(with_pip=True, clear=env_dir.exists()).create(env_dir)
        except (subprocess.CalledProcessError, OSError) as e:
            raise SystemExit(
                f"error: could not create {env_dir.name} ({e}).\n"
                f"On Debian and Ubuntu the venv module is a separate package: "
                f"sudo apt install python{TESTED[0]}.{TESTED[1]}-venv") from None
    run(python, "-m", "pip", "install", "--upgrade", "pip")
    return python


def pinned(names) -> list[str]:
    """The lock's own lines for these packages, so the light install uses the
    same versions as the full one."""
    wanted = {n.lower() for n in names}
    lines = [line.strip() for line in LOCK.read_text(encoding="utf-8").splitlines()
             if line.split("==")[0].strip().lower() in wanted]
    if len(lines) != len(wanted):
        raise SystemExit(f"error: {LOCK.name} does not pin all of {sorted(wanted)}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="store_true",
                    help="install everything needed to run the configurations (several GB)")
    ap.add_argument("--geoparser", action="store_true",
                    help="also create .venv-geoparser and download its gazetteer "
                         "(about 10 GB); implies --run")
    ap.add_argument("--any-python", action="store_true",
                    help=f"try with this Python although the pins were made for "
                         f"{TESTED[0]}.{TESTED[1]}")
    args = ap.parse_args()
    full = args.run or args.geoparser

    if sys.version_info[:2] != TESTED and not args.any_python:
        print(f"error: the pinned packages were made for Python {TESTED[0]}.{TESTED[1]}; "
              f"this is {sys.version.split()[0]}.\n"
              f"Install Python {TESTED[0]}.{TESTED[1]} and run this script with it, "
              f"or pass --any-python to try anyway.")
        return 1
    if full and shutil.which("git") is None:
        print("error: git is needed (ReFinED is installed from its repository)")
        return 1

    python = make_env(ROOT / ".venv")
    if full:
        run(python, "-m", "pip", "install", "-r", LOCK)
    else:
        run(python, "-m", "pip", "install", *pinned(SCORING_PACKAGES))

    for template in ENV_TEMPLATES:
        target = template.with_name(".env")
        if not target.exists():
            shutil.copyfile(template, target)
            print(f"wrote {target.relative_to(ROOT)} (keys go there; scoring needs none)")

    if args.geoparser:
        geo = make_env(ROOT / ".venv-geoparser")
        run(geo, "-m", "pip", "install", f"geoparser=={GEOPARSER_VERSION}")
        run(geo, "-m", "spacy", "download", "en_core_web_sm")
        run(geo, "-m", "geoparser", "install", "geonames")

    rel = python.relative_to(ROOT)
    if full:
        run(python, ROOT / "src" / "runner.py", "--list")
        print("\nInstalled. Try a configuration that needs no model and no key:")
        print(f"  {rel} src/runner.py --config majority_class --dataset synthetic --limit 20")
    else:
        print("\nRe-computing the result tables from the committed predictions "
              "(a few minutes)...")
        run(python, ROOT / "scripts" / "check_rescore.py")
        print("\nThe tables in results/summary/ reproduce. To look at the scores:")
        print(f"  {rel} scripts/score.py --split test --by condition")
        print("To run the configurations themselves: python install.py --run")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as e:
        print(f"\nfailed: {' '.join(str(c) for c in e.cmd)} (exit {e.returncode})")
        print("Nothing was removed; fix the cause and run the same command again.")
        sys.exit(e.returncode or 1)
