#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

VERSION = "v2.0_species_only"
MODEL_SCRIPT = "Step10_4B_species_only_LOSO_modeling_v2.py"
LOCAL_ENV_DIR = ".step10_4b_pydeps_cp312"
LOG_NAME = "Step10_4B_species_only_v2.log"
EXPECTED_WHEELS = {
    "numpy-2.2.6-cp312-cp312-win_amd64.whl": 12_000_000,
    "scipy-1.15.3-cp312-cp312-win_amd64.whl": 40_000_000,
    "scikit_learn-1.6.1-cp312-cp312-win_amd64.whl": 10_000_000,
    "joblib-1.5.1-py3-none-any.whl": 250_000,
    "threadpoolctl-3.6.0-py3-none-any.whl": 10_000,
}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def clean_root_argument(raw: str) -> Path:
    # Windows command-line parsing can retain a quote when a quoted path ends
    # with a backslash. Remove any accidental wrapping/trailing quotes and
    # separators before resolving the project directory.
    value = raw.strip()
    value = value.strip('"')
    value = value.rstrip("\\/")
    value = value.rstrip('"')
    value = value.rstrip("\\/")
    if not value:
        raise ValueError("The project root argument is empty after sanitization.")
    return Path(value).resolve()


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8", errors="replace")

    def line(self, message: str = "") -> None:
        rendered = f"[{now()}] {message}"
        print(rendered, flush=True)
        self.handle.write(rendered + "\n")
        self.handle.flush()

    def raw(self, message: str) -> None:
        print(message, end="", flush=True)
        self.handle.write(message)
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def stream(
    command: List[str],
    cwd: Path,
    logger: Logger,
    env: Dict[str, str] | None = None,
) -> int:
    logger.line("Command: " + subprocess.list2cmdline(command))
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for output_line in process.stdout:
        logger.raw(output_line)
    return process.wait()


def validate_runtime(logger: Logger) -> bool:
    logger.line(f"Python executable: {sys.executable}")
    logger.line(f"Python version: {sys.version.split()[0]}")
    logger.line(f"Architecture: {platform.machine()} / {platform.architecture()[0]}")
    if sys.version_info[:2] != (3, 12):
        logger.line(
            "This offline bundle requires Python 3.12 because the bundled "
            "compiled wheels target CPython 3.12."
        )
        return False
    if sys.maxsize <= 2**32:
        logger.line("This offline bundle requires 64-bit Python.")
        return False
    return True


def validate_wheels(root: Path, logger: Logger) -> bool:
    wheel_dir = root / "wheels"
    if not wheel_dir.is_dir():
        logger.line(f"Wheel directory is missing: {wheel_dir}")
        return False

    valid = True
    for filename, minimum_size in EXPECTED_WHEELS.items():
        path = wheel_dir / filename
        if not path.is_file():
            logger.line(f"Missing bundled wheel: {filename}")
            valid = False
        elif path.stat().st_size < minimum_size:
            logger.line(
                f"Bundled wheel is unexpectedly small: {filename} "
                f"({path.stat().st_size} bytes)"
            )
            valid = False
        else:
            logger.line(f"Wheel ready: {filename} ({path.stat().st_size} bytes)")
    return valid


def ensure_pip(root: Path, logger: Logger) -> bool:
    rc = stream([sys.executable, "-m", "pip", "--version"], root, logger)
    if rc == 0:
        return True
    logger.line("pip is unavailable; running Python ensurepip offline.")
    return stream(
        [sys.executable, "-m", "ensurepip", "--upgrade"], root, logger
    ) == 0


def import_check(local_env: Path) -> Tuple[bool, str]:
    env = os.environ.copy()
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(local_env) + (
        os.pathsep + current if current else ""
    )
    code = (
        "import numpy, scipy, sklearn, joblib, threadpoolctl; "
        "from sklearn.linear_model import LogisticRegression; "
        "from sklearn.ensemble import RandomForestClassifier; "
        "from sklearn.model_selection import GroupKFold; "
        "print('numpy=' + numpy.__version__ + '; scipy=' + scipy.__version__ + "
        "'; sklearn=' + sklearn.__version__ + '; joblib=' + joblib.__version__ + "
        "'; threadpoolctl=' + threadpoolctl.__version__)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    return result.returncode == 0, result.stdout


def install_local_environment(
    root: Path,
    local_env: Path,
    logger: Logger,
) -> bool:
    wheel_dir = root / "wheels"
    if local_env.exists():
        logger.line(f"Removing incomplete local environment: {local_env}")
        shutil.rmtree(local_env, ignore_errors=True)
    local_env.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-cache-dir",
        "--only-binary=:all:",
        "--find-links",
        str(wheel_dir),
        "--target",
        str(local_env),
        "numpy==2.2.6",
        "scipy==1.15.3",
        "scikit-learn==1.6.1",
        "joblib==1.5.1",
        "threadpoolctl==3.6.0",
    ]
    rc = stream(command, root, logger)
    if rc != 0:
        logger.line(f"Offline pip installation failed with exit code {rc}.")
        return False

    ok, detail = import_check(local_env)
    logger.raw(detail if detail.endswith("\n") else detail + "\n")
    if not ok:
        logger.line(
            "The project-local environment was installed, but import "
            "validation failed."
        )
        return False
    logger.line("Project-local Python environment passed import validation.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    # Sanitize before constructing the log path. This directly fixes WinError 123.
    try:
        root = clean_root_argument(args.root)
    except Exception:
        print("Failed to normalize the project root argument.", flush=True)
        traceback.print_exc()
        return 10

    logger = Logger(root / "03_logs" / LOG_NAME)
    try:
        logger.line(f"Step 10.4B fully offline bootstrap {VERSION} started.")
        logger.line(f"Raw root argument: {args.root!r}")
        logger.line(f"Normalized project root: {root}")
        logger.line("No internet access will be used.")
        logger.line(
            "Dependencies are installed only inside the project folder."
        )

        if not root.is_dir():
            logger.line(f"Normalized project root is not a directory: {root}")
            return 11
        if not validate_runtime(logger):
            return 2
        if not validate_wheels(root, logger):
            return 3

        model_script = root / MODEL_SCRIPT
        if not model_script.is_file():
            logger.line(f"Modeling script is missing: {model_script}")
            return 4

        if not ensure_pip(root, logger):
            logger.line("pip/ensurepip could not be initialized.")
            return 5

        local_env = root / LOCAL_ENV_DIR
        ok, detail = import_check(local_env)
        if ok:
            logger.line("Reusing the existing project-local environment.")
            logger.raw(detail if detail.endswith("\n") else detail + "\n")
        else:
            logger.line(
                "A valid project-local environment was not found. "
                "Installing from bundled wheels."
            )
            logger.raw(detail if detail.endswith("\n") else detail + "\n")
            if not install_local_environment(root, local_env, logger):
                return 6

        env = os.environ.copy()
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(local_env) + (
            os.pathsep + current_pythonpath if current_pythonpath else ""
        )
        env.setdefault("OMP_NUM_THREADS", "2")
        env.setdefault("MKL_NUM_THREADS", "2")
        env.setdefault("OPENBLAS_NUM_THREADS", "2")
        env.setdefault("NUMEXPR_NUM_THREADS", "2")

        logger.line(
            "Starting Step 10.4B modeling with the project-local environment."
        )
        rc = stream(
            [sys.executable, str(model_script), "--root", str(root)],
            root,
            logger,
            env=env,
        )
        if rc != 0:
            logger.line(f"Modeling script failed with exit code {rc}.")
            return rc

        result_zip = root / "Step10_4B_results_v2_species_only.zip"
        if not result_zip.is_file() or result_zip.stat().st_size == 0:
            logger.line(
                "Modeling returned success but Step10_4B_results_v2_species_only.zip "
                "was not created."
            )
            return 7

        logger.line(
            f"Step 10.4B completed successfully: {result_zip} "
            f"({result_zip.stat().st_size} bytes)"
        )
        return 0
    except KeyboardInterrupt:
        logger.line("Interrupted by user.")
        return 130
    except Exception:
        logger.line("Unexpected bootstrap error. Full traceback follows:")
        logger.raw(traceback.format_exc())
        return 1
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
