"""
STRICT DeepISA environment validator.

Enforces EXACT version alignment with validated DeepISA environment.
Exits immediately with diff on ANY mismatch.

Usage:
    from deepISA.utils.deepisa_guard import validate_deepisa_environment
    validate_deepisa_environment()
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# ─── Validated DeepISA environment ───────────────────────────────────────────
_DEEPISA_PYTHON = "3.9.18"
_DEEPISA_PACKAGES = {
    "numpy": "1.26.4",
    "pandas": "2.3.3",
    "torch": "2.8.0",
    "captum": "0.8.0",
    "scipy": "1.13.1",
    "bioframe": "0.8.0",
    "pysam": "0.23.3",
    "pyBigWig": "0.3.25",
    "matplotlib": "3.9.4",
    "seaborn": "0.13.2",
    "pyyaml": "6.0.3",
    "tqdm": "4.67.3",
    "loguru": "0.7.3",
    "scikit-learn": "1.5.2",
    "statsmodels": "0.14.4",
    "adjustText": "0.8",
}


def _get_installed_version(pkg: str) -> str | None:
    """Return installed version of a package, or None if not installed."""
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", None)
        if ver is None:
            ver = getattr(mod, "version__", None)
        return str(ver) if ver else None
    except ImportError:
        return None


def _check_samtools() -> tuple[bool, str]:
    """Check samtools availability. Returns (ok, path_or_error_msg)."""
    path = os.environ.get("SAMTOOLS", "")
    if path and Path(path).exists():
        return True, path
    for candidate in ["samtools", "/usr/bin/samtools", "/usr/local/bin/samtools"]:
        try:
            subprocess.run(
                [candidate, "--version"],
                capture_output=True,
                timeout=5,
                check=True,
            )
            return True, candidate
        except Exception:
            pass
    return False, (
        "samtools not found. Install: sudo apt install samtools  # Linux\n"
        "or brew install samtools  # Mac\n"
        "Or set SAMTOOLS env var to the samtools binary path."
    )


def validate_deepisa_environment() -> None:
    """
    Hard-check that the current environment EXACTLY matches DeepISA.

    Raises RuntimeError immediately on any mismatch.
    Prints expected vs actual diff before exiting.
    """
    errors: list[str] = []

    # ── Python version ──────────────────────────────────────────────────────
    actual_py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if actual_py != _DEEPISA_PYTHON:
        errors.append(
            f"  Python: expected {_DEEPISA_PYTHON}, got {actual_py}\n"
            f"           required: python=={_DEEPISA_PYTHON}"
        )

    # ── Package versions ───────────────────────────────────────────────────
    mismatches: list[tuple[str, str, str]] = []
    for pkg, expected_ver in _DEEPISA_PACKAGES.items():
        installed = _get_installed_version(pkg)
        if installed is None:
            mismatches.append((pkg, expected_ver, "NOT INSTALLED"))
        elif installed != expected_ver:
            mismatches.append((pkg, expected_ver, installed))

    if mismatches:
        errors.append("  Package version mismatches:")
        for pkg, expected, actual in mismatches:
            errors.append(f"    {pkg}: expected {expected}, got {actual}")

    # ── samtools ───────────────────────────────────────────────────────────
    samtools_ok, samtools_msg = _check_samtools()
    if not samtools_ok:
        errors.append(f"  samtools: {samtools_msg}")

    # ── Report or pass ──────────────────────────────────────────────────────
    if errors:
        header = (
            "\n"
            "=" * 60 + "\n"
            "DeepISA ENVIRONMENT MISMATCH\n"
            "=" * 60 + "\n"
            "Your environment does NOT match the validated DeepISA environment.\n"
            "Scientific results may differ if versions diverge.\n"
            "\n"
            "Expected (DeepISA validated):\n"
        )
        for pkg, ver in _DEEPISA_PACKAGES.items():
            header += f"  {pkg}=={ver}\n"
        header += f"  python=={_DEEPISA_PYTHON}\n"
        header += "\nMismatches:\n" + "\n".join(errors) + "\n"
        header += (
            "\n"
            "To fix: pip install -r requirements.lock.txt\n"
            "=" * 60 + "\n"
        )
        sys.stderr.write(header)
        raise RuntimeError("DeepISA environment mismatch — see diff above")

    # ── All checks passed ──────────────────────────────────────────────────
    print("[deepisa_guard] Environment validated: EXACT MATCH with DeepISA")
