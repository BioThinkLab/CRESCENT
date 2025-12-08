#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sort import run_all_cancer_types
from compress import compress
from window_sampling import generate_samples_auto_parallel
from check import run_inf
from mapping import run_integration
from threshold import to_segment
import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional


project_name = "EXAMPLE"
# Mutation type: amp / del
mutation_type = "amp"  # or mut="del"
# Whether to sort arm_level and focal segment and only use focal
only_use_focal = False
#


PROJECT_DIR = Path("./").resolve()
SRC_DIR = PROJECT_DIR / "src"   # C/C++ source files shold be placed under ./src
# Executable name (without extension; .exe will be added on Windows)
# Will be overwritten in main according to mut
EXECUTABLE_NAME = "processor_amp"

# Build output directory (for executable etc.)
BUILD_DIR = PROJECT_DIR / "build"

# Input / output base directories (fixed).
# Each subdirectory under INPUT_BASE will map to a directory with the same
# name under OUTPUT_BASE.
if only_use_focal:
    INPUT_BASE = Path("./output/sorted")
else:
    INPUT_BASE = Path("../Data/input")

# Will be overwritten in main according to mut
OUTPUT_BASE = Path("./bin_with_case_amp")

# Only compile these specified cpp files (relative to SRC_DIR or absolute path).
# Will be overwritten in main according to mut.
SPECIFIC_SOURCES: List[str] = ["gen_bin_amp_cpp.cpp"]
# ========================================================


def is_windows() -> bool:
    return os.name == "nt"


def exe_path() -> Path:
    ext = ".exe" if is_windows() else ""
    return BUILD_DIR / f"{EXECUTABLE_NAME}{ext}"


def newest_mtime_in(paths: List[Path]) -> float:
    """
    Return the newest modification time (as float) among given files/directories.
    For directories, all files under them (recursively) are considered.
    """
    m = 0.0
    for p in paths:
        if p.is_file():
            try:
                m = max(m, p.stat().st_mtime)
            except FileNotFoundError:
                continue
        elif p.is_dir():
            for sub in p.rglob("*"):
                if sub.is_file():
                    try:
                        m = max(m, sub.stat().st_mtime)
                    except FileNotFoundError:
                        continue
    return m


def collect_sources(project_dir: Path, explicit: Optional[List[str]] = None) -> List[Path]:
    """
    If 'explicit' is provided (non-empty), only return the source files listed there.
    Relative paths are resolved against SRC_DIR.
    Otherwise, scan SRC_DIR for all common C/C++ source files.
    """
    if explicit:
        out: List[Path] = []
        for s in explicit:
            p = Path(s)
            # If not absolute, treat as path under SRC_DIR
            if not p.is_absolute():
                p = (SRC_DIR / p).resolve()
            if not p.exists():
                raise RuntimeError(f"Specified source file does not exist: {p}")
            if p.suffix.lower() not in {".cpp", ".cc", ".cxx", ".c"}:
                raise RuntimeError(f"Specified file is not a C/C++ source: {p}")
            out.append(p)
        return out

    suffixes = {".cpp", ".cc", ".cxx", ".c"}
    return [p for p in SRC_DIR.rglob("*") if p.suffix.lower() in suffixes]


def choose_compiler() -> List[str]:
    # Prefer g++, fallback to clang++; modify to ["cl.exe", ...] if using MSVC
    for c in ["g++", "clang++"]:
        if shutil.which(c):
            return [c]
    raise RuntimeError("No available C++ compiler found (g++ / clang++). Please install one.")


def compile_if_needed() -> Path:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    exe = exe_path()

    sources = collect_sources(PROJECT_DIR, explicit=SPECIFIC_SOURCES)
    if not sources:
        raise RuntimeError(f"No source file to compile: {SPECIFIC_SOURCES}")

    # Changes in headers should also trigger rebuild (used only for time check)
    headers_maybe = [p for p in SRC_DIR.rglob("*") if p.suffix.lower() in {".hpp", ".hh", ".h"}]
    src_mtime = max(newest_mtime_in(sources), newest_mtime_in(headers_maybe))
    exe_mtime = exe.stat().st_mtime if exe.exists() else 0.0

    if not exe.exists() or exe_mtime < src_mtime:
        print("🔧 Need to compile (executable does not exist or is older than sources/headers).")
        compiler = choose_compiler()

        flags = ["-std=gnu++17", "-O3", "-Wall", "-Wextra", "-Wno-unused-parameter"]
        if not is_windows():
            flags += ["-pthread"]

        cmd = compiler + flags + [str(p) for p in sources] + ["-o", str(exe)]
        print("Compile command:", " ".join(cmd))

        proc = subprocess.run(cmd, cwd=PROJECT_DIR)
        if proc.returncode != 0:
            raise RuntimeError("Compilation failed, please check the error output above.")
        else:
            print(f" Compilation succeeded: {exe}")
    else:
        print(f" Executable is up to date: {exe}")

    if not is_windows():
        exe.chmod(exe.stat().st_mode | 0o111)
    return exe


def iter_child_dirs(scan_dir: Path):
    """
    Yield only first-level subdirectories of scan_dir, ignoring hidden directories.
    """
    for p in sorted(scan_dir.iterdir()):
        if p.is_dir() and not p.name.startswith("."):
            yield p


def run_one(executable: Path, input_dir: Path, output_dir: Path, log_dir: Optional[Path] = None) -> int:
    """
    Run the executable once with given input_dir and output_dir.
    If log_dir is provided, stdout/stderr are written to log files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(executable),
        str(input_dir),
        str(output_dir),
        "1" if only_use_focal else "0"   # Still passed as use_arm_filter to C++
    ]

    print(f"▶️ Running: {' '.join(cmd)}")

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = input_dir.name
        stdout_path = log_dir / f"{name}_{ts}.out.txt"
        stderr_path = log_dir / f"{name}_{ts}.err.txt"
        with open(stdout_path, "w") as so, open(stderr_path, "w") as se:
            proc = subprocess.run(cmd, stdout=so, stderr=se)
            rc = proc.returncode
        print(f"  ↳ Return code {rc}, logs: {stdout_path.name}, {stderr_path.name}")
        return rc
    else:
        proc = subprocess.run(cmd)
        return proc.returncode


def main():
    global SPECIFIC_SOURCES, OUTPUT_BASE, EXECUTABLE_NAME, mutation_type

    if only_use_focal:
        run_all_cancer_types()

    parser = argparse.ArgumentParser(
        description="Auto-compile and batch-run a C++ program (each subdirectory as one run)."
    )
    parser.add_argument(
        "--scan-dir",
        type=str,
        default=str(INPUT_BASE),
        help="Directory to scan; each first-level subdirectory is used as input_dir for one run. Default ../Data/input"
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="",   # Resolved later together with mut
        help="Base directory for outputs. Each task outputs to <output_base>/<subdir_name>. "
             "Default is ./bin_with_case_amp or ./bin_with_case_del depending on mut."
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="",
        help="Directory to save stdout/stderr logs for each task (leave empty to print directly to console)."
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="",
        help="Comma-separated list of source files to compile only (relative to SRC_DIR or absolute paths)."
    )
    parser.add_argument(
        "--mut",
        type=str,
        choices=["amp", "del"],
        default=mutation_type,
        help="Mutation type: amp or del, used to select different C++ sources and output directories."
    )
    args = parser.parse_args()

    # Choose C++ source / executable name / default output directory based on mut
    mutation_type = args.mut
    if mutation_type == "amp":
        EXECUTABLE_NAME = "processor_amp"
        if not args.output_base:
            OUTPUT_BASE = Path("./bin_with_case_amp")
        else:
            OUTPUT_BASE = Path(args.output_base)
        if not args.sources:
            SPECIFIC_SOURCES = ["gen_bin_amp_cpp.cpp"]
    else:  # del
        EXECUTABLE_NAME = "processor_del"
        if not args.output_base:
            OUTPUT_BASE = Path("./bin_with_case_del")
        else:
            OUTPUT_BASE = Path(args.output_base)
        if not args.sources:
            SPECIFIC_SOURCES = ["gen_bin_del_cpp.cpp"]

    scan_dir = Path(args.scan_dir).resolve()
    output_base = OUTPUT_BASE.resolve()
    log_dir = Path(args.log_dir).resolve() if args.log_dir else None

    # Allow command line to override SPECIFIC_SOURCES
    if args.sources:
        srcs = [s.strip() for s in args.sources.split(",") if s.strip()]
        SPECIFIC_SOURCES = srcs

    if not scan_dir.exists() or not scan_dir.is_dir():
        print(f"❌ Invalid scan directory: {scan_dir}")
        sys.exit(1)

    try:
        exe = compile_if_needed()
    except Exception as e:
        print(f"❌ Compilation stage failed: {e}")
        sys.exit(2)

    failed = []
    for child in iter_child_dirs(scan_dir):
        input_dir = child
        output_dir = output_base / child.name
        rc = run_one(exe, input_dir, output_dir, log_dir=log_dir)
        if rc != 0:
            failed.append((child.name, rc))

    if failed:
        print("\n!!️ The following tasks failed:")
        for name, rc in failed:
            print(f"  - {name}: return code {rc}")
        sys.exit(3)
    else:
        print("\n gen bins successfully\n")

    print("doing compressing or padding")
    compress()
    generate_samples_auto_parallel(project_name)
    run_inf(project_name, mutation_type)
    run_integration(mut=mutation_type, cancer_types=project_name)
    to_segment(cancer_type=project_name, mut_type=mutation_type)



if __name__ == "__main__":
    main()
