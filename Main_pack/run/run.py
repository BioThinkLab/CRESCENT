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
from typing import List, Optional, Tuple

# ✅ 新增：用于输入文件列处理
import pandas as pd
import numpy as np


project_name = "RUBIC1"
# Mutation type: amp / del
mutation_type = "amp"  # or mut="del"
# Whether to sort arm_level and focal segment and only use focal
only_use_focal = False
#

PROJECT_DIR = Path("./").resolve()
SRC_DIR = PROJECT_DIR / "src"   # C/C++ source files shold be placed under ./src
EXECUTABLE_NAME = "processor_amp"
BUILD_DIR = PROJECT_DIR / "build"

if only_use_focal:
    INPUT_BASE = Path("./output/sorted")
else:
    INPUT_BASE = Path("../Data/input")

OUTPUT_BASE = Path("./bin_with_case_amp")
SPECIFIC_SOURCES: List[str] = ["gen_bin_amp_cpp.cpp"]

# ✅ 新增：预处理后输入目录（临时）
PREP_BASE = PROJECT_DIR / "._prepared_inputs"


def is_windows() -> bool:
    return os.name == "nt"


def exe_path() -> Path:
    ext = ".exe" if is_windows() else ""
    return BUILD_DIR / f"{EXECUTABLE_NAME}{ext}"


def newest_mtime_in(paths: List[Path]) -> float:
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
    if explicit:
        out: List[Path] = []
        for s in explicit:
            p = Path(s)
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
    for p in sorted(scan_dir.iterdir()):
        if p.is_dir() and not p.name.startswith("."):
            yield p


# =========================
# ✅ 新增：输入文件预处理逻辑
# =========================

def _detect_sep(path: Path) -> str:
    # 优先按扩展名判断；否则默认 tab
    suf = path.suffix.lower()
    if suf == ".csv":
        return ","
    # .tsv / .txt 默认 tab
    return "\t"


def _is_table_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".tsv", ".txt", ".csv"}


def _normalize_dataframe(df: pd.DataFrame, src_path: Path) -> pd.DataFrame:
    if df.shape[1] < 2:
        return df

    # 0) 清理列名：去掉前后空格（非常关键）
    df.columns = [str(c).strip() for c in df.columns]

    # 1) 第一列：统一改名为 GDC_Aliquot
    first_col = df.columns[0]
    if first_col != "GDC_Aliquot":
        df = df.rename(columns={first_col: "GDC_Aliquot"})

    cols = set(df.columns)

    # 2) 统一 CopyNumber 列命名到 Copy_Number（下游要这个）
    #    兼容输入里叫 "Copy Number" 的情况
    if "Copy_Number" not in cols and "Copy Number" in cols:
        df = df.rename(columns={"Copy Number": "Copy_Number"})
        cols = set(df.columns)

    # 3) 如果没有 Copy_Number，但有 Segment_Mean，就推导生成 Copy_Number
    if "Copy_Number" not in cols:
        if "Segment_Mean" in cols:
            seg = pd.to_numeric(df["Segment_Mean"], errors="coerce")

            # Segment_Mean = log2(CN/2)  => CN = 2 * 2^Segment_Mean
            cn = 2.0 * np.power(2.0, seg)

            # 你可以选择 round / floor / keep float
            # 下游一般要整数 CN：这里用 round
            df["Copy_Number"] = np.rint(cn).astype("Int64")
        else:
            # 两者都没有：不动，让下游报错更清晰
            pass

    # 4)（可选）把 Copy_Number 放在靠前位置，降低某些程序对列顺序敏感的风险
    if "Copy_Number" in df.columns:
        cols_order = list(df.columns)
        # 确保 GDC_Aliquot 第一列
        cols_order.remove("GDC_Aliquot")
        new_order = ["GDC_Aliquot"] + cols_order
        df = df[new_order]

    return df



def prepare_input_dir(input_dir: Path, prepared_root: Path) -> Tuple[Path, int, int]:
    """
    把 input_dir 复制/预处理到 prepared_root/input_dir.name 下，返回：
      - prepared_dir
      - processed_files_count
      - modified_files_count（发生过列名/列新增改动）
    """
    prepared_dir = prepared_root / input_dir.name
    if prepared_dir.exists():
        shutil.rmtree(prepared_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    modified = 0

    # 复制所有内容；表格文件做标准化
    for p in input_dir.rglob("*"):
        rel = p.relative_to(input_dir)
        tgt = prepared_dir / rel
        if p.is_dir():
            tgt.mkdir(parents=True, exist_ok=True)
            continue

        if not _is_table_file(p):
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, tgt)
            continue

        sep = _detect_sep(p)
        try:
            df = pd.read_csv(p, sep=sep, dtype=str)  # 先全当字符串，避免奇怪类型推断
        except Exception:
            # 读取失败就原样复制，避免破坏原结构；下游自行报错
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, tgt)
            continue

        processed += 1

        before_cols = list(df.columns)

        # 转换数值列前，先让 pandas 重新管理：保留原始内容
        df2 = df.copy()
        df2 = _normalize_dataframe(df2, p)

        after_cols = list(df2.columns)
        if before_cols != after_cols or ("Copy Number" in after_cols and "Copy Number" not in before_cols):
            modified += 1

        # 写回：保持原分隔符风格
        tgt.parent.mkdir(parents=True, exist_ok=True)
        df2.to_csv(tgt, sep=sep, index=False)

    return prepared_dir, processed, modified


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
        "1" if only_use_focal else "0"
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

import re

def _normalize_chr_token(tok: str) -> str:
    """
    输入可能是 '1', 'chr1', 'CHR1', 'X', 'chrX', 'MT' 等
    输出统一为 'chr1', 'chrX', 'chrM'
    """
    s = str(tok).strip()
    if not s:
        return s

    s = s.replace("CHR", "chr").replace("Chr", "chr").replace("chR", "chr")
    if s.lower().startswith("chr"):
        core = s[3:]
    else:
        core = s

    core = core.strip()
    core_up = core.upper()

    if core_up in {"M", "MT"}:
        return "chrM"
    if core_up in {"X", "Y"}:
        return f"chr{core_up}"

    # 数字
    if re.fullmatch(r"\d+", core):
        return f"chr{int(core)}"

    # 兜底：原样加 chr 前缀
    return "chr" + core


def _guess_chr_from_filename(name: str) -> Optional[str]:
    """
    从文件名猜染色体：支持 ...chr1... / ..._1... / ...ChrX... / ...chrMT...
    """
    stem = Path(name).stem

    # 优先抓 chr 形式
    m = re.search(r"(?:^|[^A-Za-z0-9])(chr(?:\d+|X|Y|M|MT))(?:[^A-Za-z0-9]|$)", stem, flags=re.IGNORECASE)
    if m:
        return _normalize_chr_token(m.group(1))

    # 再抓裸数字 / X / Y / MT（避免误伤，尽量靠近分隔符）
    m = re.search(r"(?:^|[^A-Za-z0-9])(\d+|X|Y|MT|M)(?:[^A-Za-z0-9]|$)", stem, flags=re.IGNORECASE)
    if m:
        return _normalize_chr_token(m.group(1))

    return None


def unify_output_files(output_dir: Path, prefix: str = "cnv_", force_content_chr: bool = True) -> None:
    """
    把 output_dir 下的文件统一为：cnv_chr1.txt, cnv_chr2.txt ...
    - 允许源文件扩展名为 .tsv/.txt/.csv
    - 若 force_content_chr=True，会尝试把内容中的染色体列统一为 chr*
    """
    if not output_dir.exists():
        return

    # 收集候选：你现在看到的 .tsv/.txt 混用都覆盖
    candidates = []
    for p in output_dir.iterdir():
        if p.is_file() and p.suffix.lower() in {".tsv", ".txt", ".csv"}:
            candidates.append(p)

    # 避免重复覆盖：同一 chr 可能出现多个文件，先按修改时间排序，保留最新的
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    used = set()
    for p in candidates:
        chr_norm = _guess_chr_from_filename(p.name)
        if not chr_norm:
            # 如果文件名看不出来，就跳过（你也可以选择读内容猜）
            continue

        out_name = f"{prefix}{chr_norm}.txt"
        out_path = output_dir / out_name

        # 同一 chr 已经有输出了，跳过旧的，避免覆盖
        if chr_norm in used:
            continue
        used.add(chr_norm)

        if force_content_chr:
            # 尽量不引入复杂推断：只要发现有明显的染色体列，就标准化内容
            try:
                # 自动猜分隔符：优先 tab，其次逗号
                with open(p, "r", encoding="utf-8") as f:
                    header = f.readline()
                sep = "\t" if "\t" in header else ("," if "," in header else None)

                if sep:
                    df = pd.read_csv(p, sep=sep, dtype=str)
                    df.columns = [str(c).strip() for c in df.columns]

                    # 常见列名：Chromosome / chr / Chr / chromosome 等
                    chr_cols = [c for c in df.columns if c.lower() in {"chromosome", "chr", "chrom", "chromosome_name"}]
                    if chr_cols:
                        c0 = chr_cols[0]
                        df[c0] = df[c0].apply(_normalize_chr_token)

                    # 统一写成 tab 分隔的 txt（无论原来是 tsv/txt/csv）
                    df.to_csv(out_path, sep="\t", index=False)
                    if out_path.resolve() != p.resolve():
                        p.unlink(missing_ok=True)
                    continue
            except Exception:
                # 内容处理失败就退化为纯重命名
                pass

        # 退化：只改名 + 改扩展名
        if out_path.exists():
            out_path.unlink()
        p.rename(out_path)


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
        default="",
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

    # ✅ 新增：创建预处理根目录
    PREP_BASE.mkdir(parents=True, exist_ok=True)

    failed = []
    for child in iter_child_dirs(scan_dir):
        # ✅ 新增：对每个 child 做输入预处理（列标准化 + Segment_Mean->Copy Number）
        prepared_dir, processed_cnt, modified_cnt = prepare_input_dir(child, PREP_BASE)
        if processed_cnt > 0:
            print(f"🧹 Prepared input for {child.name}: table_files={processed_cnt}, modified={modified_cnt}")

        input_dir = prepared_dir
        output_dir = output_base / child.name

        rc = run_one(exe, input_dir, output_dir, log_dir=log_dir)
        if rc == 0:
            unify_output_files(output_dir, prefix="cnv_", force_content_chr=True)
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
