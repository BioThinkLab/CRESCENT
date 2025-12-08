# -*- coding: utf-8 -*-
"""
Integrated version (automatically iterate over inference_results_{mut}_{cancer_type}.tsv):

- Automatically: scan INFER_DIR for all matching files, detect mut/ct from filename and process
- Filterable: you can specify MUT and/or CANCER_TYPES to filter which files are processed
- No intermediate files: directly map {suffix->prob} and interpolate back into original per-chromosome TSV

Requirements:
1) Inference result file naming: inference_results_{mut}_{cancer_type}.tsv
   - Examples: inference_results_del_GBM.tsv, inference_results_amp_UCEC.tsv
   - mut ∈ {'del','amp'}
2) Inference results must contain at least columns: filename, probability, prediction
   - filename example: BLCA_chr11_83599999_1_6445.tsv
3) Original per-chromosome TSV directory structure:
   FULL_ROOT_* / {ct} / *chr{N}.tsv (or .txt)
"""

import os
import glob
from pathlib import Path
import pandas as pd
from typing import Iterable, Optional

# These will be set by run_integration; kept as module globals so that
# existing helper functions can stay unchanged as much as possible.
CHROM_LIST: Iterable = list(range(1, 22 + 1))
FILE_EXTS = [".tsv", ".txt"]


def parse_infer_filename(path: Path):
    """
    Parse mut and ct from inference result filename.
    Convention: inference_results_{mut}_{ct}.tsv

    Return: (mut, ct); if filename does not match the convention, return None.
    """
    stem = path.stem  # e.g., inference_results_del_GBM
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    # ["inference", "results", "{mut}", "{ct}"]
    if parts[0] != "inference" or parts[1] != "results":
        return None
    mut = parts[2].lower()
    ct = parts[3].upper()
    if mut not in {"amp", "del"} or len(ct) < 2:
        return None
    return mut, ct


def parse_filename(fn: str):
    """
    Parse from 'filename' column string:
      ct (first 4 chars), chrom (e.g. 'chr11'),
      start (number between 2nd and 3rd '_'),
      suffix_id (number after last '_').

    Example: BLCA_chr11_83599999_1_6445.tsv
    -> ('BLCA', 'chr11', 83599999, 6445)
    """
    base = os.path.basename(fn)
    name, _ = os.path.splitext(base)
    parts = name.split('_')
    if len(parts) < 5:
        raise ValueError(f"Could not parse filename: {fn}")
    ct = parts[0]
    chrom = parts[1]
    start = int(parts[2])
    suffix_id = int(parts[-1])
    return ct, chrom, start, suffix_id


def build_mapping_series(df_infer: pd.DataFrame):
    """
    Given a single inference result DataFrame (with columns filename, probability),
    build a nested mapping:

        maps[ct][chrom] = Series(probability, index=suffix)

    - Within each (ct, chrom), rows are sorted by start (ascending).
    - For suffix, duplicates are removed (keep first occurrence) to avoid
      multiple entries with the same suffix.
    """
    parsed = df_infer["filename"].apply(parse_filename)
    df_infer[["ct", "chrom", "start", "suffix"]] = pd.DataFrame(parsed.tolist(), index=df_infer.index)

    maps = {}
    for (ct, chrom), df_grp in df_infer.groupby(["ct", "chrom"]):
        df_sorted = df_grp.sort_values(by="start", ascending=True)
        df_unique = df_sorted.drop_duplicates(subset="suffix", keep="first")

        s = pd.Series(
            df_unique["probability"].values,
            index=df_unique["suffix"].astype(int).values,
            dtype="float64",
        ).sort_index()

        maps.setdefault(ct, {})[chrom] = s
    return maps


def find_full_file(full_ct_dir: Path, chrom):
    """
    In <full_ct_dir>, find the original per-chromosome file corresponding to chr{chrom}.

    Allowed extensions are in FILE_EXTS. The search order:
      1) Strict: *chr{chrom}.{ext}
      2) Relaxed: *chr{chrom}*{ext}

    Return Path or None if not found.
    """
    chrom_tag = f"chr{chrom}"
    for ext in FILE_EXTS:
        cand = list(full_ct_dir.glob(f"*{chrom_tag}{ext}"))
        if cand:
            if len(cand) > 1:
                print(f"[WARN] {full_ct_dir.name}: multiple matches for {chrom_tag}{ext} -> {[p.name for p in cand]}, using {cand[0].name}")
            return cand[0]
    # Relaxed match
    cand = []
    for ext in FILE_EXTS:
        cand.extend(glob.glob(str(full_ct_dir / f"*{chrom_tag}*{ext}")))
    if cand:
        print(f"[WARN] {full_ct_dir.name}: relaxed match for {chrom_tag} -> {cand}, using {cand[0]}")
        return Path(cand[0])
    return None


def process_one_ct(ct: str, maps_ct: dict, full_root: Path, out_root: Path):
    """
    Map maps_ct (chrom -> Series[suffix->prob]) back to the original per-chromosome TSVs
    under full_root/<ct>/, and write results to out_root/<ct>/ with the same filenames
    (can be the same as full_root for in-place overwrite).
    """
    full_ct_dir = full_root / ct
    out_ct_dir = out_root / ct
    if not full_ct_dir.exists():
        print(f"[WARN] Original directory does not exist: {full_ct_dir}, skipping {ct}")
        return
    out_ct_dir.mkdir(parents=True, exist_ok=True)

    for chrom in CHROM_LIST:
        chrom_key = f"chr{chrom}"
        samp = maps_ct.get(chrom_key)
        if samp is None or samp.empty:
            print(f"[INFO] {ct} {chrom_key}: inference mapping missing, skip.")
            continue

        full_file = find_full_file(full_ct_dir, chrom)
        if full_file is None:
            print(f"[WARN] {ct} {chrom_key}: original file not found (dir {full_ct_dir}), skip.")
            continue

        # Read original TSV
        try:
            df = pd.read_csv(full_file, sep="\t")
        except Exception as e:
            print(f"[ERROR] Failed to read: {full_file} -> {e}")
            continue

        # Align: df.index (assumed original row index) with samp.index (suffix)
        df["prob"] = samp.reindex(df.index)
        # Interpolate + fill head and tail
        df["prob"] = df["prob"].interpolate().ffill().bfill()

        out_path = out_ct_dir / full_file.name
        try:
            df.to_csv(out_path, sep="\t", index=False)
            print(f"[OK] {ct} {chrom_key}: wrote {out_path} ({len(df)} rows)")
        except Exception as e:
            print(f"[ERROR] Failed to write: {out_path} -> {e}")


def process_infer_file(
    infer_file: Path,
    full_root_amp: Path,
    full_root_del: Path,
    out_root_amp: Path,
    out_root_del: Path,
):
    """
    Process a single inference_results file:
    - Automatically detect mut/ct from filename
    - Build mapping
    - Write back into original per-chromosome TSVs
    """
    parsed = parse_infer_filename(infer_file)
    if parsed is None:
        print(f"[SKIP] Filename does not match convention: {infer_file.name}")
        return
    mut, ct_from_name = parsed  # mut ∈ {'del','amp'}

    # Read inference results
    df_infer = pd.read_csv(infer_file, sep="\t")
    required_cols = {"filename", "probability", "prediction"}
    if not required_cols.issubset(df_infer.columns):
        print(f"[ERROR] {infer_file.name}: missing columns {required_cols}, actual: {list(df_infer.columns)}")
        return

    # Build mapping: (ct -> chrom -> Series[suffix->prob])
    maps = build_mapping_series(df_infer)

    # Usually a file contains only one ct; prefer ct from filename, otherwise use all keys from content
    target_cts = [ct_from_name] if ct_from_name in maps else list(maps.keys())
    if not target_cts:
        print(f"[INFO] {infer_file.name}: no valid ct parsed from content, skip.")
        return

    # Pick root directories based on mut
    if mut == "amp":
        full_root, out_root = full_root_amp, out_root_amp
    else:  # 'del'
        full_root, out_root = full_root_del, out_root_del

    print(f"\n[FILE] {infer_file.name} -> mut={mut}, cts={target_cts}")
    for ct in target_cts:
        maps_ct = maps.get(ct, {})
        if not maps_ct:
            print(f"[INFO] {infer_file.name}: {ct} mapping is empty, skip.")
            continue
        print(f"[CT] Processing {ct} ...")
        process_one_ct(ct, maps_ct, full_root, out_root)



def run_integration(
    infer_dir: Path = Path("./buffer/inference_buffer"),
    mut: Optional[str] = None,
    cancer_types: Optional[Iterable[str]] = None,
    full_root_amp: Path = Path("./bin_with_case_amp"),
    full_root_del: Path = Path("./bin_with_case_del"),
    out_root_amp: Optional[Path] = Path("./bin_with_case_amp"),
    out_root_del: Optional[Path] = Path("./bin_with_case_del"),
    chrom_list: Iterable = tuple(range(1, 22 + 1)),
    file_exts: Iterable[str] = (".tsv", ".txt"),
):
    """
    High-level entry function that encapsulates the original `main()` logic.

    Parameters
    ----------
    infer_dir : Path
        Directory where inference_results_*.tsv are located.
    mut : {'amp', 'del', None}, optional
        If given, only process inference files whose mut matches this.
        If None, do not filter by mut.
    cancer_types : Iterable[str] or None, optional
        If given, only process inference files whose cancer type is in this collection.
        Values should match the "{ct}" part in inference_results_{mut}_{ct}.tsv.
        If None, do not filter by cancer type.
    full_root_amp : Path
        Root directory for original per-chromosome TSVs for amp.
    full_root_del : Path
        Root directory for original per-chromosome TSVs for del.
    out_root_amp : Path or None
        Output root directory for amp results. If None, defaults to full_root_amp.
    out_root_del : Path or None
        Output root directory for del results. If None, defaults to full_root_del.
    chrom_list : Iterable
        List/iterable of chromosomes to process (e.g., range(1, 23)).
    file_exts : Iterable[str]
        Allowed file extensions for original per-chromosome TSV files.
    """
    global CHROM_LIST, FILE_EXTS

    # Set globals so that helper functions can stay unchanged
    CHROM_LIST = list(chrom_list)
    FILE_EXTS = list(file_exts)

    if out_root_amp is None:
        out_root_amp = full_root_amp
    if out_root_del is None:
        out_root_del = full_root_del

    # Collect candidate inference result files
    infer_files = sorted(infer_dir.glob("inference_results_*.tsv"))

    if not infer_files:
        print(f"[INFO] No inference_results_*.tsv found in {infer_dir}")
        return

    # Apply manual filtering if provided
    selected = []
    for f in infer_files:
        parsed = parse_infer_filename(f)
        if parsed is None:
            continue
        mut_f, ct_f = parsed
        if mut is not None and mut_f != mut:
            continue
        if cancer_types is not None and ct_f not in cancer_types:
            continue
        selected.append(f)

    # If no filter is given, process all files; if filter yields nothing, print info
    files_to_process = selected if (mut is not None or cancer_types is not None) else infer_files
    if not files_to_process:
        print("[INFO] No inference result files match the given filter conditions.")
        return

    print(f"[INFO] Will process {len(files_to_process)} file(s):")
    for f in files_to_process:
        print(" -", f.name)

    # Process files one by one
    for infer_file in files_to_process:
        process_infer_file(
            infer_file,
            full_root_amp, full_root_del,
            out_root_amp, out_root_del
        )


def main():
    """
    Keep a simple main that calls run_integration with the previous default settings.
    You can modify or remove this if you prefer to call run_integration() from elsewhere.
    """
    run_integration(
        infer_dir=Path("./buffer/inference_buffer"),
        mut="amp",
        cancer_types="EXAMPLE",
        full_root_amp=Path("./bin_with_case_amp"),
        full_root_del=Path("./bin_with_case_del"),
        out_root_amp=Path("./bin_with_case_amp"),
        out_root_del=Path("./bin_with_case_del"),
        chrom_list=list(range(1, 22 + 1)),
        file_exts=[".tsv", ".txt"],
    )


if __name__ == "__main__":
    main()
