#!/usr/bin/env python3
"""Core orchestration logic for the CRESCENT analysis pipeline."""

from contextlib import contextmanager
from datetime import datetime
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from check import run_inf
from compress import compress
from mapping import run_integration
from sort import run_all_cancer_types
from threshold import to_segment
from window_sampling import generate_sample_chunks, generate_samples_auto_parallel


RUN_DIR = Path(__file__).resolve().parent
MAIN_DIR = RUN_DIR.parent
SRC_DIR = RUN_DIR / "src"
BUILD_DIR = RUN_DIR / "build"
PREP_DIR = RUN_DIR / "._prepared_inputs"
INPUT_DIR = MAIN_DIR / "Data" / "input"
# DEL_SCALE_SHAPES = [(20, 40), (50, 40), (400, 40)]
DEL_SCALE_SHAPES = [(50, 40), (100, 40), (500, 40)]



@contextmanager
def _runtime_directory():
    """Run legacy modules from their expected directory, then restore the caller."""
    previous = Path.cwd()
    os.chdir(RUN_DIR)
    try:
        yield
    finally:
        os.chdir(previous)


def _is_windows() -> bool:
    return os.name == "nt"


def _latest_mtime(paths: Iterable[Path]) -> float:
    latest = 0.0
    for path in paths:
        if path.is_file():
            try:
                latest = max(latest, path.stat().st_mtime)
            except FileNotFoundError:
                pass
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file():
                    try:
                        latest = max(latest, child.stat().st_mtime)
                    except FileNotFoundError:
                        pass
    return latest


def _choose_compiler() -> str:
    for compiler in ("g++", "clang++"):
        if shutil.which(compiler):
            return compiler
    raise RuntimeError("No C++ compiler found. Install g++ or clang++ and add it to PATH.")


def _compile_processor(mutation_type: str) -> Path:
    source = SRC_DIR / f"gen_bin_{mutation_type}_cpp.cpp"
    executable = BUILD_DIR / (
        f"processor_{mutation_type}.exe" if _is_windows() else f"processor_{mutation_type}"
    )
    if not source.is_file():
        raise FileNotFoundError(f"C++ source not found: {source}")

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    headers = [p for p in SRC_DIR.rglob("*") if p.suffix.lower() in {".h", ".hh", ".hpp"}]
    source_mtime = _latest_mtime([source, *headers])
    executable_mtime = executable.stat().st_mtime if executable.exists() else 0.0
    if executable.exists() and executable_mtime >= source_mtime:
        print(f"Executable is up to date: {executable}")
        return executable

    flags = ["-std=gnu++17", "-O3", "-Wall", "-Wextra", "-Wno-unused-parameter"]
    if not _is_windows():
        flags.append("-pthread")
    command = [_choose_compiler(), *flags, str(source), "-o", str(executable)]
    print("Compile command:", " ".join(command))
    completed = subprocess.run(command, cwd=RUN_DIR)
    if completed.returncode != 0:
        raise RuntimeError("C++ compilation failed; see the compiler output above.")

    if not _is_windows():
        executable.chmod(executable.stat().st_mode | 0o111)
    print(f"Compilation succeeded: {executable}")
    return executable


def _table_separator(path: Path) -> str:
    return "," if path.suffix.lower() == ".csv" else "\t"


def _is_table(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".csv", ".tsv", ".txt"}


def _normalize_input_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.shape[1] < 2:
        return frame
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.columns[0] != "GDC_Aliquot":
        frame = frame.rename(columns={frame.columns[0]: "GDC_Aliquot"})
    if "Copy_Number" not in frame.columns and "Copy Number" in frame.columns:
        frame = frame.rename(columns={"Copy Number": "Copy_Number"})
    if "Copy_Number" not in frame.columns and "Segment_Mean" in frame.columns:
        segment_mean = pd.to_numeric(frame["Segment_Mean"], errors="coerce")
        frame["Copy_Number"] = np.rint(2.0 * np.power(2.0, segment_mean)).astype("Int64")
    if "Copy_Number" in frame.columns:
        columns = [column for column in frame.columns if column != "GDC_Aliquot"]
        frame = frame[["GDC_Aliquot", *columns]]
    return frame


def _prepare_input(project_dir: Path) -> Tuple[Path, int, int]:
    prepared_dir = PREP_DIR / project_dir.name
    if prepared_dir.exists():
        shutil.rmtree(prepared_dir)
    prepared_dir.mkdir(parents=True)

    processed = 0
    modified = 0
    for source in project_dir.rglob("*"):
        target = prepared_dir / source.relative_to(project_dir)
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if not _is_table(source):
            shutil.copy2(source, target)
            continue

        separator = _table_separator(source)
        try:
            frame = pd.read_csv(source, sep=separator, dtype=str)
        except Exception:
            shutil.copy2(source, target)
            continue
        processed += 1
        original_columns = list(frame.columns)
        frame = _normalize_input_table(frame.copy())
        if original_columns != list(frame.columns):
            modified += 1
        frame.to_csv(target, sep=separator, index=False)
    return prepared_dir, processed, modified


def _run_processor(
    executable: Path,
    input_dir: Path,
    output_dir: Path,
    only_use_focal: bool,
    log_dir: Optional[Path] = None,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [str(executable), str(input_dir), str(output_dir), "1" if only_use_focal else "0"]
    print("Running:", " ".join(command))
    if log_dir is None:
        return subprocess.run(command, cwd=RUN_DIR).returncode

    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stdout_path = log_dir / f"{input_dir.name}_{timestamp}.out.txt"
    stderr_path = log_dir / f"{input_dir.name}_{timestamp}.err.txt"
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        return_code = subprocess.run(
            command, cwd=RUN_DIR, stdout=stdout, stderr=stderr
        ).returncode
    print(f"Return code {return_code}; logs: {stdout_path.name}, {stderr_path.name}")
    return return_code


def _normalize_chromosome(value: object) -> str:
    token = str(value).strip()
    if token.lower().startswith("chr"):
        token = token[3:]
    token = token.upper()
    if token == "MT":
        token = "M"
    if token.isdigit():
        token = str(int(token))
    if not re.fullmatch(r"\d+|X|Y|M", token):
        raise ValueError(f"Invalid chromosome: {value!r}")
    return token


def _chromosome_from_filename(name: str) -> Optional[str]:
    match = re.search(
        r"(?:^|[^A-Za-z0-9])chr(\d+|X|Y|M|MT)(?:[^A-Za-z0-9]|$)",
        Path(name).stem,
        flags=re.IGNORECASE,
    )
    return _normalize_chromosome(match.group(1)) if match else None


def _standardize_bin_files(output_dir: Path) -> None:
    """Rename bin tables and normalize their chromosome column."""
    if not output_dir.exists():
        return
    candidates = [
        path for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".csv", ".tsv", ".txt"}
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    used = set()
    for source in candidates:
        chromosome = _chromosome_from_filename(source.name)
        if chromosome is None or chromosome in used:
            continue
        used.add(chromosome)
        target = output_dir / f"cnv_chr{chromosome}.txt"
        try:
            with source.open("r", encoding="utf-8") as stream:
                header = stream.readline()
            separator = "\t" if "\t" in header else "," if "," in header else None
            if separator:
                frame = pd.read_csv(source, sep=separator, dtype=str)
                frame = _normalize_bin_table(frame)
                frame.to_csv(target, sep="\t", index=False)
                if target.resolve() != source.resolve():
                    source.unlink(missing_ok=True)
                continue
        except Exception:
            pass
        if target.exists() and target.resolve() != source.resolve():
            target.unlink()
        if target.resolve() != source.resolve():
            source.rename(target)


def _normalize_bin_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize AMP and DEL bin tables to start, end, chr, features."""
    frame.columns = [str(column).strip() for column in frame.columns]
    lookup = {column.lower(): column for column in frame.columns}
    chromosome_column = next(
        (
            lookup[name]
            for name in ("chr", "chromosome", "chrom", "chromosome_name")
            if name in lookup
        ),
        None,
    )
    start_column = lookup.get("start")
    end_column = lookup.get("end")
    if not chromosome_column or not start_column or not end_column:
        raise ValueError(
            f"Bin table requires chromosome, start, and end columns; found {list(frame.columns)}"
        )

    frame = frame.rename(
        columns={chromosome_column: "chr", start_column: "start", end_column: "end"}
    )
    frame["chr"] = frame["chr"].apply(
        lambda value: f"chr{_normalize_chromosome(value)}"
    )
    remaining = [column for column in frame.columns if column not in {"start", "end", "chr"}]
    return frame[["start", "end", "chr", *remaining]]


def _available_chromosomes(data_dir: Path) -> List[str]:
    chromosomes = {
        chromosome
        for path in data_dir.iterdir()
        if path.is_file() and (chromosome := _chromosome_from_filename(path.name))
    }
    return sorted(
        chromosomes,
        key=lambda chromosome: (0, int(chromosome)) if chromosome.isdigit() else (1, chromosome),
    )


def _select_chromosomes(data_dir: Path, requested: Sequence[object]) -> List[str]:
    available = _available_chromosomes(data_dir)
    if not requested:
        return available
    selected = []
    for value in requested:
        chromosome = _normalize_chromosome(value)
        if chromosome not in selected:
            selected.append(chromosome)
    missing = [chromosome for chromosome in selected if chromosome not in available]
    if missing:
        raise ValueError(
            f"Chromosomes not found for {data_dir.name}: {missing}. Available: {available}"
        )
    return selected


def _clear_buffer(directory: Path) -> None:
    """Clear generated files while retaining the directory placeholder."""
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.name == ".gitkeep":
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _disk_locations(buffer_dir: Path):
    """Return distinct filesystems used by samples and Python temporary files."""
    locations = [("sample buffer", buffer_dir), ("system temp", Path(tempfile.gettempdir()))]
    distinct = []
    seen = set()
    for label, path in locations:
        path.mkdir(parents=True, exist_ok=True)
        key = path.resolve().anchor.lower()
        if key not in seen:
            seen.add(key)
            distinct.append((label, path))
    return distinct


def _warn_about_disk_space(
    buffer_dir: Path, threshold_gb: float, context: str, force: bool = False
) -> None:
    threshold_bytes = float(threshold_gb) * 1024 ** 3
    readings = []
    for label, path in _disk_locations(buffer_dir):
        free_bytes = shutil.disk_usage(path).free
        readings.append((label, path.resolve().anchor, free_bytes / 1024 ** 3))
    low = [reading for reading in readings if reading[2] < float(threshold_gb)]
    if not low and not force:
        return

    print(f"\nWARNING: disk-space risk {context}.")
    for label, drive, free_gb in readings:
        marker = " LOW" if free_gb * 1024 ** 3 < threshold_bytes else ""
        print(f"  {label} ({drive}): {free_gb:.2f} GB free{marker}")
    print(
        f"  Warning threshold: {float(threshold_gb):.2f} GB. "
        "Execution will continue."
    )


def _looks_like_space_failure(error: BaseException) -> bool:
    current = error
    while current is not None:
        message = str(current).lower()
        if (
            getattr(current, "errno", None) == 28
            or "no space left" in message
            or "disk full" in message
            or "rows but found fewer" in message
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _validate_projects(project_names: Sequence[str], input_root: Path) -> List[str]:
    projects = []
    for name in project_names:
        clean_name = str(name).strip()
        if not clean_name or clean_name in projects:
            continue
        project_dir = input_root / clean_name
        if not project_dir.is_dir():
            raise FileNotFoundError(f"Input project directory not found: {project_dir}")
        projects.append(clean_name)
    if not projects:
        raise ValueError("PROJECT_NAMES must contain at least one project name.")
    return projects


def run_pipeline(
    project_names: Sequence[str],
    mutation_type: str = "amp",
    chromosome_numbers: Sequence[object] = (),
    only_use_focal: bool = False,
    classification_threshold: float = 0.5,
    use_chunking: bool = True,
    samples_per_chunk: int = 500,
    low_disk_warning_gb: float = 15,
) -> None:
    """Run CRESCENT for the configured projects and chromosomes."""
    mutation_type = mutation_type.lower().strip()
    if mutation_type not in {"amp", "del"}:
        raise ValueError("MUTATION_TYPE must be 'amp' or 'del'.")
    if not 0.0 <= float(classification_threshold) <= 1.0:
        raise ValueError("CLASSIFICATION_THRESHOLD must be between 0 and 1.")
    if use_chunking and int(samples_per_chunk) <= 0:
        raise ValueError("SAMPLES_PER_CHUNK must be greater than zero.")
    if float(low_disk_warning_gb) < 0:
        raise ValueError("LOW_DISK_WARNING_GB cannot be negative.")

    with _runtime_directory():
        use_focal_filter = only_use_focal and mutation_type == "amp"
        input_root = RUN_DIR / "output" / "sorted" if use_focal_filter else INPUT_DIR
        if use_focal_filter:
            run_all_cancer_types(
                arm_file=str(MAIN_DIR / "Data" / "GRCh38_Chromosome_Arm_Ranges.tsv"),
                input_root=str(INPUT_DIR),
                output_root=str(input_root),
            )

        projects = _validate_projects(project_names, input_root)
        output_root = RUN_DIR / f"bin_with_case_{mutation_type}"
        executable = _compile_processor(mutation_type)
        PREP_DIR.mkdir(parents=True, exist_ok=True)

        failures = []
        for project_name in projects:
            prepared_dir, processed, modified = _prepare_input(input_root / project_name)
            print(f"Prepared {project_name}: table_files={processed}, modified={modified}")
            project_output = output_root / project_name
            return_code = _run_processor(
                executable, prepared_dir, project_output, use_focal_filter
            )
            if return_code:
                failures.append((project_name, return_code))
            else:
                _standardize_bin_files(project_output)
        if failures:
            details = ", ".join(f"{name}={code}" for name, code in failures)
            raise RuntimeError(f"C++ processing failed: {details}")

        print("Compressing or padding bin files")
        for project_name in projects:
            project_output = output_root / project_name
            compress(
                base_input_dir=str(project_output),
                output_dir=str(project_output),
                include_root=True,
            )
        instance_dir = RUN_DIR / "buffer" / "instance"
        inference_dir = RUN_DIR / "buffer" / "inference_buffer"
        scale_shapes = DEL_SCALE_SHAPES if mutation_type == "del" else None

        for project_name in projects:
            project_output = output_root / project_name
            chromosomes = _select_chromosomes(project_output, chromosome_numbers)
            if not chromosomes:
                raise RuntimeError(f"No chromosome bin files found in {project_output}")
            print(f"{project_name}: processing chromosomes {chromosomes}")

            for chromosome in chromosomes:
                chromosome_name = f"chr{chromosome}"
                print(f"\n===== {project_name} {chromosome_name} =====")
                inference_path = inference_dir / (
                    f"inference_results_{mutation_type}_{project_name}.tsv"
                )
                inference_path.unlink(missing_ok=True)
                try:
                    if use_chunking:
                        for chunk_index, chunk_count, sample_count in generate_sample_chunks(
                            project_name,
                            chromosome_name,
                            chunk_size=int(samples_per_chunk),
                            base_dir=str(output_root),
                            scale_shapes=scale_shapes,
                        ):
                            print(
                                f"Inferring chunk {chunk_index}/{chunk_count} "
                                f"({sample_count} samples)"
                            )
                            run_inf(
                                project_name,
                                mutation_type,
                                chr_list=[chromosome_name],
                                append_results=chunk_index > 1,
                            )
                            _clear_buffer(instance_dir)
                    else:
                        _warn_about_disk_space(
                            instance_dir,
                            low_disk_warning_gb,
                            "before unchunked sample generation",
                        )
                        generate_samples_auto_parallel(
                            project_name,
                            chrom_list=[chromosome_name],
                            base_dir=str(output_root),
                            scale_shapes=scale_shapes,
                        )
                        run_inf(
                            project_name,
                            mutation_type,
                            chr_list=[chromosome_name],
                            append_results=False,
                        )
                        _clear_buffer(instance_dir)
                except Exception as error:
                    if _looks_like_space_failure(error):
                        _warn_about_disk_space(
                            instance_dir,
                            low_disk_warning_gb,
                            "after a possible disk-space failure",
                            force=True,
                        )
                    raise
                amp_root = output_root if mutation_type == "amp" else RUN_DIR / "bin_with_case_amp"
                del_root = output_root if mutation_type == "del" else RUN_DIR / "bin_with_case_del"
                run_integration(
                    mut=mutation_type,
                    cancer_types=[project_name],
                    full_root_amp=amp_root,
                    full_root_del=del_root,
                    out_root_amp=amp_root,
                    out_root_del=del_root,
                    chrom_list=[chromosome],
                )
                inference_path.unlink(missing_ok=True)
                print(f"Completed {chromosome_name}; temporary samples cleared")

            _clear_buffer(inference_dir)
            to_segment(
                cancer_type=project_name,
                mut_type=mutation_type,
                threshold=classification_threshold,
                chrom_list=chromosomes,
            )
        print("\nCRESCENT analysis completed")
