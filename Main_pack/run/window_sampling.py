import shutil


# =============================================================================
# Configuration Section
# =============================================================================

CONFIG = {
    "IS_TEST": False,
    "DATA_BASE_DIR": "./bin_with_case_amp",
    "FILE_EXTENSION": ".txt",
    "FEATURE_COL_START": 3,
    "FIXED_FEATURE_COLS": 40,
    "FEATURE_COL_END": 3 + 40,
    "SCALE_INPUT_SHAPES": [(50, 40), (200, 40), (1000, 40)],
    # "SCALE_INPUT_SHAPES": [(50, 40), (250, 40), (800, 40)],
    "HAS_GLOBAL": False,
    "APPLY_ROW_NORMALIZATION": False,
    "OUTPUT_DIR": "./buffer/instance",
    # New parameter: specify how to pad when column number is insufficient
    # "interp": linear interpolation, "mean": fill with row-wise mean
    "FILL_METHOD": "mean"
}


def extract_matrix_from_feature_data(
    feature_data, center_idx, H, W,
    norm_method='col',       # None, 'col', or 'matrix'
    outlier_threshold=10.0   # float, e.g. 3.0 for clipping z-scores
):
    """
    Extract an H×W matrix centered at center_idx from feature_data.
    - If the original row number < H, first compress feature_data to 2000 rows by interpolation.
    - At boundaries, directly take the top/bottom H rows and roll along the row axis
      so that center_idx is aligned to the center.
    - If column number < W: compute per-column mean after removing max value, then
      take the global mean and pad with constant columns.
    - If column number > W: sort by column means, then merge extra columns into the
      first W columns and average them.
    - norm_method: None (no normalization), 'col' (column-wise z-score), 'matrix' (global z-score).
    - outlier_threshold: clip standardized values into [-threshold, threshold] if provided.
    """
    # Ensure floating point type
    feature_data = feature_data.astype(float)

    # 1. Pre-normalization and outlier handling
    if norm_method == 'col':
        means = np.mean(feature_data, axis=0)
        stds  = np.std(feature_data, axis=0)
        stds[stds == 0] = 1.0
        feature_data = (feature_data - means) / stds
    elif norm_method == 'matrix':
        mean_all = np.mean(feature_data)
        std_all  = np.std(feature_data)
        std_all  = std_all if std_all != 0 else 1.0
        feature_data = (feature_data - mean_all) / std_all

    # Outlier clipping
    if outlier_threshold is not None:
        feature_data = np.clip(feature_data, -outlier_threshold, outlier_threshold)

    # Original parameters
    num_rows, num_cols = feature_data.shape
    half = H // 2

    # If original rows < H, compress to 2000 rows by interpolation
    if num_rows < H:
        target_rows = 2000
        new_idx = np.linspace(0, num_rows - 1, num=target_rows)
        compressed = np.zeros((target_rows, num_cols), dtype=float)
        for j in range(num_cols):
            compressed[:, j] = np.interp(new_idx, np.arange(num_rows), feature_data[:, j])
        feature_data = compressed
        num_rows = target_rows

    # Row direction: center extraction + boundary rolling
    if 0 <= center_idx - half and center_idx + half < num_rows:
        start = center_idx - half
        mat = feature_data[start:start + H]
    else:
        if center_idx < half:
            mat = feature_data[0:H]
            orig_pos = center_idx
        else:
            mat = feature_data[num_rows - H:num_rows]
            orig_pos = center_idx - (num_rows - H)
        shift = half - orig_pos
        mat = np.roll(mat, shift=shift, axis=0)

    # Column sorting by descending column mean
    col_means = np.mean(mat, axis=0)
    sorted_idx = np.argsort(-col_means)
    mat = mat[:, sorted_idx]

    # Column padding or merging
    cur_W = mat.shape[1]
    if cur_W < W:
        # Original constant column padding logic
        col_means_no_max = []
        for j in range(cur_W):
            col = mat[:, j]
            col_means_no_max.append((col.sum() - col.max()) / (len(col) - 1))
        final_mean = float(np.mean(col_means_no_max))
        missing = W - cur_W
        pad = np.full((H, missing), final_mean, dtype=mat.dtype)
        mat = np.concatenate([mat, pad], axis=1)

    elif cur_W > W:
        # Merge extra columns into the first W columns, then average
        combined = np.zeros((H, W), dtype=mat.dtype)
        counts   = np.zeros(W,   dtype=int)

        # Accumulate each column i into bucket (i % W)
        for i in range(cur_W):
            target = i % W
            combined[:, target] += mat[:, i]
            counts[target]    += 1

        # Divide by column counts to get mean
        for j in range(W):
            combined[:, j] /= counts[j]

        mat = combined

    # Final shape is fixed as H×W
    return mat



def write_sample_tsv(filepath, scale_matrices, label):
    """
    Write multiple matrices into one file, each matrix as a section.
    The file starts with label information.
    """
    # Write atomically so an interrupted/full-disk write cannot leave a
    # malformed .tsv that the inference loader later tries to parse.
    temp_filepath = filepath + ".tmp"
    try:
        with open(temp_filepath, "w") as f:
            f.write(f"label: {label}\n")
            for i, mat in enumerate(scale_matrices):
                H, W = mat.shape
                f.write(f"scale: {i}, shape: {H}x{W}\n")
                for row in mat:
                    f.write("\t".join(map(str, row)) + "\n")
                f.write("\n")
        os.replace(temp_filepath, filepath)
    except Exception:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        raise


import numpy as np
import os


def process_one_chromosome(
    type_name,
    chrom,
    df_chrom,
    row_start=0,
    row_end=None,
    write_progress=True,
    scale_shapes=None,
):
    """
    For each row in the input file, use the row's start position as center,
    extract multi-scale feature matrices and write samples.
    Returns None, writes files internally and updates the progress file,
    supporting checkpoint continuation.
    """
    # Configuration and paths
    output_dir   = CONFIG["OUTPUT_DIR"]
    feat_start   = CONFIG["FEATURE_COL_START"]
    feat_end     = CONFIG["FEATURE_COL_END"]
    scale_shapes = scale_shapes or CONFIG["SCALE_INPUT_SHAPES"]
    progress_path = os.path.join(output_dir, f".{type_name}_progress")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Treat each row as a sampling center
    selected_rows = df_chrom.iloc[row_start:row_end]
    for idx, row in selected_rows.iterrows():
        center_pos = float(row.iloc[0])   # First column is start coordinate (bp)
        scale_matrices = []
        skip = False

        # Multi-scale slicing
        for H, W in scale_shapes:
            try:
                mat = extract_matrix_from_feature_data(
                    df_chrom.iloc[:, feat_start:feat_end].values,
                    idx, H, W
                )
            except Exception:
                skip = True
                break

            # Skip if shape mismatch
            if mat.shape != (H, W):
                skip = True
                break

            scale_matrices.append(mat)

        # Skip this row if any scale fails
        if skip:
            continue

        # Write sample file
        label = 1
        # Use orig_idx if exists, otherwise use idx
        origin_row = int(row.get('orig_idx', idx)) + 1
        fn = f"{type_name}_{chrom}_{int(center_pos)}_{label}_{origin_row}.tsv"
        write_sample_tsv(
            os.path.join(output_dir, fn),
            scale_matrices,
            label
        )

    # After finishing this chromosome, record progress
    if write_progress:
        with open(progress_path, "a") as pf:
            pf.write(f"{chrom}\n")

import os
import glob
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed


def _clear_sample_output(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for name in os.listdir(output_dir):
        if name == ".gitkeep":
            continue
        path = os.path.join(output_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def _normalize_chrom(value):
    value = str(value).strip()
    if value.lower().startswith("chr"):
        value = value[3:]
    return value.upper()


def _load_chromosome_data(type_name, chromosome, base_dir=None):
    base_dir = base_dir or CONFIG["DATA_BASE_DIR"]
    type_dir = os.path.join(base_dir, type_name)
    file_list = glob.glob(os.path.join(type_dir, "*" + CONFIG["FILE_EXTENSION"]))
    wanted = _normalize_chrom(chromosome)
    frames = []
    output_chromosome = None

    for filepath in file_list:
        frame = pd.read_csv(filepath, sep="\t", dtype={"chr": str})
        frame["origin_file"] = os.path.basename(filepath)
        frame["orig_idx"] = frame.index
        for chrom in frame["chr"].unique():
            if _normalize_chrom(chrom) == wanted:
                frames.append(frame[frame["chr"] == chrom])
                output_chromosome = chrom

    if not frames:
        raise ValueError(f"Chromosome {chromosome} not found in {type_dir}")
    return output_chromosome, pd.concat(frames, ignore_index=True)


def generate_sample_chunks(
    type_name, chromosome, chunk_size=500, base_dir=None, scale_shapes=None
):
    """Generate bounded sample chunks and yield metadata for each completed chunk."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    output_dir = CONFIG["OUTPUT_DIR"]
    chrom, frame = _load_chromosome_data(type_name, chromosome, base_dir=base_dir)
    total_rows = len(frame)
    total_chunks = (total_rows + chunk_size - 1) // chunk_size

    for chunk_index, start in enumerate(range(0, total_rows, chunk_size), start=1):
        end = min(start + chunk_size, total_rows)
        _clear_sample_output(output_dir)
        process_one_chromosome(
            type_name,
            chrom,
            frame,
            row_start=start,
            row_end=end,
            write_progress=False,
            scale_shapes=scale_shapes,
        )
        yield chunk_index, total_chunks, end - start

def generate_samples_auto_parallel(
    type_name, chrom_list=None, base_dir=None, scale_shapes=None
):
    """
    Generate samples in parallel, optionally only processing specified chromosomes.

    :param type_name: data type name, corresponding to subdirectory name
    :param chrom_list: list of chromosomes to process (e.g. ['1','2','X']),
                       process all chromosomes if None
    """
    base_dir = base_dir or CONFIG["DATA_BASE_DIR"]
    output_dir = CONFIG["OUTPUT_DIR"]
    _clear_sample_output(output_dir)

    type_dir = os.path.join(base_dir, type_name)
    file_list = glob.glob(os.path.join(type_dir, "*" + CONFIG["FILE_EXTENSION"]))

    wanted = None if chrom_list is None else {_normalize_chrom(c) for c in chrom_list}

    # ———— Merge files by chromosome ————
    chrom_data = {}
    for fp in file_list:
        df = pd.read_csv(fp, sep="\t", dtype={'chr': str})
        df['origin_file'] = os.path.basename(fp)
        df['orig_idx'] = df.index
        for chrom in df['chr'].unique():
            if wanted is not None and _normalize_chrom(chrom) not in wanted:
                continue
            chrom_data.setdefault(chrom, []).append(df[df['chr'] == chrom])

    # Concatenate DataFrames of the same chromosome
    for chrom in list(chrom_data.keys()):
        chrom_data[chrom] = pd.concat(chrom_data[chrom], ignore_index=True)

    # ———— Filter chromosomes if specified ————
    if chrom_list is not None and not chrom_data:
        raise ValueError(f"Specified chromosomes not found: {chrom_list}")

    # ———— Parallel execution ————
    max_workers = CONFIG.get("MAX_WORKERS", 20)
    errors = []
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = []
        for chrom, dfs in chrom_data.items():
            futures.append(
                exe.submit(
                    process_one_chromosome,
                    type_name,
                    chrom,
                    dfs,
                    scale_shapes=scale_shapes,
                )
            )
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                errors.append(e)

    if errors:
        raise RuntimeError(
            f"Sample generation failed in {len(errors)} worker(s); "
            f"first error: {errors[0]}"
        )

    # ———— Clean up progress file ————
    progress_path = os.path.join(output_dir, f".{type_name}_progress")
    if os.path.exists(progress_path):
        os.remove(progress_path)

    print("finished")


def generate_samples():
    base_dir = CONFIG["DATA_BASE_DIR"]
    file_ext = CONFIG["FILE_EXTENSION"]
    feat_start, feat_end = CONFIG["FEATURE_COL_START"], CONFIG["FEATURE_COL_END"]
    scale_shapes = CONFIG["SCALE_INPUT_SHAPES"]
    output_dir = CONFIG["OUTPUT_DIR"]

    for type_name, chrom_dict in CONFIG["TYPE_CENTERS"].items():
        print(f"Processing type: {type_name}")
        type_dir = os.path.join(base_dir, type_name)
        file_list = glob.glob(os.path.join(type_dir, "*" + file_ext))

        if not file_list:
            print(f"Warning: no data files in {type_dir}.")
            continue

        # Collect all chromosome data
        chrom_data = {}
        for filepath in file_list:
            try:
                df = pd.read_csv(filepath, sep="\t")
            except Exception as e:
                print(f"Error reading file {filepath}:", e)
                continue
            for chrom in df["chr"].unique():
                chrom_data.setdefault(chrom, []).append(df)

        for chrom in chrom_data:
            chrom_data[chrom] = pd.concat(chrom_data[chrom], ignore_index=True)

        # Iterate through each chromosome
        for chrom, centers in chrom_dict.items():
            if chrom not in chrom_data:
                print(f"Skipping: {type_name} has no data for {chrom}")
                continue
            df_chrom = chrom_data[chrom]
            try:
                df_chrom["start"] = pd.to_numeric(df_chrom.iloc[:, 0], errors="coerce")
            except Exception as e:
                print(f"Error converting start positions for {chrom}:", e)
                continue

            starts = df_chrom.iloc[:, 0].values.astype(float)
            feature_data = df_chrom.iloc[:, feat_start:feat_end].astype(float).values

            # Process positive and negative templates
            for label, key in ((1, 'pos'), (0, 'neg')):
                for target_center in centers.get(key, []):
                    scaled_center = float(target_center) * 1e6
                    idx = np.argmin(np.abs(starts - scaled_center))
                    scale_matrices = []

                    # Multi-scale extraction
                    for (H, W) in scale_shapes:
                        try:
                            mat = extract_matrix_from_feature_data(feature_data, idx, H, W)
                        except Exception as e:
                            print(f"Skipping {type_name} {chrom} {'pos' if label==1 else 'neg'} {target_center}: extraction error -> {e}")
                            skip = True
                            break

                        if mat.shape != (H, W):
                            print(f"Skipping {type_name} {chrom} {'pos' if label==1 else 'neg'} {target_center}: shape {mat.shape} mismatch ({H},{W})")
                            skip = True
                            break

                        scale_matrices.append(mat)

                    filename = f"{type_name}_{chrom}_{int(scaled_center)}_{'pos' if label==1 else 'neg'}.tsv"
                    otd="/workspace/xuzheng/pyc_workspace/GeneratedSamples_buffer_test"
                    write_sample_tsv(os.path.join(otd, filename), scale_matrices, label)

if __name__ == "__main__":
    cancer="BLCA"
    generate_samples_auto_parallel(cancer)
