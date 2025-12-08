import traceback
import os
import numpy as np
import pandas as pd

def extract_matrix_from_feature_data(feature_data, center_idx, H, W):
    """
    从 feature_data 中以 center_idx 为索引，提取 H 行数据，
    并按列求和排序。
    """
    num_rows = feature_data.shape[0]
    half = H // 2

    start_idx = max(center_idx - half, 0)
    end_idx = min(start_idx + H, num_rows)

    if end_idx - start_idx < H:
        start_idx = max(end_idx - H, 0)

    try:
        mat = feature_data[start_idx:end_idx]
    except Exception as e:
        print("Error extracting matrix:", e)
        traceback.print_exc()
        raise

    col_sums = np.sum(mat, axis=0)
    sorted_indices = np.argsort(-col_sums)
    sorted_mat = mat[:, sorted_indices]

    if sorted_mat.shape[1] >= W:
        sorted_mat = sorted_mat[:, :W]
    else:
        print(f"警告：提取的矩阵列数 {sorted_mat.shape[1]} 少于预期 {W} 列。")

    return sorted_mat

def write_sample_tsv(scale_matrices, label):
    """将提取的特征矩阵写入固定文件 /workspace/xuzheng/pyc_workspace/Model/utils/input_buffer.tsv"""
    output_filepath = "/workspace/xuzheng/pyc_workspace/Model/utils/input_buffer.tsv"
    with open(output_filepath, "w") as f:
        f.write(f"label: {label}\n")
        for i, mat in enumerate(scale_matrices):
            H, W = mat.shape
            f.write(f"scale: {i}, shape: {H}x{W}\n")
            for row in mat:
                f.write("\t".join(map(str, row)) + "\n")
            f.write("\n")
    print(f"写入: {output_filepath}")

def compress_matrix(mat, target_rows=8000):
    """
    将输入矩阵 mat（形状为 [原始高度, 列数]）通过线性插值压缩（或拉伸）到 target_rows 行，
    保持列数不变。
    """
    original_rows = mat.shape[0]
    new_indices = np.linspace(0, original_rows - 1, num=target_rows)
    new_mat = np.zeros((target_rows, mat.shape[1]))
    for col in range(mat.shape[1]):
        new_mat[:, col] = np.interp(new_indices, np.arange(original_rows), mat[:, col])
    return new_mat

def process_single_file(file_path, center, range, label=-1):
    """
    处理单个文件，以 center 作为 Start 的索引，提取不同尺度的矩阵，并返回一个指定范围的 chr/start/end 子 DataFrame。

    参数:
        file_path: 输入文件路径（TSV）
        center: 特征矩阵中心索引
        range: 范围字符串，如 'a-b'，用于截取 chr/start/end 的 DataFrame
        label: 标签（写入 input_buffer.tsv 时用）

    返回:
        None ， DataFrame，包含指定行范围的 chr/start/end 列
    """
    CONFIG = {
        "FEATURE_COL_START": 3,
        "FEATURE_COL_END": 3 + 40,
        "SCALE_INPUT_SHAPES":  [(200, 40), (700, 40), (1500, 40)],
    }

    try:
        df = pd.read_csv(file_path, sep="\t")
    except Exception as e:
        print(f"读取文件 {file_path} 时出错：", e)
        return None, None

    try:
        df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    except Exception as e:
        print("转换起始位置时出错：", e)
        return None, None

    starts = df.iloc[:, 0].values.astype(float)
    feature_data = df.iloc[:, CONFIG["FEATURE_COL_START"]:CONFIG["FEATURE_COL_END"]].astype(float).values

    center_idx = int(center)
    if center_idx < 0 or center_idx >= len(starts):
        print(f"错误: center 索引 {center_idx} 超出范围 (0, {len(starts) - 1})")
        return None, None

    scale_matrices = []
    for (H, W) in CONFIG["SCALE_INPUT_SHAPES"]:
        mat = extract_matrix_from_feature_data(feature_data, center_idx, H, W)
        if mat.shape != (H, W):
            print(f"错误：索引 {center_idx} 对应的矩阵尺寸为 {mat.shape}，预期为 ({H}, {W})")
        scale_matrices.append(mat)

    # 新增逻辑：从全局提取一个矩阵，并通过平移和线性插值压缩到8000行，追加到结果中
    full_mat = feature_data[:, :20]
    N = full_mat.shape[0]
    shift_val = (N // 2) - center_idx
    shifted_mat = np.roll(full_mat, shift=shift_val, axis=0)
    compressed_mat = compress_matrix(shifted_mat, target_rows=8000)
    scale_matrices.append(compressed_mat)

    write_sample_tsv(scale_matrices, label)

    # 处理 range 字符串，提取 chr/start/end 子 DataFrame
    try:
        a_str, b_str = range.strip().split("-")
        a, b = int(a_str), int(b_str)
        # 保证索引不越界
        b = min(b, len(df) - 1)
        sub_df = df.loc[a:b, ["chr", "start", "end"]].reset_index(drop=True)
    except Exception as e:
        print(f"解析 range 参数 '{range}' 时出错：", e)
        sub_df = None

    return None, sub_df
