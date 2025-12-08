import traceback
import os
import glob
import numpy as np
import pandas as pd
import shutil
# =============================================================================
# 配置区（除 TYPE_CENTERS 部分，其他配置仍保留）
# =============================================================================
depth=40
CONFIG = {
    "DATA_BASE_DIR": "/workspace/xuzheng/pyc_workspace/preprocess/Version0209/output/bin_with_case/",
    "FILE_EXTENSION": ".txt",
    "FEATURE_COL_START": 3,
    "FIXED_FEATURE_COLS": depth,
    "FEATURE_COL_END": 3 + depth,
    "SCALE_INPUT_SHAPES": [(100, depth), (500, depth), (2000, depth)],
    "TYPE_CENTERS": {
        # 该部分在本函数中不再使用
    },
    "OUTPUT_DIR": "/Users/sanjati/jangoTemp/temp2/pycharmD/GeneratedSamples"
}


def extract_matrix_from_feature_data(feature_data, center_idx, H, W):
    """
    从 feature_data 中以 center_idx 为中心，提取 H 行数据；
    同时对提取的矩阵按列求和排序。
    若提取的矩阵列数大于 W，则截取前 W 列；不足时则发出警告。
    """
    num_rows = feature_data.shape[0]
    half = H // 2

    # 计算起始和结束索引
    start_idx = center_idx - half
    end_idx = start_idx + H

    if start_idx < 0:
        start_idx = 0
        end_idx = min(H, num_rows)
    if end_idx > num_rows:
        end_idx = num_rows
        start_idx = max(end_idx - H, 0)

    try:
        rows = [feature_data[j] for j in range(start_idx, end_idx)]
        mat = np.stack(rows, axis=0)
    except Exception as e:
        print("Error stacking rows:", e)
        traceback.print_exc()
        raise

    # 按列求和后降序排序
    col_sums = np.sum(mat, axis=0)
    sorted_indices = np.argsort(-col_sums)
    sorted_mat = mat[:, sorted_indices]

    if sorted_mat.shape[1] >= W:
        sorted_mat = sorted_mat[:, :W]
    else:
        print(f"警告：提取的矩阵列数 {sorted_mat.shape[1]} 少于预期 {W} 列。")

    return sorted_mat


import pandas as pd
import numpy as np
import traceback

def process_file(file_path, center, label=-1):
    """
    处理指定的 TSV 文件 file_path，根据 center（单位为碱基对）定位中心位置，
    按照配置中指定的 SCALE_INPUT_SHAPES 提取不同尺度的矩阵，
    生成多列 DataFrame，而不是一列中包含制表符。
    """
    feat_start = CONFIG["FEATURE_COL_START"]
    feat_end = CONFIG["FEATURE_COL_END"]
    scale_shapes = CONFIG["SCALE_INPUT_SHAPES"]

    try:
        df = pd.read_csv(file_path, sep="\t")
    except Exception as e:
        print(f"读取文件 {file_path} 时出错：", e)
        return None

    try:
        df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    except Exception as e:
        print("转换起始位置时出错：", e)
        return None

    try:
        starts = df.iloc[:, 0].values.astype(float)
        feature_data = df.iloc[:, feat_start:feat_end].astype(float).values
    except Exception as e:
        print("提取数据时出错：", e)
        return None

    scaled_center = float(center)
    idx = np.argmin(np.abs(starts - scaled_center))

    output_lines = []
    output_lines.append(f"label:\t{label}")  # 这里用 `\t` 使其与多列格式一致

    for i, (H, W) in enumerate(scale_shapes):
        try:
            mat = extract_matrix_from_feature_data(feature_data, idx, H, W)
            if mat.shape != (H, W):
                print(f"错误：中心 {scaled_center} 对应的矩阵尺寸为 {mat.shape}，预期为 ({H}, {W})")
        except Exception as e:
            print(f"处理尺度 {H}x{W} 时出错：", e)
            traceback.print_exc()
            continue

        output_lines.append(f"scale:\t{i}\tshape:\t{H}x{W}")  # 保证以制表符分列
        output_lines.extend(["\t".join(map(str, row)) for row in mat])  # 直接加入矩阵行
        output_lines.append("")  # 空行分隔不同尺度

    # **转换为多列 DataFrame**
    result_df = pd.DataFrame([line.split("\t") for line in output_lines])

    return result_df
# 示例调用：
if __name__ == "__main__":
    # 假设提供文件路径和中心位置（单位：百万）
    file_path = "/path/to/your/file.tsv"
    center = 1000000  # 示例中心值
    result = process_file(file_path, center)
    if result is not None:
        print(result.head())