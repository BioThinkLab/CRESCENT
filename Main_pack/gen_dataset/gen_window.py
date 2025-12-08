# last modified 0429
import traceback
import os
import glob
import numpy as np
import pandas as pd
import shutil
import multiprocessing as mp


# =============================================================================
# 配置区
# =============================================================================
cancer_type = "BLCA"
CONFIG = {
    "IS_TEST": False,
    "CANCER":cancer_type,
    "DATA_BASE_DIR": "../bin_with_case_amp_compressed/",
    "FILE_EXTENSION": ".txt",
    "FEATURE_COL_START": 3,
    "FIXED_FEATURE_COLS": 40,
    "FEATURE_COL_END": 3 + 40,
    "SCALE_INPUT_SHAPES": [(50, 40), (200, 40), (1000, 40)],

    # "SCALE_INPUT_SHAPES": [(50, 40), (250, 40), (800, 40)],
    "HAS_GLOBAL": False,
    "APPLY_ROW_NORMALIZATION": False,
    # 注意：下面的数值均为“百万”为单位，例如150.5实际代表150500000
    "TYPE_CENTERS": {
        "BRCA": {
            "chr1": {
                "pos": [72.3235,73.13059,],
                "neg": []
            },
            "chr2": {
                "pos": [89.033, 97.199, ],
                "neg": []
            },
        },
        "GBM": {
            "chr12":{
             "pos": [],
             "neg": [36.08574, 37.48287]
            },
            "chr1": {
                "pos": [],
                "neg": [131.8334]
            }
            # "chr11": {
            #     "pos": [79.50035,79.50193,79.50228,79.57244,79.58682,79.6129],
            #     "neg": [79.51601,79.53267,79.5523]
            # },
            # "chr5": {
            #     "pos": [140.25],
            #     "neg": []
            # },
            # "chr6": {
            #     "pos": [32.55, 31.13,122.36],
            #     "neg": []
            # },
         },
    },
    # "OUTPUT_DIR": "/workspace/xuzheng/pyc_workspace/GeneratedSamples_buffer_after_zscore",
    # "OUTPUT_DIR": "/workspace/xuzheng/pyc_workspace/GeneratedSamples_buffer_test",
    "OUTPUT_DIR": f"../GeneratedSamples_buffer_compressed_{cancer_type}",

    # 新增参数，指定当矩阵列数不足时使用哪种填充方式
    # "interp": 线性插值，"mean": 填充每行均值
    "FILL_METHOD": "mean"  # 可修改为 "mean"
}

if CONFIG["IS_TEST"]:
    CONFIG["OUTPUT_DIR"] = "../GeneratedSamples_buffer_test"

# 清空输出目录
if os.path.exists(CONFIG["OUTPUT_DIR"]):
    shutil.rmtree(CONFIG["OUTPUT_DIR"])
os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)


def extract_matrix_from_feature_data(
    feature_data, center_idx, H, W,
    norm_method='col',       # None, 'col', or 'matrix'
    outlier_threshold=10.0   # float, e.g. 3.0 for clipping z-scores
):
    """
    从 feature_data 中以 center_idx 为中心，提取 H 行、W 列数据；
    - 如果原始行数 < H，则先将 feature_data 在行方向上按比例压缩到 2000 行。
    - 边界处直接取顶/底部 H 行，再沿行方向循环滚动，使 center_idx 对应行到达中间位置。
    - 列数少于 W 时：先计算“去最大值后”的每列均值，再取整体均值，生成常数列补齐；
    - 列数多于 W 时：按列均值排序后，依次将多余列合并到前 W 列，再求均值。
    - norm_method: None（不标准化），'col'（按列 z-score），'matrix'（全矩阵 z-score）。
    - outlier_threshold: 若提供，则对标准化后的数据按 [-threshold, threshold] 进行裁剪。
    """
    # 确保为浮点类型
    feature_data = feature_data.astype(float)

    # 1. 预先标准化并处理离群值
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

    # 离群值裁剪
    if outlier_threshold is not None:
        feature_data = np.clip(feature_data, -outlier_threshold, outlier_threshold)

    # 原始参数
    num_rows, num_cols = feature_data.shape
    half = H // 2

    # 如果原始行数不足 H，先插值压缩到 2000 行
    if num_rows < H:
        target_rows = 2000
        new_idx = np.linspace(0, num_rows - 1, num=target_rows)
        compressed = np.zeros((target_rows, num_cols), dtype=float)
        for j in range(num_cols):
            compressed[:, j] = np.interp(new_idx, np.arange(num_rows), feature_data[:, j])
        feature_data = compressed
        num_rows = target_rows

    # 行方向：中心提取 + 边界滚动
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

    # 列排序：按列均值降序
    col_means = np.mean(mat, axis=0)
    sorted_idx = np.argsort(-col_means)
    mat = mat[:, sorted_idx]

    # 列数不符时的填充或合并截断
    cur_W = mat.shape[1]
    if cur_W < W:
        # 原来的补齐常数列逻辑
        col_means_no_max = []
        for j in range(cur_W):
            col = mat[:, j]
            col_means_no_max.append((col.sum() - col.max()) / (len(col) - 1))
        final_mean = float(np.mean(col_means_no_max))
        missing = W - cur_W
        pad = np.full((H, missing), final_mean, dtype=mat.dtype)
        mat = np.concatenate([mat, pad], axis=1)

    elif cur_W > W:
        # 新增：把多余列合并到前 W 列，然后分别除以对应的计数
        combined = np.zeros((H, W), dtype=mat.dtype)
        counts   = np.zeros(W,   dtype=int)

        # 对每一列 i，累加到桶 (i % W)
        for i in range(cur_W):
            target = i % W
            combined[:, target] += mat[:, i]
            counts[target]    += 1

        # 除以各自的列数，得到平均值
        for j in range(W):
            combined[:, j] /= counts[j]

        mat = combined

    # 此时 mat 的形状已固定为 H×W
    return mat



def write_sample_tsv(filepath, scale_matrices, label):
    """
    将多个矩阵写入同一个文件，每个矩阵写成一个 section，
    文件开头写入 label 信息。
    """
    with open(filepath, "w") as f:
        f.write(f"label: {label}\n")
        for i, mat in enumerate(scale_matrices):
            H, W = mat.shape
            f.write(f"scale: {i}, shape: {H}x{W}\n")
            for row in mat:
                f.write("\t".join(map(str, row)) + "\n")
            f.write("\n")
    print(f"写入: {filepath}")


import numpy as np

import os


def process_one_chromosome(type_name, chrom, df_chrom):
    """
    针对输入文件的每一行，以该行的起始位置为中心，抽取多尺度特征矩阵并写出样本。
    返回 None，也会在内部写文件并更新进度文件，支持断点续跑。
    """
    # 配置与路径
    output_dir   = CONFIG["OUTPUT_DIR"]
    feat_start   = CONFIG["FEATURE_COL_START"]
    feat_end     = CONFIG["FEATURE_COL_END"]
    scale_shapes = CONFIG["SCALE_INPUT_SHAPES"]
    progress_path = os.path.join(output_dir, f".{type_name}_progress")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 对每一行都当作中心点来抽样
    for idx, row in df_chrom.iterrows():
        center_pos = float(row.iloc[0])   # 第一列是 start 坐标（bp）
        scale_matrices = []
        skip = False

        # 多尺度切片
        for H, W in scale_shapes:
            try:
                mat = extract_matrix_from_feature_data(
                    df_chrom.iloc[:, feat_start:feat_end].values,
                    idx, H, W
                )
            except Exception:
                skip = True
                break

            # 如果尺寸不符，也跳过
            if mat.shape != (H, W):
                skip = True
                break

            scale_matrices.append(mat)

        # 如果任一尺度失败，则跳过该行
        if skip:
            continue

        # 写出样本文件
        label = 1
        # 如果 df_chrom 中有 orig_idx 列，就用它，否则用 idx
        origin_row = int(row.get('orig_idx', idx)) + 1
        fn = f"{type_name}_{chrom}_{int(center_pos)}_{label}_{origin_row}.tsv"
        write_sample_tsv(
            os.path.join(output_dir, fn),
            scale_matrices,
            label
        )

    # 处理完本染色体后，记录进度
    with open(progress_path, "a") as pf:
        pf.write(f"{chrom}\n")

import os
import glob
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

def generate_samples_auto_parallel(type_name, chrom_list=None):
    """
    并行生成样本，支持只处理指定染色体。

    :param type_name: 数据类型名称，对应子目录名
    :param chrom_list: 要处理的染色体列表（如 ['1','2','X']），传 None 则处理所有染色体
    """
    base_dir = CONFIG["DATA_BASE_DIR"]
    output_dir = CONFIG["OUTPUT_DIR"]
    type_dir = os.path.join(base_dir, type_name)
    file_list = glob.glob(os.path.join(type_dir, "*" + CONFIG["FILE_EXTENSION"]))

    # ———— 合并各文件按染色体分组 ————
    chrom_data = {}
    for fp in file_list:
        df = pd.read_csv(fp, sep="\t", dtype={'chr': str})
        df['origin_file'] = os.path.basename(fp)
        df['orig_idx'] = df.index
        for chrom in df['chr'].unique():
            chrom_data.setdefault(chrom, []).append(df[df['chr'] == chrom])
    # 拼接同一染色体的所有 DataFrame
    for chrom in list(chrom_data.keys()):
        chrom_data[chrom] = pd.concat(chrom_data[chrom], ignore_index=True)

    # ———— 过滤，只保留指定染色体 ————
    if chrom_list is not None:
        # 将传入的列表/集合标准化为字符串类型
        wanted = set(map(str, chrom_list))
        # 取交集，忽略用户指定但不存在的数据
        available = set(chrom_data.keys())
        to_process = available & wanted
        if not to_process:
            raise ValueError(f"没有找到指定的染色体：{chrom_list}，可用染色体：{sorted(available)}")
        # 重新构建字典
        chrom_data = {c: chrom_data[c] for c in to_process}

    # ———— 并行执行处理 ————
    max_workers = CONFIG.get("MAX_WORKERS", 20)
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = []
        for chrom, dfs in chrom_data.items():
            futures.append(
                exe.submit(process_one_chromosome, type_name, chrom, dfs)
            )
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"子进程出错: {e}")

    # ———— 清理进度文件 ————
    progress_path = os.path.join(output_dir, f".{type_name}_progress")
    if os.path.exists(progress_path):
        os.remove(progress_path)






def generate_samples():
    base_dir = CONFIG["DATA_BASE_DIR"]
    file_ext = CONFIG["FILE_EXTENSION"]
    feat_start, feat_end = CONFIG["FEATURE_COL_START"], CONFIG["FEATURE_COL_END"]
    scale_shapes = CONFIG["SCALE_INPUT_SHAPES"]
    output_dir = CONFIG["OUTPUT_DIR"]

    for type_name, chrom_dict in CONFIG["TYPE_CENTERS"].items():
        print(f"处理 type: {type_name}")
        type_dir = os.path.join(base_dir, type_name)
        file_list = glob.glob(os.path.join(type_dir, "*" + file_ext))

        if not file_list:
            print(f"警告: {type_dir} 无数据文件。")
            continue

        # 收集所有染色体数据
        chrom_data = {}
        for filepath in file_list:
            try:
                df = pd.read_csv(filepath, sep="\t")
            except Exception as e:
                print(f"读取文件 {filepath} 时出错：", e)
                continue
            for chrom in df["chr"].unique():
                chrom_data.setdefault(chrom, []).append(df)
        for chrom in chrom_data:
            chrom_data[chrom] = pd.concat(chrom_data[chrom], ignore_index=True)

        # 遍历每个染色体
        for chrom, centers in chrom_dict.items():
            if chrom not in chrom_data:
                print(f"跳过: {type_name} 无 {chrom} 数据")
                continue
            df_chrom = chrom_data[chrom]
            try:
                df_chrom["start"] = pd.to_numeric(df_chrom.iloc[:, 0], errors="coerce")
            except Exception as e:
                print(f"转换 {chrom} 的起始位置时出错：", e)
                continue
            starts = df_chrom.iloc[:, 0].values.astype(float)
            feature_data = df_chrom.iloc[:, feat_start:feat_end].astype(float).values


            # 处理正例和负例模板
            for label, key in ((1, 'pos'), (0, 'neg')):
                for target_center in centers.get(key, []):
                    scaled_center = float(target_center) * 1e6
                    idx = np.argmin(np.abs(starts - scaled_center))
                    scale_matrices = []

                    # 多尺度提取
                    for (H, W) in scale_shapes:
                        try:
                            mat = extract_matrix_from_feature_data(feature_data, idx, H, W)
                        except Exception as e:
                            print(f"跳过 {type_name} {chrom} {'正' if label==1 else '负'}例 {target_center}：提取出错 -> {e}")
                            skip = True
                            break
                        if mat.shape != (H, W):
                            print(f"跳过 {type_name} {chrom} {'正' if label==1 else '负'}例 {target_center}：尺寸 {mat.shape} 不符 ({H},{W})")
                            skip = True
                            break
                        scale_matrices.append(mat)

                    filename = f"{type_name}_{chrom}_{int(scaled_center)}_{'pos' if label==1 else 'neg'}.tsv"
                    otd="/workspace/xuzheng/pyc_workspace/GeneratedSamples_buffer_test"
                    write_sample_tsv(os.path.join(otd, filename), scale_matrices, label)

if __name__ == "__main__":
    if CONFIG["IS_TEST"]:
        generate_samples()  # 手动
    else:
        generate_samples_auto_parallel(CONFIG["CANCER"])


    # 执行逻辑：
    # gen_sample_framework -> GeneratedSamples_buffer
    # check -> output_buffer/inference.txt
    # move inference to local
    # mappingA ->

    # 手动检查方法：
    # 调用generate_samples()，check.py然后手动检车inference.txt