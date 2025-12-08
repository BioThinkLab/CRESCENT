#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_heatmap.py

功能：从之前生成的样本 TSV 文件中加载数据（每个样本含若干尺度的矩阵），
     然后为每个尺度绘制热力图。
     根据文件内容首行的 label 来将输出结果保存到目标目录下的 pos 和 neg 子目录中，
     原输出不再保留。
"""

import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def load_sample_file(filepath):
    """
    从单个样本 TSV 文件中加载数据。
    文件格式：
      第一行: "label: {label}"
      对于每个尺度：
         一行: "scale: {i}, shape: {H}x{W}"
         接下来 H 行: 数值，以制表符分隔
         空行分隔不同尺度

    返回: (scale_matrices, label, sample_id)
         其中 scale_matrices 为一个列表，每个元素是 torch.Tensor，尺寸为 (1, H, W)
         label 为 0 或 1
         sample_id 为文件名（或其他标识）
    """
    scale_matrices = []
    label = None
    sample_id = os.path.basename(filepath)

    with open(filepath, "r") as f:
        lines = f.readlines()

    idx = 0
    # 第一行获取 label
    if idx < len(lines):
        line = lines[idx].strip()
        if line.startswith("label:"):
            label_str = line.split(":", 1)[1].strip()
            label = int(label_str)
        idx += 1

    # 后续读取各尺度的数据
    while idx < len(lines):
        line = lines[idx].strip()
        # 跳过空行
        if not line:
            idx += 1
            continue

        # line 形如 "scale: 0, shape: 5x20"
        if line.startswith("scale:"):
            # 解析形状
            parts = line.split(",")
            shape_part = parts[1].split(":")[1].strip()  # "5x20"
            H_str, W_str = shape_part.split("x")
            H = int(H_str)
            W = int(W_str)

            idx += 1
            matrix_rows = []
            for _ in range(H):
                row_vals = lines[idx].strip().split("\t")
                row = [float(x) for x in row_vals]
                matrix_rows.append(row)
                idx += 1

            mat = np.stack(matrix_rows, axis=0)  # (H, W)
            # 转为 (1, H, W) 的张量
            tensor_mat = torch.tensor(mat, dtype=torch.float32).unsqueeze(0)
            scale_matrices.append(tensor_mat)
        else:
            idx += 1

    return scale_matrices, label, sample_id

def plot_heatmap_for_sample(sample_file, output_dir=None):
    """
    读取单个样本文件并绘制热力图。
    - sample_file: 样本 TSV 文件路径
    - output_dir: 若指定则将图片保存到该目录下的 pos 或 neg 子目录中，否则直接 plt.show()
    """
    scale_matrices, label, sample_id = load_sample_file(sample_file)

    # 若有多个尺度，就画多个子图
    fig, axes = plt.subplots(1, len(scale_matrices), figsize=(6 * len(scale_matrices), 6))
    if len(scale_matrices) == 1:
        axes = [axes]  # 兼容只有1个尺度时axes不是列表

    for i, mat in enumerate(scale_matrices):
        # mat 形状为 (1, H, W)，需要 squeeze(0) 成 (H, W)
        mat_2d = mat.squeeze(0).numpy()  # (H, W)
        sns.heatmap(mat_2d, ax=axes[i], cmap="viridis")
        axes[i].set_title(f"Scale {i}, shape={mat_2d.shape}")

    fig.suptitle(f"Label: {label}, Sample: {sample_id}", fontsize=14)

    if output_dir is not None:
        # **根据首行的 label 决定保存到 pos 或 neg 子目录**
        if label == 1:
            subdir = "pos"
        elif label == 0:
            subdir = "neg"
        else:
            subdir = "unknown"

        final_dir = os.path.join(output_dir, subdir)
        os.makedirs(final_dir, exist_ok=True)

        out_path = os.path.join(final_dir, f"{sample_id}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[Heatmap] 已保存到 {out_path}")
    else:
        plt.show()

def main(mut,cancer_type=None):
    """
    示例：遍历某目录下的所有 TSV 样本文件，为每个文件绘制热力图并保存，同时显示当前进度。
    """
    if mut == 'amp':
        samples_dir = f"/Users/sanjati/jangoTemp/temp2/pycharmD/GeneratedSamples_{mut}_compressed"
    elif mut == 'del':
        samples_dir = f"/Users/sanjati/jangoTemp/temp2/pycharmD/GeneratedSamples_{mut}_compressed"
    elif mut == 'sim':
        samples_dir = f"/Users/sanjati/jangoTemp/temp2/pycharmD/pre_train_dataset"
    elif mut == 'test':
        samples_dir = f"/Users/sanjati/jangoTemp/temp2/pycharmD/GeneratedSamples_{mut}"


    # 输出热力图的目录
    output_dir = f"/Users/sanjati/jangoTemp/temp2/pycharmD/HeatmapOutput_{mut}_compressed"

    os.makedirs(output_dir, exist_ok=True)

    sample_files = glob.glob(os.path.join(samples_dir, "*.tsv"))
    total_files = len(sample_files)
    print(f"在 {samples_dir} 中找到 {total_files} 个样本文件。")
    if cancer_type is not None:
        filtered = []
        for sf in sample_files:
            prefix = os.path.basename(sf).split("_")[0]
            if prefix.upper() == cancer_type.upper():
                filtered.append(sf)
        print(f"FILTER_BY_CANCER_TYPE=True，过滤只保留 cancer_type={cancer_type} 的文件，共 {len(filtered)} 个。")
        sample_files = filtered
    total_files = len(sample_files)


    # 清空输出目录（可选）
    for f in os.listdir(output_dir):
        path = os.path.join(output_dir, f)
        if os.path.isfile(path) or os.path.islink(path):
            os.unlink(path)
        elif os.path.isdir(path):
            import shutil
            shutil.rmtree(path)

    # 遍历所有样本，显示当前处理进度
    for idx, sf in enumerate(sample_files, 1):
        print(f"正在处理 {idx}/{total_files}: {os.path.basename(sf)}")
        plot_heatmap_for_sample(sf, output_dir=output_dir)

if __name__ == "__main__":
    main("amp", cancer_type="GBM")
