import os
import glob
import pandas as pd
import argparse


def process_file(file_path, output_dir):
    """
    处理单个文件：
      - 读取文件（假设以制表符分隔）
      - 保留 is_arm_level 为 False 的行
      - 仅保留 Chromosome, Start, End, Copy_Number 四列
      - 输出到指定目录，文件名保持不变
    """
    try:
        # 读取文件，假定第一行为列头，文件以制表符分隔
        df = pd.read_csv(file_path, sep="\t")
    except Exception as e:
        print(f"读取文件 {file_path} 出错: {e}")
        return

    # 筛选 is_arm_level 为 False 的行（注意文件中可能为字符串形式 "False"）
    df_filtered = df[df['is_arm_level'].astype(str).str.lower() == "false"]

    # 保留所需的列，如果文件中不存在则报错
    required_cols = ['Chromosome', 'Start', 'End', 'Copy_Number']
    try:
        df_filtered = df_filtered[required_cols]
    except KeyError as e:
        print(f"文件 {file_path} 中缺少必要的列: {e}")
        return

    # 输出文件到指定目录，保持原有文件名
    output_path = os.path.join(output_dir, os.path.basename(file_path))
    try:
        df_filtered.to_csv(output_path, sep="\t", index=False)
        print(f"已处理文件 {file_path} -> {output_path}")
    except Exception as e:
        print(f"保存文件 {output_path} 出错: {e}")


def main(output_dir):
    # 如果输出目录不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 搜索当前目录下所有 .txt 和 .tsv 文件（根据需要可以调整）
    files = glob.glob("*.txt") + glob.glob("*.tsv")
    if not files:
        print("未在当前目录下找到匹配的文件。")
        return

    for file_path in files:
        process_file(file_path, output_dir)


if __name__ == "__main__":
    main("/legacy/Data/cluster_2025/BRCA/focal")
