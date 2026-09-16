import csv

import os
import pandas as pd

BASE_INPUT_DIR = './bin_with_case_amp/'
OUTPUT_DIR = './bin_with_case_amp/'


K = 40  # 保留均值最高的前 K 列
# ==============================

def detect_sep(file_path: str) -> str:
    """
    自动识别文件分隔符：
    读取前 2048 字节，用 csv.Sniffer 来猜分隔符，fallback 到 '\t'
    """
    with open(file_path, 'r', newline='') as f:
        sample = f.read(2048)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters='\t,; ')
        return dialect.delimiter
    except csv.Error:
        return '\t'


def process_file(file_path: str, output_path: str, k: int, sep: str = '\t') -> None:
    """
    处理单个文件（支持 .txt/.tsv 任意分隔符）：
      1) 按 sep 读取
      2) 提取前3列（meta）、最后1列（suffix），中间当做数值列
      3) 如中间列不足 k，则补齐全为 2 的列
      4) 计算均值，选出前 k 列，剩余列循环分配给这 k 列并取平均
      5) 输出结构同原来，但中间只留 k 列，分隔符固定为 '\t'
    """
    df = pd.read_csv(file_path, sep=sep)

    # 1. 元信息列和 suffix 列
    meta_cols = df.columns[:3].tolist()
    suffix_cols = [
        col for col in df.columns[3:]
        if str(col).strip().lower() in {"sum", "prob"}
    ]

    # 2. 中间原始数据列
    data_cols = [col for col in df.columns[3:] if col not in suffix_cols]

    # 3. 补齐不足 k 列
    if len(data_cols) < k:
        num_pad = k - len(data_cols)
        pad_cols = [f'pad_{i+1}' for i in range(num_pad)]
        for col in pad_cols:
            df[col] = 2
        data_cols.extend(pad_cols)

    # 4. 计算均值并选前 k
    means    = df[data_cols].mean()
    top_cols = means.nlargest(k).index.tolist()

    # 5. 剩余列循环分配给 top_cols
    rem_cols    = [c for c in data_cols if c not in top_cols]
    assignments = {col: top_cols[i % k] for i, col in enumerate(rem_cols)}

    # 6. 按组合并并取平均
    merged_cols = {}
    for top in top_cols:
        group = [top] + [c for c, tgt in assignments.items() if tgt == top]
        merged_cols[top] = df[group].sum(axis=1) / len(group)

    # 7. 组装输出
    df_out = pd.concat([
        df[meta_cols],
        pd.DataFrame(merged_cols),
        df[suffix_cols]
    ], axis=1)
    df_out = df_out[meta_cols + top_cols + suffix_cols]

    # 8. 写入 TSV（固定 '\t' ）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, sep='\t', index=False)


def compress(base_input_dir=None, output_dir=None, include_root=False):
    VALID_EXTS = ('.txt', '.tsv')
    input_root = os.path.abspath(base_input_dir or BASE_INPUT_DIR)
    output_root = os.path.abspath(output_dir or OUTPUT_DIR)

    for root, dirs, files in os.walk(input_root):
        # 如果也想处理 BASE_INPUT_DIR 本身，就去掉这行判断
        if not include_root and os.path.abspath(root) == input_root:
            continue

        rel_dir = os.path.relpath(root, input_root)
        out_dir = os.path.join(output_root, rel_dir)

        for fname in files:
            if not fname.lower().endswith(VALID_EXTS):
                continue

            in_path  = os.path.join(root, fname)
            out_path = os.path.join(out_dir, fname)

            sep = detect_sep(in_path)
            process_file(in_path, out_path, K, sep=sep)
            print(f"Processed {in_path} (detected sep={repr(sep)}) -> {out_path}")

