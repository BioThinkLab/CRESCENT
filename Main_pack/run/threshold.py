#!/usr/bin/env python3
import re

import os
import glob
import numpy as np
import pandas as pd


def detect_regions_threshold(
    path,
    threshold,
    min_length=3,
    top_frac=None,             # None => 不做二次筛选
    select_ascending=False,    # False: 降序取前 top_frac；True: 升序取前 top_frac
):
    """
    对单个 TSV/TXT 文件：
      1) 读取全部列，将列名转为小写；
      2) 自动识别 'chromosome' 或 'chr'，'start'，'end' 以及 'prob' 列；
      3) 按概率阈值 threshold 构造 mask，提取所有连续片段（长度 ≥ min_length）；
      4) 若 top_frac 为非 None 且 >0，则对每个片段内按“数据列（end 与 prob 之间）”逐行求和 row_sum：
         - select_ascending=False：保留 row_sum 降序排序前 top_frac 比例；
         - select_ascending=True ：保留 row_sum 升序排序前 top_frac 比例；
         每段至少保留 1 行；
      5) 返回列 ['Chromosome','Start','End'] 的 DataFrame。
    """
    if threshold is None:
        raise ValueError("必须提供 threshold 参数（若仅想要固定阈值结果，设置 top_frac=None 即可）。")

    # 1) 读文件
    df = pd.read_csv(path, sep='\t', header=0, dtype=str)

    # 2) 列名标准化：转小写
    df.columns = [c.lower() for c in df.columns]

    # 自动识别各列
    if 'chromosome' in df.columns:
        chrom_col = 'chromosome'
    elif 'chr' in df.columns:
        chrom_col = 'chr'
    else:
        raise ValueError(f"文件 {path} 中未找到 'chromosome' 或 'chr' 列")

    if 'start' in df.columns:
        start_col = 'start'
    else:
        raise ValueError(f"文件 {path} 中未找到 'start' 列")
    if 'end' in df.columns:
        end_col = 'end'
    else:
        raise ValueError(f"文件 {path} 中未找到 'end' 列")

    if 'prob' in df.columns:
        prob_col = 'prob'
    else:
        raise ValueError(f"文件 {path} 中未找到 'prob' 列")

    # —— 识别“数据列”：严格按列顺序在 end 与 prob 之间的列（不含端点） ——
    cols = list(df.columns)
    i_end = cols.index(end_col)
    i_prob = cols.index(prob_col)
    lo = min(i_end, i_prob) + 1
    hi = max(i_end, i_prob)
    data_cols = cols[lo:hi]  # 端点外开区间

    # —— 基础数据 ——
    df[chrom_col] = df[chrom_col].astype(str)
    probs = pd.to_numeric(df[prob_col], errors='coerce').fillna(0.0).values
    L = len(probs)

    # 根据阈值构造 mask
    mask = probs > float(threshold)

    # 提取连续片段
    regions = []
    cur = None
    for i, m in enumerate(mask):
        if m and cur is None:
            cur = i
        if cur is not None and ((not m) or i == L - 1):
            j = i if m else i - 1
            if j - cur + 1 >= int(min_length):
                regions.append((cur, j))
            cur = None

    if not regions:
        return pd.DataFrame([], columns=['Chromosome', 'Start', 'End'])

    # 是否启用二次筛选
    apply_second_filter = (
        top_frac is not None and float(top_frac) > 0 and len(data_cols) > 0
    )

    if apply_second_filter:
        df_data = df[data_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        row_sum = df_data.sum(axis=1).values
    else:
        row_sum = None  # 不用

    keep_frames = []
    for s, e in regions:
        sub = df.iloc[s:e+1].copy()

        if apply_second_filter:
            sub_sum = row_sum[s:e+1]
            n = len(sub)
            keep_n = max(1, int(np.ceil(n * float(top_frac))))
            order = np.argsort(sub_sum) if select_ascending else np.argsort(-sub_sum)
            top_idx = order[:keep_n]
            sub = sub.iloc[top_idx].copy()
            # 可选：保持按 Start 升序输出（更直观）
            sub[start_col] = pd.to_numeric(sub[start_col], errors='coerce')
            sub = sub.sort_values(start_col)

        # 仅输出标准列
        out = sub[[chrom_col, start_col, end_col]].copy()
        out.columns = ['Chromosome', 'Start', 'End']
        out['Start'] = pd.to_numeric(out['Start'], errors='coerce').fillna(0).astype(int)
        out['End']   = pd.to_numeric(out['End'],   errors='coerce').fillna(0).astype(int)
        keep_frames.append(out)

    return pd.concat(keep_frames, ignore_index=True)


def merge_regions(df):
    """
    合并同 Chromosome 且 End == 下一区段 Start 的行。
    假设输入 df 已按 ['Chromosome','Start'] 排序。
    """
    if df.empty:
        return df

    merged = []
    for chrom, group in df.groupby('Chromosome', sort=False):
        group = group.sort_values('Start').reset_index(drop=True)
        cs, ce = group.loc[0, ['Start', 'End']]
        for idx in range(1, len(group)):
            s, e = group.loc[idx, ['Start', 'End']]
            if ce == s:
                ce = e
            else:
                merged.append({'Chromosome': chrom, 'Start': cs, 'End': ce})
                cs, ce = s, e
        merged.append({'Chromosome': chrom, 'Start': cs, 'End': ce})

    merged_df = pd.DataFrame(merged, columns=['Chromosome', 'Start', 'End'])
    merged_df['Start'] = merged_df['Start'].astype(int)
    merged_df['End']   = merged_df['End'].astype(int)
    return merged_df


def to_segment(cancer_type, mut_type, threshold=0.5, peak=None):


    data_dir = (
        f"./bin_with_case_{mut_type}/{cancer_type}"
    )
    out_path = (
        f"result/{cancer_type}/{mut_type}/{cancer_type}_om_{mut_type}.tsv"
    )

    min_length  = 1

    # —— 不同 mut_type 使用不同的 top_frac（k）与选择方向 ——
    # 将某个 mut_type 的值设为 None 即可关闭二次筛选（仅输出阈值连通段结果）
    if peak is None:
        top_frac_by_mut = {
            "amp": 0.8,   # amp：段内 row_sum【降序】取前k倍
            "del": 0.7,   # del：段内 row_sum【升序】取前k倍（改成 None 则不做二次筛选）
        }
    else:
        top_frac_by_mut = {
            "amp": peak,   # amp：段内 row_sum【降序】取前k倍
            "del": peak,   # del：段内 row_sum【升序】取前k倍（改成 None 则不做二次筛选）
        }
    select_ascending_by_mut = {
        "amp": False,  # amp 降序（大优先）
        "del": True,   # del 升序（小优先）
    }

    mut_key = str(mut_type).lower()
    top_frac = top_frac_by_mut.get(mut_key, None)  # 允许为 None
    select_ascending = select_ascending_by_mut.get(mut_key, False)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"目录不存在: {data_dir}")

    # 同时匹配 .tsv 和 .txt 文件，并滤掉性染色体文件（文件名含 X 或 Y）
    txt_files  = glob.glob(os.path.join(data_dir, "*.txt"))
    tsv_files  = glob.glob(os.path.join(data_dir, "*.tsv"))
    input_files = txt_files + tsv_files
    print(len(input_files))
    # —— 剔除文件名包含 chrX/chrY（大小写不敏感），或仅为大写 X/Y 的文件 ——
    def _exclude_xy(fname: str) -> bool:
        base = os.path.basename(fname)
        base_upper = base.upper()
        # 匹配 chrX / chrY (不区分大小写)
        if re.search(r'CHR[XY]', base_upper):
            return False
        # 仅为 X 或 Y 的文件名（不带扩展名的部分）
        stem, _ = os.path.splitext(base)
        if stem in ('X', 'Y'):
            return False
        return True

    input_files = [f for f in input_files if _exclude_xy(f)]
    if not input_files:
        print("未在目录中找到任何 .tsv 或 .txt 文件（或全部被性染色体过滤）。")
        print(f"数据源目录:{data_dir}")
        print(f"总数据文件数:{len(txt_files)},{len(tsv_files)},{len(input_files)}")
        return

    all_regions = []
    for infile in input_files:
        regs = detect_regions_threshold(
            infile,
            threshold=threshold,               # 必填；若仅要阈值结果，请把 top_frac_by_mut[mut_type] 设为 None
            min_length=min_length,
            top_frac=top_frac,
            select_ascending=select_ascending,
        )
        if not regs.empty:
            all_regions.append(regs)

    if not all_regions:
        print("所有文件经处理后均无符合条件的片段。")
        return

    # 合并所有结果
    result_df = pd.concat(all_regions, ignore_index=True)
    result_df = result_df.sort_values(['Chromosome', 'Start']).reset_index(drop=True)
    merged_df = merge_regions(result_df)

    # 写出最终结果
    merged_df.to_csv(out_path, sep='\t', index=False)
    print(f"[mut_type={mut_type}, threshold={threshold}, "
          f"top_frac={top_frac}, ascending={select_ascending}] "
          f"已将处理后的峰区域写入: {out_path}")


if __name__ == "__main__":
    main()
