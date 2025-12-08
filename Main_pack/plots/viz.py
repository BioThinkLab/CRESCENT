#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
from pathlib import Path
import csv

# ====== 写死输入文件路径（最后一个为 C 基准）======
cancer = "BRCA"
files = [
    f"/Users/sanjati/jangoTemp/temp2/pycharmD/shin_Data/{cancer}/amp/{cancer}_gistic_amp.tsv",
    f"/Users/sanjati/jangoTemp/temp2/pycharmD/shin_Data/{cancer}/amp/{cancer}_rubic_amp.tsv",
    f"/Users/sanjati/jangoTemp/temp2/pycharmD/shin_Data/{cancer}/amp/{cancer}_om_amp.tsv"  # C
]
labels = [Path(p).stem for p in files]
out_csv = Path.cwd() / "per_chrom_overlap_vs_C.csv"   # 输出CSV路径

# ====== 基础工具 ======
def read_intervals(tsv_path):
    """读取单个TSV，返回 dict[chrom] -> list[(s,e)]"""
    chrom_intervals = defaultdict(list)
    with open(tsv_path, "r") as f:
        rows = [r.strip().split() for r in f if r.strip()]
    if not rows:
        return chrom_intervals
    header = [c.lower() for c in rows[0]]
    # 允许大小写不敏感，列名必须包含 Chromosome/Start/End
    try:
        idx_chr = header.index("chromosome")
        idx_s   = header.index("start")
        idx_e   = header.index("end")
    except ValueError:
        raise ValueError(f"{tsv_path} 的表头需包含 Chromosome/Start/End，实际为：{header}")

    for row in rows[1:]:
        if len(row) <= max(idx_chr, idx_s, idx_e):
            continue
        chrom = row[idx_chr]
        try:
            s, e = int(row[idx_s]), int(row[idx_e])
        except Exception:
            continue
        if s == e:
            continue
        if s > e:
            s, e = e, s
        chrom_intervals[chrom].append((s, e))
    return chrom_intervals

def merge_overlaps(intervals):
    """合并重叠区间，返回不重叠的 [(s,e)]（半开）"""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = []
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    return merged

def total_length(merged_intervals):
    """已合并区间的总长度"""
    return sum(e - s for s, e in merged_intervals)

def intersect_length(ints1, ints2):
    """两个已合并且有序的区间集合的交集长度"""
    i, j, tot = 0, 0, 0
    while i < len(ints1) and j < len(ints2):
        a1, a2 = ints1[i]
        b1, b2 = ints2[j]
        start, end = max(a1, b1), min(a2, b2)
        if start < end:
            tot += end - start
        if a2 < b2:
            i += 1
        else:
            j += 1
    return tot

# ====== 主逻辑：按染色体统计 Fi 与 C 的交集占比 ======
def main():
    datasets = [read_intervals(p) for p in files]
    C = datasets[-1]

    # 需要遍历的染色体集合：出现于任一（前置Fi或C）的并集
    chroms = set()
    for d in datasets:
        chroms.update(d.keys())
    chroms = sorted(chroms)

    # 预先合并每个文件每条染色体的区间，便于复用
    merged = []
    for d in datasets:
        per = {chrom: merge_overlaps(d.get(chrom, [])) for chrom in chroms}
        merged.append(per)
    merged_C = merged[-1]

    # 构建CSV表头：Chromosome + 对每个Fi的三列
    header = ["Chromosome"]
    for i in range(len(files) - 1):
        name = labels[i]
        header += [f"{name}_total", f"{name}_intersect_{labels[-1]}", f"{name}_ratio_vs_{labels[-1]}"]

    rows = []

    # 按染色体统计
    for chrom in chroms:
        row = [chrom]
        for i in range(len(files) - 1):
            mi = merged[i][chrom]
            mc = merged_C.get(chrom, [])
            cov = total_length(mi)
            inter = intersect_length(mi, mc) if (mi and mc) else 0
            ratio = (inter / cov) if cov > 0 else 0.0
            row += [cov, inter, f"{ratio:.6f}"]
        rows.append(row)

    # 再加一行 ALL 汇总
    all_row = ["ALL"]
    for i in range(len(files) - 1):
        # 汇总 cov & inter 跨染色体累加（注意交集要逐染色体算后累加）
        cov_sum = 0
        inter_sum = 0
        for chrom in chroms:
            mi = merged[i][chrom]
            mc = merged_C.get(chrom, [])
            cov_sum += total_length(mi)
            inter_sum += intersect_length(mi, mc) if (mi and mc) else 0
        ratio_all = (inter_sum / cov_sum) if cov_sum > 0 else 0.0
        all_row += [cov_sum, inter_sum, f"{ratio_all:.6f}"]
    rows.append(all_row)

    # 输出CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"# Base (C) file: {labels[-1]}"])
        writer.writerow(header)
        writer.writerows(rows)

    # 同时在终端打印一个精简版预览
    print(f"基准(C)：{labels[-1]}")
    print("—— 每染色体统计 ——")
    for row in rows[:-1]:
        print(row)
    print("—— 汇总 ALL ——")
    print(rows[-1])
    print(f"\nCSV 已写入: {out_csv}")

if __name__ == "__main__":
    main()
