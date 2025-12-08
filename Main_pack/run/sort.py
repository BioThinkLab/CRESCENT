#!/usr/bin/env python3
import os
import pandas as pd


def load_arm_dict(arm_file):
    """
    读取染色体臂信息，并构造 arm_dict:
    arm_dict[chrom] = {
        "p": {"start": ..., "end": ..., "length": ...},
        "q": {"start": ..., "end": ..., "length": ...}
    }
    """
    arms_df = pd.read_csv(arm_file, sep="\t")

    arms_df.rename(columns={
        "start0": "p_arm_start",
        "end0": "p_arm_end",
        "start1": "q_arm_start",
        "end1": "q_arm_end"
    }, inplace=True)

    arms_df["p_arm_length"] = arms_df["p_arm_end"] - arms_df["p_arm_start"]
    arms_df["q_arm_length"] = arms_df["q_arm_end"] - arms_df["q_arm_start"]

    arm_dict = {}
    for _, row in arms_df.iterrows():
        chrom = row["chromosome"]
        arm_dict[chrom] = {
            "p": {
                "start": row["p_arm_start"],
                "end": row["p_arm_end"],
                "length": row["p_arm_length"],
            },
            "q": {
                "start": row["q_arm_start"],
                "end": row["q_arm_end"],
                "length": row["q_arm_length"],
            }
        }
    return arm_dict


def get_all_cancer_types(input_root):
    """
    在 input_root 下找所有子目录，子目录名即为 cancer_type
    """
    if not os.path.exists(input_root):
        raise FileNotFoundError(f"输入根目录不存在: {input_root}")

    cancer_types = []
    for name in os.listdir(input_root):
        path = os.path.join(input_root, name)
        if os.path.isdir(path):
            cancer_types.append(name)

    cancer_types.sort()
    return cancer_types


def process_one_cancer_type(cancer_type, arm_dict, input_root, output_root, k):
    """
    对单个 cancer_type：
      1. 读取该目录下所有输入文件并拼接
      2. 做 arm-level 标注
      3. 将结果输出到一个总文件中
    """
    input_dir = os.path.join(input_root, cancer_type)
    output_dir = os.path.join(output_root, cancer_type)

    if not os.path.exists(input_dir):
        print(f"[WARN] 输入目录不存在: {input_dir}，跳过 {cancer_type}")
        return

    # 收集该 cancer_type 目录下的所有输入文件（这里默认用 .tsv / .txt）
    file_paths = []
    for fname in os.listdir(input_dir):
        fpath = os.path.join(input_dir, fname)
        if os.path.isfile(fpath) and (fname.endswith(".tsv") or fname.endswith(".txt")):
            file_paths.append(fpath)

    if not file_paths:
        print(f"[WARN] {input_dir} 下没有找到任何 .tsv / .txt 文件，跳过 {cancer_type}")
        return

    # ---------- 1. 读取并拼接所有 CNV 文件 ----------
    dfs = []
    for f in file_paths:
        try:
            df = pd.read_csv(f, sep="\t")
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] 读取文件失败，已跳过: {f}，错误: {e}")

    if not dfs:
        print(f"[WARN] {cancer_type} 没有成功读取的输入文件，跳过")
        return

    cnv_df = pd.concat(dfs, ignore_index=True)

    # 去除不需要的列
    for col in ["Major_Copy_Number", "Minor_Copy_Number"]:
        if col in cnv_df.columns:
            cnv_df.drop(columns=[col], inplace=True)

    # 新列
    cnv_df["is_arm_level"] = False
    cnv_df["segment_length"] = 0
    cnv_df["arm_range"] = "N/A"
    cnv_df["arm_length"] = "N/A"

    # ---------- 2. arm-level 判定 ----------
    for idx, row in cnv_df.iterrows():
        chrom = row["Chromosome"]
        seg_start = float(row["Start"])
        seg_end = float(row["End"])
        segment_length = seg_end - seg_start

        is_arm = False
        arm_range = "N/A"
        arm_length_val = "N/A"

        if chrom in arm_dict:
            fully_in = False
            intersect = False

            for arm in ["p", "q"]:
                arm_info = arm_dict[chrom][arm]
                a_start = arm_info["start"]
                a_end = arm_info["end"]

                # 完全落在单个臂内
                if seg_start >= a_start and seg_end <= a_end:
                    fully_in = True
                    arm_range = f"{a_start} - {a_end}"
                    arm_length_val = arm_info["length"]

                    if segment_length >= k * arm_info["length"]:
                        is_arm = True
                    break

                # 与臂部分有交集
                elif (seg_end > a_start) and (seg_start < a_end):
                    intersect = True

            # 跨臂（跨着丝粒）
            if not fully_in and intersect:
                is_arm = True
                arm_range = "-1"
                arm_length_val = "-1"

        cnv_df.at[idx, "segment_length"] = segment_length
        cnv_df.at[idx, "arm_range"] = arm_range
        cnv_df.at[idx, "arm_length"] = arm_length_val
        cnv_df.at[idx, "is_arm_level"] = is_arm

    # ---------- 3. 输出到一个总文件 ----------
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"cnv_arm_level_{cancer_type}.tsv")
    cnv_df.to_csv(out_file, sep="\t", index=False)

    print(f"[INFO] {cancer_type} 处理完成 → {out_file}")


def run_all_cancer_types(
    arm_file="../Data/GRCh38_Chromosome_Arm_Ranges.tsv",
    input_root="../Data/input",
    output_root="./output/sorted",
    k=0.3,
):
    """
    自动扫描所有 cancer_type 目录，执行：
      - 读取该目录下所有 .tsv / .txt 输入文件并拼接
      - 计算 arm-level 标注
      - 输出一个总文件 cnv_arm_level_{cancer_type}.tsv
    """
    arm_dict = load_arm_dict(arm_file)
    cancer_types = get_all_cancer_types(input_root)

    print("[INFO] 检测到的 cancer_type:", cancer_types)

    for ct in cancer_types:
        print(f"[INFO] 开始处理 {ct} ...")
        process_one_cancer_type(
            cancer_type=ct,
            arm_dict=arm_dict,
            input_root=input_root,
            output_root=output_root,
            k=k,
        )


if __name__ == "__main__":
    run_all_cancer_types()
