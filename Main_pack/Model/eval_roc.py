import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_curve

from Model.legacy.Model_del import ComplexMultiStreamCNN
from utils.Dataset import TSVDataset
from utils.evaluate import evaluate_detailed  # 返回 val_loss, acc, prec, rec, f1, auc, cm, gt, probs, best_threshold

# --------------------- 配置参数 ---------------------
DATA_DIR   = "/workspace/xuzheng/pyc_workspace/GeneratedSamples_del2/"
BASE_DIR   = f"/workspace/xuzheng/pyc_workspace/experimental/final/del_result/"
BATCH_SIZE = 32
# --------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def multi_branch_collate(batch):
    if not batch:
        return None
    num_branches = len(batch[0][0])
    branch_batches = [
        torch.stack([sample[0][j] for sample in batch], dim=0)
        for j in range(num_branches)
    ]
    labels = torch.tensor([sample[1] for sample in batch], dtype=torch.long)
    file_names = [sample[2] for sample in batch]
    return branch_batches, labels, file_names

# --------------------- 加载全量数据集并检查 ---------------------
full_dataset = TSVDataset(DATA_DIR)
if len(full_dataset) == 0:
    raise RuntimeError(f"No samples found under DATA_DIR={DATA_DIR}. 请检查路径是否正确。")

# --------------------- 扫描子目录，只保留全大写字母名称 ---------------------
cancer_types = [
    d for d in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, d)) and d.isupper() and d.isalpha()
]
if not cancer_types:
    raise RuntimeError(f"No uppercase-letter-only subdirectories found under {BASE_DIR}")
print("Found cancer types:", cancer_types)

# --------------------- 对每个 cancer_type 循环处理 ---------------------
for CANCER_TYPE in cancer_types:
    print(f"\n==== Processing {CANCER_TYPE} ====")
    model_dir = os.path.join(BASE_DIR, CANCER_TYPE)
    model_path = os.path.join(model_dir, "best_model.pth")
    if not os.path.isfile(model_path):
        print(f"[WARN] {model_path} 不存在，跳过")
        continue

    # 构建仅含本 cancer_type 且非增强样本的测试集索引
    test_indices = [
        i for i, (_, _, fn) in enumerate(full_dataset)
        if fn.split('_')[0] == CANCER_TYPE and '_aug_' not in fn
    ]
    if not test_indices:
        print(f"[WARN] 在 {DATA_DIR} 中没有找到 '{CANCER_TYPE}' 的原始样本，跳过")
        continue

    # 动态获取 input_shapes
    example_input, _, _ = full_dataset[test_indices[0]]
    input_shapes = [(t.shape[1], t.shape[2]) for t in example_input]

    test_loader = DataLoader(
        Subset(full_dataset, test_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        collate_fn=multi_branch_collate
    )

    # 初始化并载入模型
    model = ComplexMultiStreamCNN(input_shapes).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 评估并收集 gt/probs
    _, _, _, _, _, _, _, gt_list, prob_list, _ = evaluate_detailed(
        model, test_loader,
        criterion=torch.nn.CrossEntropyLoss(),
        device=device
    )
    gt    = np.array(gt_list)
    probs = np.array(prob_list)

    # 计算 ROC 曲线数据
    fpr, tpr, thresholds = roc_curve(gt, probs)
    df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})

    # 在本 cancer_type 目录下创建 roc_results 子文件夹
    local_output_dir = os.path.join(model_dir, "roc_results")
    os.makedirs(local_output_dir, exist_ok=True)

    # 保存 CSV
    csv_path = os.path.join(local_output_dir, f"roc_{CANCER_TYPE}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # 保存 TSV
    tsv_path = os.path.join(local_output_dir, f"roc_{CANCER_TYPE}.tsv")
    df.to_csv(tsv_path, sep='\t', index=False)
    print(f"Saved TSV: {tsv_path}")

    # 保存 NPZ
    npz_path = os.path.join(local_output_dir, f"roc_{CANCER_TYPE}.npz")
    np.savez(npz_path, fpr=fpr, tpr=tpr, thresholds=thresholds)
    print(f"Saved NPZ: {npz_path}")

print("\nAll done.")


