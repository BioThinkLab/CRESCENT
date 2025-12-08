import datetime
import copy
import torch
import torch.nn as nn
import random as _py_random
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_curve

from Models import ComplexMultiStreamCNN
from utils.Dataset import TSVDataset
from utils.train import train_one_epoch
from utils.evaluate import evaluate_detailed  # return:( val_loss, acc, prec, rec, f1, auc, cm, gt, probs, best_threshold)




output_dir = "../experimental/"


def get_prefix(fn: str) -> str:
    # 用来识别增强数据集
    name = os.path.splitext(fn)[0]
    return name.split('_aug_')[0]

# --------------------- 固定随机性 ---------------------
def set_seed(seed: int = 42):
    _py_random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------- 移动平均平滑函数 ---------------------
def moving_average(x, w=5):
    return np.convolve(x, np.ones(w) / w, mode='valid')

# --------------------- 数据集的 collate 函数 ---------------------
def multi_branch_collate(batch):
    if not batch:
        return None
    num_branches = len(batch[0][0])
    branch_batches = []
    for j in range(num_branches):
        branch_batches.append(torch.stack([sample[0][j] for sample in batch], dim=0))
    labels = torch.tensor([sample[1] for sample in batch], dtype=torch.long)
    file_names = [sample[2] for sample in batch]
    return branch_batches, labels, file_names


# --------------------- 训练与验证函数 ---------------------
def auto_cross_val(data_dir,
                   mut="del",
                   target_cancer_types=None,
                   seed=42):
    """
    data_dir: 样本目录
    mut: :del"或"amp",变异类型目录名
    target_cancer_types: None 或者 list of str，指定要留出的 cancer_type 列表
    """
    set_seed(seed)
    dataset = TSVDataset(data_dir)
    if len(dataset) == 0:
        print("No valid training data found！")
        return

    # input scale shape (auto loaded from data file)
    first_sample, _, _ = dataset[0]
    input_shapes = [(t.shape[1], t.shape[2]) for t in first_sample]
    print("input branch scale：", input_shapes)

    # collect 收集所有或指定的癌症类型
    all_types = set(fn.split('_')[0] for _, _, fn in dataset)
    if target_cancer_types is not None:
        cancer_types = [t for t in target_cancer_types if t in all_types]
        missing = set(target_cancer_types) - set(cancer_types)
        if missing:
            print(f"警告：以下指定的 cancer_type 在数据中未找到，将被忽略: {missing}")
    else:
        cancer_types = sorted(all_types)
    print("运行的 cancer_type 列表：", cancer_types)

    #
    num_epochs, batch_size, learning_rate = 10, 8, 1e-5  # amp:lr=1e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建会话目录
    base_dir = os.path.join(output_dir, mut)
    session_dir = os.path.join(
        base_dir,
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(seed)
    )
    os.makedirs(session_dir, exist_ok=True)
    # saving hyperamater to file
    with open(os.path.join(session_dir, "hyperparameters.txt"), "w") as f:
        f.write(f"num_epochs: {num_epochs}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"device: {device}\n")
        f.write("scheduler: ReduceLROnPlateau\n")

    # 全局 summary 文件路径
    summary_tsv = os.path.join(session_dir, "summary_metrics.tsv")
    header = [
        'cancer_type','best_epoch','best_loss','best_auc','best_f1',
        'train_size','val_size'
    ]
    if not os.path.exists(summary_tsv):
        with open(summary_tsv, 'w') as f:
            f.write("\t".join(header) + "\n")

    for val_cancer_type in cancer_types:
        print(f"\n===== 验证留出类型: {val_cancer_type} =====")
        indices = list(range(len(dataset)))
        # 留出当前类型作为验证集，其他全部作为训练集
        train_indices = [i for i in indices if dataset[i][2].split('_')[0] != val_cancer_type]
        val_indices   = [i for i in indices if dataset[i][2].split('_')[0] == val_cancer_type]

        print(f"训练集大小: {len(train_indices)}, 验证集大小: {len(val_indices)}")

        train_loader = DataLoader(
            Subset(dataset, train_indices), batch_size=batch_size,
            shuffle=True, collate_fn=multi_branch_collate,
            num_workers=1, persistent_workers=True, pin_memory=True, prefetch_factor=2
        )
        val_loader = DataLoader(
            Subset(dataset, val_indices), batch_size=batch_size,
            shuffle=False, collate_fn=multi_branch_collate,
            num_workers=1
        )

        model = ComplexMultiStreamCNN(input_shapes, num_classes=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=3e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
        )
        criterion = nn.BCEWithLogitsLoss()

        # training log
        train_loss_hist, train_acc_hist = [], []
        val_loss_hist, val_acc_hist, val_auc_hist, val_f1_hist = [], [], [], []
        best_auc, best_epoch, best_state = -float('inf'), -1, None

        # 准备可视化目录
        outs_dir = os.path.join(session_dir, val_cancer_type)
        os.makedirs(outs_dir, exist_ok=True)
        loss_fig, loss_ax = plt.subplots(figsize=(6, 4))
        acc_fig, acc_ax = plt.subplots(figsize=(6, 4))

        # 训练与验证循环
        for epoch in range(1, num_epochs+1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, *_ = \
                evaluate_detailed(model, val_loader, criterion, device)
            _, train_acc, _, _, _, _, *_ = evaluate_detailed(model, train_loader, criterion, device)

            train_loss_hist.append(train_loss)
            train_acc_hist.append(train_acc)
            val_loss_hist.append(val_loss)
            val_acc_hist.append(val_acc)
            val_auc_hist.append(val_auc)
            val_f1_hist.append(val_f1)

            print(
                f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
                f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | AUC: {val_auc:.4f} | F1: {val_f1:.4f}"
            )
            scheduler.step(val_loss)

            # 保存验证文件列表
            current_val_filenames = []
            for batch in val_loader:
                _, _, names = batch
                current_val_filenames.extend(names)
            with open(os.path.join(outs_dir, f"val_filenames_epoch_{epoch}.txt"), "w") as vf:
                vf.write("\n".join(current_val_filenames))

            # 更新最优模型
            if val_auc >= best_auc:
                best_auc, best_epoch = val_auc, epoch
                best_state = copy.deepcopy(model.state_dict())

            # 绘制并保存损失与准确率曲线
            loss_ax.clear()
            loss_ax.plot(range(1, epoch+1), train_loss_hist, label='Training Loss')
            loss_ax.plot(range(1, epoch+1), val_loss_hist, label='Validation Loss')
            loss_ax.set_xlabel('Epoch'); loss_ax.set_ylabel('Loss'); loss_ax.legend(); loss_ax.grid(True)
            loss_fig.savefig(os.path.join(outs_dir, "loss_curve_raw.png"), dpi=150)

            acc_ax.clear()
            acc_ax.plot(range(1, epoch+1), train_acc_hist, label='Training Acc')
            acc_ax.plot(range(1, epoch+1), val_acc_hist, label='Validation Acc')
            acc_ax.set_xlabel('Epoch'); acc_ax.set_ylabel('Accuracy'); acc_ax.legend(); acc_ax.grid(True)
            acc_fig.savefig(os.path.join(outs_dir, "accuracy_curve_raw.png"), dpi=150)

        # 保存每个 epoch 的指标
        epoch_tsv = os.path.join(outs_dir, "epoch_metrics.tsv")
        with open(epoch_tsv, 'w') as f:
            f.write("epoch\ttrain_loss\ttrain_acc\tval_loss\tval_acc\tval_auc\tval_f1\n")
            for e in range(num_epochs):
                f.write(
                    f"{e+1}\t{train_loss_hist[e]:.6f}\t{train_acc_hist[e]:.6f}"
                    f"\t{val_loss_hist[e]:.6f}\t{val_acc_hist[e]:.6f}"
                    f"\t{val_auc_hist[e]:.6f}\t{val_f1_hist[e]:.6f}\n"
                )

        # 平滑并保存曲线
        window = 5
        # 平滑 Loss
        train_s = moving_average(train_loss_hist, window)
        val_s   = moving_average(val_loss_hist, window)
        epochs_s = list(range(1 + (window-1)//2, num_epochs - (window-1)//2 + 1))
        plt.figure()
        plt.plot(epochs_s, train_s, label=f'Training Loss (MA{window})')
        plt.plot(epochs_s, val_s,   label=f'Validation Loss (MA{window})')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(outs_dir, "loss_curve_smoothed.png"))
        plt.close()

        # 平滑 AUC
        auc_s = moving_average(val_auc_hist, window)
        plt.figure()
        plt.plot(epochs_s, auc_s, marker='o', label=f'Val AUC (MA{window})')
        plt.xlabel('Epoch'); plt.ylabel('AUC'); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(outs_dir, "auc_curve_smoothed.png")); plt.close()

        # 平滑 Accuracy
        acc_train_s = moving_average(train_acc_hist, window)
        acc_val_s   = moving_average(val_acc_hist, window)
        plt.figure()
        plt.plot(epochs_s, acc_train_s, label=f'Train Acc (MA{window})')
        plt.plot(epochs_s, acc_val_s,   label=f'Val   Acc (MA{window})')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(outs_dir, "accuracy_curve_smoothed.png")); plt.close()

        # 保存最优模型并评估
        torch.save(best_state, os.path.join(outs_dir, "best_model.pth"))
        model.load_state_dict(best_state)
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, val_cm, gt, probs, _ = \
            evaluate_detailed(model, val_loader, criterion, device)
        print(f"\n*** 验证集 {val_cancer_type} 评估 ***")
        print(
            f"Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Prec: {val_prec:.4f},"
            f" Rec: {val_rec:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}"
        )

        # 写入 summary
        record = [
            val_cancer_type,
            str(best_epoch),
            f"{val_loss_hist[best_epoch-1]:.6f}",
            f"{val_auc_hist[best_epoch-1]:.6f}",
            f"{val_f1_hist[best_epoch-1]:.6f}",
            str(len(train_indices)),
            str(len(val_indices))
        ]
        with open(summary_tsv, 'a') as f:
            f.write("\t".join(record) + "\n")

        # saving ROC curve data & imgs 保存 ROC 曲线数据和图像
        fpr, tpr, thresholds = roc_curve(gt, probs)
        df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})
        df.to_csv(os.path.join(outs_dir, 'roc_data.tsv'), sep='\t', index=False)
        plt.figure(); plt.plot(fpr, tpr, label=f'ROC (AUC={val_auc:.4f})'); plt.plot([0,1],[0,1],'--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(outs_dir, "roc_val.png")); plt.close()

    return session_dir


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import trapz

def plot_metrics_and_roc(root_dir,
                         cancer_types,
                         metrics_filename='epoch_metrics.tsv',
                         roc_filename='roc.tsv',
                         save_path=None):
    """
    在一个 2×N 子图里：
      - 第一行：train/val loss
      - 第二行：ROC（只标 'ROC'，图例在右下，AUC 框在图例上方）
      - 序号标在子图左上角外侧
      - 英文字体为 Times New Roman
    """
    # 全局样式
    plt.rcParams.update({
        'font.size':       12,
        'axes.titlesize':  14,
        'axes.labelsize':  12,
        'figure.dpi':      200,
        'lines.linewidth': 2,
        'font.family':     'serif',
        'font.serif':      ['Times New Roman', 'Times'],
    })

    letters = [chr(i) for i in range(ord('a'), ord('z')+1)]
    label_counter = 0

    n = len(cancer_types)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    # 留出外侧空间用于序号
    fig.subplots_adjust(left=0.12, top=0.92, wspace=0.3, hspace=0.4)

    for col, ct in enumerate(cancer_types):
        # ——— Loss 子图 ———
        ax0 = axes[0, col]
        ax0.text(-0.05, 1.02, f'({letters[label_counter]})',
                 transform=ax0.transAxes,
                 fontsize='medium', fontweight='bold',
                 va='bottom', ha='left')
        label_counter += 1

        path0 = os.path.join(root_dir, ct, metrics_filename)
        if os.path.isfile(path0):
            df = pd.read_csv(path0, sep='\t')
            ax0.plot(df['epoch'], df['train_loss'], label='train_loss', linestyle='-')
            ax0.plot(df['epoch'], df['val_loss'],   label='val_loss',   linestyle='--')
            ax0.set_ylabel('Loss')
            ax0.legend(fontsize='small')
            ax0.grid(True, linestyle=':', alpha=0.6)
        else:
            ax0.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax0.set_title(f'{ct} — Loss')

        # ——— ROC 子图 ———
        ax1 = axes[1, col]
        ax1.text(-0.05, 1.02, f'({letters[label_counter]})',
                 transform=ax1.transAxes,
                 fontsize='medium', fontweight='bold',
                 va='bottom', ha='left')
        label_counter += 1

        path1 = os.path.join(root_dir, ct, roc_filename)
        if os.path.isfile(path1):
            df_roc = pd.read_csv(path1, sep='\t')
            fpr, tpr = df_roc['fpr'].values, df_roc['tpr'].values

            # 只标 'ROC'
            ax1.plot(fpr, tpr, label='ROC')
            ax1.plot([0, 1], [0, 1], '--', color='gray', label='random')

            auc = trapz(tpr, fpr)
            ax1.legend(loc='lower right', fontsize='small', frameon=True)

            ax1.text(0.95, 0.2, f'AUC = {auc:.3f}',
                     transform=ax1.transAxes,
                     fontsize='small',
                     ha='right', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='white',
                               edgecolor='black',
                               alpha=0.8))

            ax1.set_xlabel('FPR')
            ax1.set_ylabel('TPR')
            ax1.grid(True, linestyle=':', alpha=0.6)
        else:
            ax1.text(0.5, 0.5, 'No roc.tsv', ha='center', va='center')
        ax1.set_title(f'{ct} — ROC')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"已保存组合图到 {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    # 1) 配置
    k_runs = 5
    # cancer_list = ['BRCA','BLCA','GBM','UCEC','HNSC','LAML','LUSC','LUAD','ESCA','CESC','KICH','KIRC','KIRP','COAD']
    # cancer_list = ['BRCA','GBM','BLCA','UCEC']
    cancer_list = ['CHOL','LGG']


    # cancer_list = ['GBM']
    session_dirs = []
    mut_type = "amp"

    # 2) 生成 k 个随机种子
    seeds = np.random.randint(1, 1000000, size=k_runs)
    seeds = np.random.randint(1, 1000000, size=k_runs)
    # seeds = [257477]
    for se in seeds:
        print(f"\n===== Run with seed {se} =====")
        # 2.2 执行一次交叉验证训练
        out_dir = auto_cross_val(
            data_dir=f"../GeneratedSamples_{mut_type}_compressed/",
            mut=mut_type,
            target_cancer_types=cancer_list,
            seed=int(se),
        )
        if out_dir is None:
            continue
        combined_png = os.path.join(output_dir, "combined_metrics_and_roc.png")
        session_dirs.append(out_dir)
        # 2.3 将 seed 记录到 hyperparameters.txt
        with open(os.path.join(out_dir, "hyperparameters.txt"), 'a') as f:
            f.write(f"seed: {se}\n")
    print("✅ 所有 runs 完成，并已生成合并图。")