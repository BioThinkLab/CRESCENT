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
from utils.evaluate import evaluate_detailed  # return:(val_loss, acc, prec, rec, f1, auc, cm, gt, probs, best_threshold)

output_dir = "../experimental/"


def get_prefix(fn: str) -> str:
    # Used to identify augmented datasets
    name = os.path.splitext(fn)[0]
    return name.split('_aug_')[0]


# --------------------- Fix Random Seed ---------------------
def set_seed(seed: int = 42):
    _py_random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------- Moving Average Smoothing Function ---------------------
def moving_average(x, w=5):
    return np.convolve(x, np.ones(w) / w, mode='valid')


# --------------------- Dataset Collate Function ---------------------
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


# --------------------- Training and Validation Function ---------------------
def auto_cross_val(data_dir,
                   mut="del",
                   target_cancer_types=None,
                   seed=42):
    """
    data_dir: sample directory
    mut: "del" or "amp", mutation type directory name
    target_cancer_types: None or list of str, specifies cancer types to be left out
    """
    set_seed(seed)
    dataset = TSVDataset(data_dir)
    if len(dataset) == 0:
        print("No valid training data found!")
        return

    # Input scale shapes (auto loaded from data file)
    first_sample, _, _ = dataset[0]
    input_shapes = [(t.shape[1], t.shape[2]) for t in first_sample]
    print("Input branch scales:", input_shapes)

    # Collect all or specified cancer types
    all_types = set(fn.split('_')[0] for _, _, fn in dataset)
    if target_cancer_types is not None:
        cancer_types = [t for t in target_cancer_types if t in all_types]
        missing = set(target_cancer_types) - set(cancer_types)
        if missing:
            print(f"Warning: The following specified cancer types were not found in the data and will be ignored: {missing}")
    else:
        cancer_types = sorted(all_types)
    print("Cancer types to run:", cancer_types)

    num_epochs, batch_size, learning_rate = 10, 8, 1e-5  # amp: lr=1e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create session directory
    base_dir = os.path.join(output_dir, mut)
    session_dir = os.path.join(
        base_dir,
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(seed)
    )
    os.makedirs(session_dir, exist_ok=True)

    # Save hyperparameters to file
    with open(os.path.join(session_dir, "hyperparameters.txt"), "w") as f:
        f.write(f"num_epochs: {num_epochs}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"device: {device}\n")
        f.write("scheduler: ReduceLROnPlateau\n")

    # Global summary file
    summary_tsv = os.path.join(session_dir, "summary_metrics.tsv")
    header = [
        'cancer_type', 'best_epoch', 'best_loss', 'best_auc', 'best_f1',
        'train_size', 'val_size'
    ]
    if not os.path.exists(summary_tsv):
        with open(summary_tsv, 'w') as f:
            f.write("\t".join(header) + "\n")

    for val_cancer_type in cancer_types:
        print(f"\n===== Held-out Validation Type: {val_cancer_type} =====")
        indices = list(range(len(dataset)))

        # Hold out current type as validation, use the rest for training
        train_indices = [i for i in indices if dataset[i][2].split('_')[0] != val_cancer_type]
        val_indices = [i for i in indices if dataset[i][2].split('_')[0] == val_cancer_type]

        print(f"Training size: {len(train_indices)}, Validation size: {len(val_indices)}")

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

        # Training logs
        train_loss_hist, train_acc_hist = [], []
        val_loss_hist, val_acc_hist, val_auc_hist, val_f1_hist = [], [], [], []
        best_auc, best_epoch, best_state = -float('inf'), -1, None

        # Prepare visualization directory
        outs_dir = os.path.join(session_dir, val_cancer_type)
        os.makedirs(outs_dir, exist_ok=True)
        loss_fig, loss_ax = plt.subplots(figsize=(6, 4))
        acc_fig, acc_ax = plt.subplots(figsize=(6, 4))

        # Training and validation loop
        for epoch in range(1, num_epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, *_ = \
                evaluate_detailed(model, val_loader, criterion, device)
            _, train_acc, *_ = evaluate_detailed(model, train_loader, criterion, device)

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

            # Save validation filenames
            current_val_filenames = []
            for batch in val_loader:
                _, _, names = batch
                current_val_filenames.extend(names)
            with open(os.path.join(outs_dir, f"val_filenames_epoch_{epoch}.txt"), "w") as vf:
                vf.write("\n".join(current_val_filenames))

            # Update best model
            if val_auc >= best_auc:
                best_auc, best_epoch = val_auc, epoch
                best_state = copy.deepcopy(model.state_dict())

            # Plot and save loss & accuracy curves
            loss_ax.clear()
            loss_ax.plot(range(1, epoch + 1), train_loss_hist, label='Training Loss')
            loss_ax.plot(range(1, epoch + 1), val_loss_hist, label='Validation Loss')
            loss_ax.set_xlabel('Epoch')
            loss_ax.set_ylabel('Loss')
            loss_ax.legend()
            loss_ax.grid(True)
            loss_fig.savefig(os.path.join(outs_dir, "loss_curve_raw.png"), dpi=150)

            acc_ax.clear()
            acc_ax.plot(range(1, epoch + 1), train_acc_hist, label='Training Acc')
            acc_ax.plot(range(1, epoch + 1), val_acc_hist, label='Validation Acc')
            acc_ax.set_xlabel('Epoch')
            acc_ax.set_ylabel('Accuracy')
            acc_ax.legend()
            acc_ax.grid(True)
            acc_fig.savefig(os.path.join(outs_dir, "accuracy_curve_raw.png"), dpi=150)

        # Save per-epoch metrics
        epoch_tsv = os.path.join(outs_dir, "epoch_metrics.tsv")
        with open(epoch_tsv, 'w') as f:
            f.write("epoch\ttrain_loss\ttrain_acc\tval_loss\tval_acc\tval_auc\tval_f1\n")
            for e in range(num_epochs):
                f.write(
                    f"{e+1}\t{train_loss_hist[e]:.6f}\t{train_acc_hist[e]:.6f}"
                    f"\t{val_loss_hist[e]:.6f}\t{val_acc_hist[e]:.6f}"
                    f"\t{val_auc_hist[e]:.6f}\t{val_f1_hist[e]:.6f}\n"
                )

        # Save best model and evaluate
        torch.save(best_state, os.path.join(outs_dir, "best_model.pth"))
        model.load_state_dict(best_state)
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, val_cm, gt, probs, _ = \
            evaluate_detailed(model, val_loader, criterion, device)

        print(f"\n*** Validation Evaluation for {val_cancer_type} ***")
        print(
            f"Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Prec: {val_prec:.4f},"
            f" Rec: {val_rec:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}"
        )

        # Write to summary
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

        # Save ROC curve data and image
        fpr, tpr, thresholds = roc_curve(gt, probs)
        df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})
        df.to_csv(os.path.join(outs_dir, 'roc_data.tsv'), sep='\t', index=False)

        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC (AUC={val_auc:.4f})')
        plt.plot([0, 1], [0, 1], '--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(outs_dir, "roc_val.png"))
        plt.close()

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
    In a 2×N subplot:
      - First row: train/val loss
      - Second row: ROC (only labeled 'ROC', legend in lower-right, AUC box above legend)
      - Subplot labels at the upper-left outside the axes
      - Font: Times New Roman
    """
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'figure.dpi': 200,
        'lines.linewidth': 2,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times'],
    })

    letters = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    label_counter = 0

    n = len(cancer_types)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    fig.subplots_adjust(left=0.12, top=0.92, wspace=0.3, hspace=0.4)

    for col, ct in enumerate(cancer_types):
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
            ax0.plot(df['epoch'], df['val_loss'], label='val_loss', linestyle='--')
            ax0.set_ylabel('Loss')
            ax0.legend(fontsize='small')
            ax0.grid(True, linestyle=':', alpha=0.6)
        else:
            ax0.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax0.set_title(f'{ct} — Loss')

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
        print(f"Combined figure saved to {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    # 1) Configuration
    k_runs = 5
    cancer_list = ['CHOL', 'LGG']

    session_dirs = []
    mut_type = "amp"

    # 2) Generate k random seeds
    seeds = np.random.randint(1, 1000000, size=k_runs)

    for se in seeds:
        print(f"\n===== Run with seed {se} =====")

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

        # Record seed to hyperparameters.txt
        with open(os.path.join(out_dir, "hyperparameters.txt"), 'a') as f:
            f.write(f"seed: {se}\n")

    print("✅ All runs completed and combined figures generated.")
