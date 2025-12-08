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
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8),squeeze=False)
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
    # 1) 父目录，下面每个子目录是一次 session
    parent_dir = '/workspace/xuzheng/pyc_workspace/experimental'

    # 2) 想要绘制的所有 cancer 类型
    cancers_to_plot = ['BLCA','UCEC','BRCA','GBM']

    # 3) 遍历所有 session 子目录
    for session_name in os.listdir(parent_dir):
        session_dir = os.path.join(parent_dir, session_name)
        if not os.path.isdir(session_dir):
            continue

        # 4) 筛出这个 session 里实际存在的 cancer 子文件夹
        available = [
            ct for ct in cancers_to_plot
            if os.path.isdir(os.path.join(session_dir, ct))
        ]
        if not available:
            print(f"跳过 {session_name}：没有匹配的 cancer 子目录")
            continue

        # 5) 生成并保存图到 session_dir/combined_metrics_and_roc.png
        out_png = os.path.join(session_dir, 'combined_metrics_and_roc.png')
        plot_metrics_and_roc(
            root_dir=session_dir,
            cancer_types=available,
            metrics_filename='epoch_metrics.tsv',
            roc_filename='roc.tsv',
            save_path=out_png
        )
