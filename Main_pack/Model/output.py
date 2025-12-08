# 该文件的主要作用是推理，接受单一文件的输入，输出分类结果(prob)到文件中
# 指定癌症，使用data_base_dir,自动遍历染色体和读取数据文件
import os.path

import numpy as np
import torch

from utils.abstract_matrix import process_single_file
from utils.Dataset import TSVDataset
from utils.conf import CONFIG

import torch
from torch import nn
# torch.manual_seed(42)

# ---------------------
# 模型定义
# ---------------------

import torch.nn as nn

# —— 深度可分离卷积模块 ——
import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),              # ① 归一化更稳定
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),                  # ② 中等强度 Dropout
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# —— 深度可分离卷积模块 ——
class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   groups=in_ch,
                                   bias=bias)
        self.pointwise = nn.Conv2d(in_ch, out_ch,
                                   kernel_size=1,
                                   bias=bias)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# 1) 定义 ECA 通道注意力
class ECABlock(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 1D 卷积：在 channel 维度上做局部交互
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        y = self.avg_pool(x).squeeze(-1).squeeze(-1)   # -> (B, C)
        y = y.unsqueeze(1)                            # -> (B, 1, C)
        y = self.conv(y)                              # -> (B, 1, C)
        y = self.sigmoid(y).squeeze(1)                # -> (B, C)
        y = y.unsqueeze(-1).unsqueeze(-1)             # -> (B, C, 1, 1)
        return x * y
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, drop_prob=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout2d(p=drop_prob) if drop_prob > 0 else nn.Identity()
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ComplexMultiStreamCNN(nn.Module):
    def __init__(self,
                 input_shapes,
                 feature_dim=256,
                 num_classes=2,
                 drop_prob=0.1):
        super().__init__()
        self.feature_dim = feature_dim

        # 原始分支 & 累加分支
        self.branches = nn.ModuleList()
        self.sum_branches = nn.ModuleList()
        for _ in input_shapes:
            self.branches.append(self._make_branch(feature_dim, drop_prob))
            self.sum_branches.append(self._make_branch(feature_dim, drop_prob))

        # 多头自注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            batch_first=True
        )

        # Transformer 内部的残差 + LayerNorm + FFN
        self.norm1 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        self.norm2 = nn.LayerNorm(feature_dim)

        # 最终分类器
        # self.classifier = nn.Linear(feature_dim, num_classes)
        hidden = feature_dim
        self.classifier = MLPHead(feature_dim, hidden,
                                  num_classes, drop=drop_prob)

    def _make_branch(self, feature_dim, drop_prob):
        return nn.Sequential(
            # 基础卷积模块
            # nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            SeparableConv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 两个残差块
            ResidualBlock(64, 128, stride=1, drop_prob=drop_prob),
            ResidualBlock(128, 256, stride=1, drop_prob=drop_prob),
            ResidualBlock(256, 256, stride=1, drop_prob=drop_prob),

            # ← 在这里插入 ECA 通道注意力
            ECABlock(256, k_size=3),

            # 全局池化 & 展平
            nn.AdaptiveAvgPool2d((1, 1)),  # -> (batch,64,1,1)
            nn.Flatten(),                  # -> (batch,64)

            # Dropout + 映射到 feature_dim
            nn.Dropout(p=drop_prob),
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        # 1) 原始分支特征
        orig_feats = [
            branch(x) for branch, x in zip(self.branches, inputs)
        ]

        # 2) 累加分支特征
        sum_feats = []
        for sum_branch, x in zip(self.sum_branches, inputs):
            # 在第 2 维度上求和，再扩展回原始尺寸
            sum_seq = x.sum(dim=2, keepdim=True)
            sum_mat = sum_seq.expand(-1, -1, x.size(2), -1)
            sum_feats.append(sum_branch(sum_mat))

        # 3) 拼接所有特征
        all_feats = orig_feats + sum_feats            # list of (batch, feature_dim)
        feat_stack = torch.stack(all_feats, dim=1)    # -> (batch, 2N, feature_dim)

        # 4) 多头自注意力融合
        attn_out, _ = self.multihead_attn(feat_stack,
                                         feat_stack,
                                         feat_stack)

        # 5) Transformer-style 残差 + Norm + FFN
        res1 = feat_stack + attn_out
        normed1 = self.norm1(res1)
        ffn_out = self.ffn(normed1)
        res2 = normed1 + ffn_out
        normed2 = self.norm2(res2)

        # 6) 维度平均 + 分类
        fused = normed2.mean(dim=1)  # -> (batch, feature_dim)
        logits = self.classifier(fused)
        return logits

import numpy as np

import numpy as np

def extract_matrix_from_feature_data(
    feature_data, center_idx, H, W,
    norm_method='col',       # None, 'col', or 'matrix'
    outlier_threshold=10.0  # float, e.g. 3.0 for clipping z-scores
):
    """
    从 feature_data 中以 center_idx 为中心，提取 H 行、W 列数据；
    - 如果原始行数 < H，则先将 feature_data 在行方向上按比例压缩到 2000 行。
    - 边界处直接取顶/底部 H 行，再沿行方向循环滚动，使 center_idx 对应行到达中间位置。
    - 列数少于 W 时：先计算“去最大值后”的每列均值，再取整体均值，生成常数列补齐；
      列数多于 W 时：按列均值排序后取前 W 列。
    - norm_method: None（不标准化），'col'（按列 z-score），'matrix'（全矩阵 z-score）。
    - outlier_threshold: 若提供，则对标准化后的数据按 [-threshold, threshold] 进行裁剪。
    """
    # 确保为浮点类型
    feature_data = feature_data.astype(float)

    # 1. 预先标准化并处理离群值
    if norm_method == 'col':
        means = np.mean(feature_data, axis=0)
        stds = np.std(feature_data, axis=0)
        stds[stds == 0] = 1.0
        feature_data = (feature_data - means) / stds
    elif norm_method == 'matrix':
        mean_all = np.mean(feature_data)
        std_all = np.std(feature_data)
        std_all = std_all if std_all != 0 else 1.0
        feature_data = (feature_data - mean_all) / std_all

    # 离群值裁剪
    if outlier_threshold is not None:
        feature_data = np.clip(feature_data, -outlier_threshold, outlier_threshold)

    # 原始参数
    num_rows, num_cols = feature_data.shape
    half = H // 2

    # 如果原始行数不足 H，先压缩到 2000 行
    if num_rows < H:
        target_rows = 2000
        new_idx = np.linspace(0, num_rows - 1, num=target_rows)
        compressed = np.zeros((target_rows, num_cols), dtype=float)
        for j in range(num_cols):
            compressed[:, j] = np.interp(new_idx, np.arange(num_rows), feature_data[:, j])
        feature_data = compressed
        num_rows = target_rows

    # 行方向：中心提取 + 边界滚动
    if 0 <= center_idx - half and center_idx + half < num_rows:
        start = center_idx - half
        mat = feature_data[start:start + H]
    else:
        if center_idx < half:
            mat = feature_data[0:H]
            orig_pos = center_idx
        else:
            mat = feature_data[num_rows - H:num_rows]
            orig_pos = center_idx - (num_rows - H)
        shift = half - orig_pos
        mat = np.roll(mat, shift=shift, axis=0)

    # 列排序：按列均值降序
    col_means = np.mean(mat, axis=0)
    sorted_idx = np.argsort(-col_means)
    mat = mat[:, sorted_idx]

    # 列数不符时的填充或截断
    cur_W = mat.shape[1]
    if cur_W < W:
        col_means_no_max = []
        for j in range(cur_W):
            col = mat[:, j]
            col_means_no_max.append((col.sum() - col.max()) / (len(col) - 1))
        final_mean = float(np.mean(col_means_no_max))
        missing = W - cur_W
        pad = np.full((H, missing), final_mean, dtype=mat.dtype)
        mat = np.concatenate([mat, pad], axis=1)
    elif cur_W > W:
        mat = mat[:, :W]

    return mat


# def calculate_index(input_file, num=600):
#     """
#     对单个 TSV 文件等间隔处理，返回 position 索引列表及每个索引为中心的覆盖区间。
#
#     参数:
#       input_file: TSV 文件路径
#       num: 要等间隔选取的点数量（默认 800）
#
#     返回:
#       positions: 包含 num 个整数位置的列表（0-based，对应数据行的索引）
#       position_ranges: 一个字典，key 是 position，value 是 'a-b' 的字符串，表示该 position 为中心的行范围
#     """
#     with open(input_file, 'r') as f:
#         lines = f.readlines()
#
#     data_lines = lines[1:]  # 去掉表头
#     total = len(data_lines)
#
#     if total < num:
#         positions = list(range(total))
#     else:
#         positions = np.linspace(0, total - 1, num=num, dtype=int).tolist()
#
#     # 划分区间
#     position_ranges = {}
#     boundaries = [0] + [(positions[i] + positions[i+1]) // 2 + 1 for i in range(len(positions) - 1)] + [total]
#
#     for i, pos in enumerate(positions):
#         start = boundaries[i]
#         end = boundaries[i+1] - 1  # 包含边界
#         position_ranges[pos] = f"{start}-{end}"
#
#     return positions, position_ranges

def calculate_index(input_file, step=5):
    """
    每隔 step 行选一个中心点。
    """
    with open(input_file, 'r') as f:
        total = len(f.readlines()) - 1

    positions = list(range(0, total, step))
    # 如果希望最后一段也能包含到最后一行，可以在末尾加上 total-1
    if positions[-1] != total - 1:
        positions.append(total - 1)

    # 每个 pos 依然只映射到自身
    position_ranges = {pos: f"{pos}-{pos}" for pos in positions}
    return positions, position_ranges


import torch
import pandas as pd



def process(input_file, model, df_res=None):
    """
    1. 调用 calculate_index(input_file) 得到 positions 与 range_dict；
    2. 用 pandas 读取整个 TSV （一次性读入）得到 df_full；把所有特征列提取成一个 NumPy 数组 feature_data_full；
    3. 对每个 position：
         a) 从 feature_data_full 调用 extract_matrix_from_feature_data(...)
            生成多尺度子矩阵列表；
         b) 组合成模型输入，推理得到 prob_0, prob_1；
         c) 根据 range_dict[pos] 得到行范围，把 prob_0/prob_1 写回 df_slice；
    4. 所有 pos 循环结束后，把各个 df_slice concat → full_df，做 和第二版 一样的均值对比、二值化→is_true、筛列 等后处理；
    5. 把累积结果拼到传入的 df_res 中并返回 (result_list, df_res)。
    """
    # 如果外部没有传入 df_res，就新建一个空 DataFrame 用来累积
    if df_res is None:
        df_res = pd.DataFrame()

    # 多尺度输入形状（请与模型初始化时保持一致）
    input_shapes = [(200, 40), (700, 40), (1500, 40)]

    # 1. 先生成 positions, range_dict（保持第二版原样）
    positions, range_dict = calculate_index(input_file)

    # 2. 用 pandas 读入整个 TSV
    try:
        df_full = pd.read_csv(input_file, sep='\t', dtype={0: str})
    except Exception as e:
        print(f"无法读取文件 {input_file}：{e}")
        return [], df_res

    # 假定列 0,1,2 分别是 ["chr","start","end"]，从第 3 列(索引=3) 开始到最后全是特征
    feat_start = 3
    feat_end = df_full.shape[1]
    feature_data_full = df_full.iloc[:, feat_start:feat_end].astype(float).values  # numpy 数组

    # 3. 模型准备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    result = []        # 用来存 {position: prob_1}
    all_df_parts = []  # 用来收集每个 position 对应的行区间 DataFrame + prob

    with torch.no_grad():
        for pos in positions:
            print(pos)
            # 3.1 先把这一 pos 对应的行区间切出来（只放 chr/start/end 列，后面附加 prob）
            range_str = range_dict[pos]      # 比如 "100-150"
            start_idx, end_idx = map(int, range_str.split('-'))
            df_slice = df_full.iloc[start_idx:end_idx+1].copy()
            if df_slice.empty:
                continue

            # 3.2 依次用 extract_matrix_from_feature_data 生成每个尺度 (H, W) 的子矩阵
            matrices = []
            success = True
            for (H, W) in input_shapes:
                try:
                    mat = extract_matrix_from_feature_data(
                        feature_data_full,
                        center_idx=pos,
                        H=H,
                        W=W,
                        norm_method='col',        # 这里示例用“按列归一化”
                        outlier_threshold=10.0    # 这里示例用阈值 10.0 裁剪
                    )
                except Exception as e:
                    print(f"跳过 pos={pos}，提取尺度 ({H},{W}) 出错：{e}")
                    success = False

                if not success or mat is None or mat.shape != (H, W):
                    # 只要有一个尺度失败，就跳过这个 pos
                    success = False
                    break

                matrices.append(mat)

            if not success:
                continue

            # 3.3 把每个 numpy 矩阵转成 Tensor 并移动到 device, 注意要加 batch 维度和 channel 维度
            #     这里假定 ComplexMultiStreamCNN 的 forward 接受一个列表：
            #     [tensor(size=[1,1,H1,W1]), tensor(size=[1,1,H2,W2]), tensor(size=[1,1,H3,W3])]
            inputs = []
            for mat in matrices:
                t = torch.from_numpy(mat).float().unsqueeze(0).unsqueeze(0).to(device)
                inputs.append(t)

            # 3.4 推理
            logits = model(inputs)                # shape 应该是 [1, num_classes]
            probs = torch.softmax(logits, dim=1)   # [1, 2] 假设二分类
            prob_0 = probs[0, 0].item()
            prob_1 = probs[0, 1].item()

            # 3.5 记录 pos 对应的正类概率
            result.append({pos: prob_1})

            # 3.6 把 prob_0, prob_1 写回 df_slice
            df_slice["prob_0"] = prob_0
            df_slice["prob_1"] = prob_1
            all_df_parts.append(df_slice)

    # 4. 后处理：如果至少有一个子区间返回 df
    if all_df_parts:
        full_df = pd.concat(all_df_parts, ignore_index=True)

        # 4.1 计算平均概率
        mean_0 = full_df["prob_0"].mean()
        mean_1 = full_df["prob_1"].mean()

        # 4.2 根据均值选择最终保留哪一列作为 prob
        if mean_0 < mean_1:
            full_df["prob"] = full_df["prob_0"]
        else:
            full_df["prob"] = full_df["prob_1"]

        # 4.3 二值化
        full_df["is_true"] = (full_df["prob"] > 0.5).astype(int)

        # 4.4 只保留 ["chr","start","end","prob","is_true"]
        final_df = full_df[["chr", "start", "end", "prob", "is_true"]]

        # 4.5 合并到传入的 df_res
        if df_res is None or df_res.empty:
            df_res = final_df
        else:
            df_res = pd.concat([df_res, final_df], ignore_index=True)

    return result, df_res


def run(cancer_type, model):
    chromosomes = [f"chr{i}" for i in range(1, 23)]
    df_res = None
    for chromosome in chromosomes:
        filename = 'cnv_'+chromosome+'.txt'
        input_file = os.path.join(CONFIG["DATA_BASE_DIR"], cancer_type, filename)
        result, df_res = process(input_file, model, df_res=df_res)
        output_dir = "/workspace/xuzheng/pyc_workspace/Model/output/"+cancer_type+'/'
        output_file = chromosome+".txt"
        file = os.path.join(output_dir, output_file)
        with open(file, "w") as f:
            f.writelines(f"{line}\n" for line in result)

    df_res.to_csv(f'/workspace/xuzheng/pyc_workspace/Model/output/{cancer_type}_result.tsv')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cancer_type = 'BLCA'
    target_cancer_types=['UCEC','BLCA', 'BRCA', 'GBM', 'SARC','KIRP', 'HNSC', ]
   # model_path = '/workspace/xuzheng/pyc_workspace/experimental/tmp/amp/20250603_013356/'+cancer_type+'/best_model.pth'
    input_shapes = [(200, 40), (700, 40), (1500, 40)]
   #  model = ComplexMultiStreamCNN(input_shapes).to(device)
   #  model.load_state_dict(torch.load(model_path, map_location=device))
    for cancer_type in target_cancer_types:
        target_cancer_types = ['UCEC', 'BLCA', 'BRCA', 'GBM', 'SARC', 'KIRP', 'HNSC', ]
        model = ComplexMultiStreamCNN(input_shapes).to(device)
        model_path = '/workspace/xuzheng/pyc_workspace/experimental/tmp/amp/20250603_013356G/' + cancer_type + '/best_model.pth'
        model.load_state_dict(torch.load(model_path, map_location=device))
        run(cancer_type, model)
