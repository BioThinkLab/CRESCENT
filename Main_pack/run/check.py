
import pandas as pd
import torch
import sys
import os

from torch.utils.data import DataLoader
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
from Model.Models  import ComplexMultiStreamCNN
from Model.utils.Dataset import TSVDataset

from tqdm import tqdm
import sys

def multi_stream_collate(batch):
    """
    batch: list of (matrices, label, filename)
        - matrices: list[Tensor], 每个 Tensor shape=(1,H,W)
    返回:
        - inputs: list[Tensor]，长度 = 流的个数，每个 Tensor shape=(B, 1, H, W)
        - labels: LongTensor shape=(B,)
        - filenames: list[str] 长度=B
    """
    # 解包
    matrices_list, labels, filenames = zip(*batch)  # 各自是长度 B 的 tuple

    # 流的数量（比如多尺度、多通道）
    num_streams = len(matrices_list[0])

    # 对每一路流，在 batch 维度上 stack
    inputs = [
        torch.stack([mats[j] for mats in matrices_list], dim=0)
        for j in range(num_streams)
    ]

    labels = torch.tensor(labels, dtype=torch.long)
    filenames = list(filenames)
    return inputs, labels, filenames

def run_inference(
    data_dir,
    model_path,
    batch_size=32,
    output_file="inference_results_GBM_old.tsv",
    chr_list=None,
    show_progress=True,
    progress_mininterval=1.0,
    progress_miniters=20
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载数据集
    dataset = TSVDataset(data_dir, chromosome_list=chr_list)
    if len(dataset) == 0:
        raise ValueError(f"No samples found in {data_dir}")
    print("dataset initiated")

    # 2. DataLoader（更适合大数据的设置）
    num_workers = 10
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=multi_stream_collate
    )
    print("dataset loaded")

    # 3. 构建模型并加载权重
    sample, _, _ = dataset[0]
    input_shapes = [(t.shape[1], t.shape[2]) for t in sample]
    model = ComplexMultiStreamCNN(input_shapes, num_classes=1).to(device)

    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. 推理循环（低开销进度条）
    results = []
    total_batches = len(loader)
    use_bar = show_progress and total_batches > 0

    # 仅在交互式终端显示，非 TTY 则自动关闭，避免无谓开销
    disable_bar = not sys.stderr.isatty() or not use_bar

    pbar = tqdm(
        total=total_batches,
        desc="Inference",
        unit="batch",
        dynamic_ncols=True,
        mininterval=progress_mininterval,  # 至少隔这么久才刷新
        miniters=progress_miniters,        # 或至少处理这么多 batch 才刷新
        leave=False,
        disable=disable_bar
    )

    with torch.no_grad():
        for inputs, _, file_names in loader:
            # 送入 device（尽量让拷贝异步，配合 pin_memory）
            if device.type == "cuda":
                inputs = [x.to(device, non_blocking=True) for x in inputs]
            else:
                inputs = [x.to(device) for x in inputs]

            logits = model(inputs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            pc = probs.detach().cpu().tolist()
            prc = preds.detach().cpu().tolist()
            for fn, p, pr in zip(file_names, pc, prc):
                results.append({
                    'filename':    fn,
                    'probability': float(p),
                    'prediction':  int(pr)
                })

            # 只做一次轻量的计数更新，不打印字符串，避免 I/O 抖动
            if not disable_bar:
                pbar.update(1)

    if not disable_bar:
        pbar.close()

    # 5. 保存结果
    out_dir = './buffer/inference_buffer'
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, output_file)
    df = pd.DataFrame(results)
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Saved inference results to {output_path}")

def run_inf(project_name, mut):
    cancer_list = project_name
    mut_type = mut
    DATA_DIR = "./buffer/instance"
    output_file=f"inference_results_{mut_type}_{project_name}.tsv"
    print(f"input_dir={DATA_DIR}")
    MODEL_PATH = f"../Model/model_{mut}.pth"
    run_inference(
        DATA_DIR,
        MODEL_PATH,
        batch_size=72,
        output_file= output_file,
        chr_list=None
    )
