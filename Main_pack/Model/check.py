import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from Models import ComplexMultiStreamCNN
from utils.Dataset import TSVDataset


from tqdm import tqdm
import sys

def run_inference(
    data_dir,
    model_path,
    batch_size=32,
    output_file="inference_results_GBM_old.tsv",
    chr_list=None,
    show_progress=True,          # 可开可关
    progress_mininterval=1.0,    # 进度条最小刷新间隔(秒)
    progress_miniters=20         # 至少处理多少个 batch 后才刷新一次
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
        collate_fn=lambda batch: (
            [torch.stack([s[0][j] for s in batch], 0) for j in range(len(batch[0][0]))],
            torch.tensor([s[1] for s in batch], dtype=torch.long),
            [s[2] for s in batch]
        )
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
    out_dir = '/workspace/xuzheng/pyc_workspace/output_buffer/'
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, output_file)
    df = pd.DataFrame(results)
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Saved inference results to {output_path}")

if __name__ == '__main__':
    is_test = True
    cancer_list = ["BLCA"]
    mut_type = "amp"
    del_id = "20250811_024239_761124"
    amp_id = "20250805_161812_530512"
    # amp_id = "20250819_200254_322635"
    for cancer_type in cancer_list:
        if mut_type == "amp":
            experiment_id = amp_id
        else:
            experiment_id = del_id

        chr_list = None
        # chr_list = [2]
        if not is_test:
            if mut_type == "amp":
                DATA_DIR = f"/workspace/xuzheng/pyc_workspace/GeneratedSamples_buffer_compressed_{cancer_type}/"
            else:
                DATA_DIR = f"/workspace/xuzheng/pyc_workspace/GeneratedSamples_buffer_del_{cancer_type}/"
            output_file=f"inference_results_{mut_type}_{cancer_type}.tsv"
        else:
            DATA_DIR = "/workspace/xuzheng/pyc_workspace/GeneratedSamples_buffer_del_test/"
            output_file=f"inference_results_test.tsv"

        print(f"input_dir={DATA_DIR}")



        MODEL_PATH = (
            f"/workspace/xuzheng/pyc_workspace/experimental/tmp/{mut_type}/"
            f"{experiment_id}/{cancer_type}/best_model.pth"
        )
        run_inference(
            DATA_DIR,
            MODEL_PATH,
            batch_size=72,
            # output_file=f"inference_results_{cancer_type}_del.tsv",
            output_file= output_file,

            chr_list=chr_list
        )
