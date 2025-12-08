import pandas as pd
import os
import glob
import re
from torch.utils.data import Dataset
import torch

class TSVDataset(Dataset):
    """
    A lazy-loading TSV dataset for PyTorch.

    Args:
        directory (str, optional): Directory containing TSV files.
        file_paths (list[str], optional): Explicit list of TSV file paths.
        mode (str): 'all' (default) include all files; 'val' exclude files containing '_aug_'.
        train (bool): If True, applies transform; otherwise no augmentation.
        transform (callable): Augmentation function for each output tensor input of shape (1,H,W) or (C,H,W).
        chromosome_list (list[str], optional): List of chromosome identifiers to include (e.g. ['1','2','X'] or ['chr1','chr2']).
    """

    def __init__(self,
                 directory=None,
                 file_paths=None,
                 mode='all',
                 train=True,
                 transform=None,
                 chromosome_list=None):
        # Determine initial file list
        if file_paths is not None:
            paths = file_paths
        else:
            if directory is None:
                raise ValueError("`directory` or `file_paths` must be provided.")
            paths = glob.glob(os.path.join(directory, "*.tsv"))
            if mode == 'val':
                paths = [p for p in paths if '_aug_' not in os.path.basename(p)]

        # Filter by chromosome if requested
        if chromosome_list is not None:
            # Normalize requested list: allow with or without 'chr' prefix
            norm_chrs = set()
            for c in chromosome_list:
                c_str = str(c)
                if c_str.lower().startswith('chr'):
                    norm_chrs.add(c_str)
                else:
                    norm_chrs.add('chr' + c_str)
            filtered = []
            for p in paths:
                name = os.path.basename(p)
                # Expect pattern like XXX_chrx_XXXXX... , extract after '_chr' and before next '_'
                m = re.search(r'_chr([^_]+)_', name)
                if m:
                    ch = 'chr' + m.group(1)
                    if ch in norm_chrs:
                        filtered.append(p)
            paths = filtered

        self.file_paths = sorted(paths)
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        sample = self.parse_file(file_path)
        if sample is None:
            raise ValueError(f"Failed to parse sample: {file_path}")

        matrices, label, filename = sample

        # Apply transform if in training mode
        if self.train and self.transform is not None:
            matrices = [self.transform(m) for m in matrices]

        return matrices, label, filename

    def parse_file(self, path):
        # Implement TSV parsing logic here
        # Should return (list_of_tensors, label_int, filename_str)
        raise NotImplementedError("parse_file method must be implemented.")

    @staticmethod
    def parse_file(input_source):
        """
        Parse a TSV file or equivalent DataFrame into tensors and metadata.
        """
        matrices = []
        label = None

        # Read lines from file or DataFrame
        if isinstance(input_source, str):
            filename = os.path.basename(input_source)
            with open(input_source, 'r') as f:
                lines = [l.strip() for l in f if l.strip()]
        elif isinstance(input_source, pd.DataFrame):
            filename = None
            if 'line' in input_source.columns:
                lines = input_source['line'].dropna().astype(str).tolist()
            else:
                first_col = input_source.columns[0]
                lines = input_source[first_col].dropna().astype(str).tolist()
        else:
            raise TypeError("Input must be a file path or pandas DataFrame.")

        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("label:"):
                label = int(line.split(':', 1)[1].strip())
                i += 1
            elif line.startswith("scale:"):
                # Parse matrix shape
                parts = line.split(',')
                shape_part = next(p for p in parts if 'shape:' in p)
                H, W = map(int, shape_part.split('shape:')[1].split('x'))
                i += 1
                # Read matrix rows
                matrix_data = []
                for _ in range(H):
                    if i >= len(lines):
                        raise ValueError(f"Expected {H} rows but found fewer.")
                    row = list(map(float, lines[i].split()))
                    if len(row) != W:
                        raise ValueError(f"Expected {W} columns but found {len(row)}.")
                    matrix_data.append(row)
                    i += 1
                # 每个矩阵都返回 shape=(1, H, W)
                matrices.append(torch.tensor(matrix_data, dtype=torch.float32).unsqueeze(0))
            else:
                i += 1

        # Drop samples without label or matrix
        if label is None or not matrices:
            return None
        return matrices, label, filename

    @property
    def all_file_paths(self):
        return list(self.file_paths)

    def get_cancer_types(self):
        return {os.path.basename(p).split('_')[0] for p in self.file_paths}
