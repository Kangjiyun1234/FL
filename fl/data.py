# fl/data.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class NodeDataset:
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor


def make_node_dataset(
    node_id: int,
    num_samples: int,
    num_features: int,
    noise_std: float,
    non_iid: bool,
    val_ratio: float,
    seed: int,
    device: str,
) -> NodeDataset:
    """
    노드별 데이터 생성.
    - non_iid=True: 노드마다 feature mean을 살짝 이동시켜 분포 차이를 만듦.
    - 회귀 타깃: y = X @ w + b + noise
    """
    rng = np.random.default_rng(seed + node_id)

    # Non-IID shift
    shift = (node_id * 0.7) if non_iid else 0.0
    X = rng.normal(loc=shift, scale=1.0, size=(num_samples, num_features)).astype(np.float32)

    # True weights (global underlying function)
    w = rng.normal(loc=0.0, scale=1.0, size=(num_features, 1)).astype(np.float32)
    b = np.float32(0.3)

    y = (X @ w).reshape(-1) + b
    y += rng.normal(loc=0.0, scale=noise_std, size=y.shape).astype(np.float32)

    # Split train/val
    idx = np.arange(num_samples)
    rng.shuffle(idx)
    val_n = int(num_samples * val_ratio)
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]

    X_train = torch.from_numpy(X[train_idx]).to(device)
    y_train = torch.from_numpy(y[train_idx]).to(device).unsqueeze(1)

    X_val = torch.from_numpy(X[val_idx]).to(device)
    y_val = torch.from_numpy(y[val_idx]).to(device).unsqueeze(1)

    return NodeDataset(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
