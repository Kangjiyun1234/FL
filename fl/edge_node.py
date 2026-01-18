from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from fl.config import TrainConfig
from fl.model import PumpSensorModel
from fl.data import create_sequences

StateDict = Dict[str, torch.Tensor]


@dataclass
class NodeDataset:
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor


class EdgeNode:
    def __init__(self, node_id: int, data, sensor_cols, train_cfg: TrainConfig, device: str = "cpu"):
        self.node_id = node_id
        self.data = data
        self.sensor_cols = sensor_cols
        self.train_cfg = train_cfg
        self.device = torch.device(device)

        # 모델 생성
        self.model = PumpSensorModel(
            input_size=len(sensor_cols),
            hidden_size=train_cfg.hidden_size,
            num_classes=train_cfg.num_classes,
        ).to(self.device)

        # 🔴 수정: weight_decay가 없으면 기본값 0 사용
        weight_decay = getattr(train_cfg, 'weight_decay', 0.0)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=train_cfg.lr, 
            weight_decay=weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss()

        # 데이터 준비
        self.dataset = self._prepare_dataset()

        # 편의 속성
        self.num_train_samples = int(self.dataset.X_train.shape[0])
        self.val_data = (self.dataset.X_val, self.dataset.y_val)

    def _prepare_dataset(self) -> NodeDataset:
        seq_len = self.train_cfg.sequence_length

        X, y = create_sequences(self.data, self.sensor_cols, seq_len)

        if len(X) == 0:
            raise ValueError(
                f"Node {self.node_id}: Not enough data to create sequences. "
                f"len(data)={len(self.data)}, sequence_length={seq_len}"
            )

        # tensor 변환
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)

        # 🔴 수정: train_split이 없으면 기본값 0.8 사용
        train_split = getattr(self.train_cfg, 'train_split', 0.8)
        split_idx = int(len(X_t) * train_split)

        X_train, y_train = X_t[:split_idx], y_t[:split_idx]
        X_val, y_val = X_t[split_idx:], y_t[split_idx:]

        return NodeDataset(
            X_train=X_train.to(self.device),
            y_train=y_train.to(self.device),
            X_val=X_val.to(self.device),
            y_val=y_val.to(self.device),
        )

    def _get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        train_ds = TensorDataset(self.dataset.X_train, self.dataset.y_train)
        val_ds = TensorDataset(self.dataset.X_val, self.dataset.y_val)

        train_loader = DataLoader(train_ds, batch_size=self.train_cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.train_cfg.batch_size, shuffle=False)

        return train_loader, val_loader

    def get_state_dict(self) -> StateDict:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def set_state_dict(self, state: StateDict) -> None:
        # state는 CPU 텐서일 수 있으므로 load 후 device로 이동
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)

    def train_local(self) -> Tuple[float, float]:
        train_loader, val_loader = self._get_loaders()

        # Train
        self.model.train()
        last_train_loss = 0.0
        for _ in range(self.train_cfg.local_epochs):
            for xb, yb in train_loader:
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
                last_train_loss = float(loss.item())

        # Val
        self.model.eval()
        total_val_loss = 0.0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                bs = int(yb.shape[0])
                total_val_loss += float(loss.item()) * bs
                total += bs

        avg_val_loss = total_val_loss / max(total, 1)

        return last_train_loss, avg_val_loss
