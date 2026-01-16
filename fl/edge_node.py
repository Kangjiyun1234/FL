# fl/edge_node.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fl.config import TrainConfig
from fl.data import NodeDataset
from fl.model import SimpleRegressor


@dataclass
class EdgeNode:
    node_id: int
    dataset: NodeDataset
    train_cfg: TrainConfig

    def __post_init__(self) -> None:
        self.model = SimpleRegressor(in_dim=self.dataset.X_train.shape[1]).to(self.train_cfg.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_cfg.lr,
            weight_decay=self.train_cfg.weight_decay,
        )

    @property
    def num_train_samples(self) -> int:
        return int(self.dataset.X_train.shape[0])

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def set_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state, strict=True)

    def train_local(self) -> float:
        """
        로컬 학습 수행 후 평균 loss 반환 (요청하신 로깅/평균 로스 개선 반영).
        """
        self.model.train()

        ds = TensorDataset(self.dataset.X_train, self.dataset.y_train)
        loader = DataLoader(ds, batch_size=self.train_cfg.batch_size, shuffle=True)

        losses_sum = 0.0
        steps = 0

        for _epoch in range(self.train_cfg.local_epochs):
            for Xb, yb in loader:
                self.optimizer.zero_grad(set_to_none=True)
                preds = self.model(Xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()

                losses_sum += float(loss.item())
                steps += 1

        avg_loss = losses_sum / max(steps, 1)
        return avg_loss

    @torch.no_grad()
    def evaluate_val(self) -> float:
        """노드의 validation 데이터로 loss 측정."""
        self.model.eval()
        preds = self.model(self.dataset.X_val)
        loss = self.criterion(preds, self.dataset.y_val)
        return float(loss.item())
