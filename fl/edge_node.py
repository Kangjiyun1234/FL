from __future__ import annotations
from typing import List, Tuple, Optional
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fl.config import TrainConfig
from fl.model import PumpMaintenanceClassifier, TurbofanLSTM

StateDict = dict[str, torch.Tensor]


@dataclass
class NodeDataset:
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor


class EdgeNode:
    def __init__(
        self,
        node_id: int,
        data: pd.DataFrame,
        sensor_cols: List[str],
        train_cfg: TrainConfig,
        device: str = "cpu",
        *,
        eval_only: bool = False,          # ✅ 추가: 평가 전용 노드
        train_split: Optional[float] = None,  # ✅ 필요 시 split override
        debug_first_batch: bool = False,  # ✅ 필요 시 디버그
    ):
        self.node_id = node_id
        self.device = device
        self.train_cfg = train_cfg
        self.data = data
        self.sensor_cols = sensor_cols
        self.eval_only = eval_only
        self.debug_first_batch = debug_first_batch

        # split override (기본값: cfg 값)
        self._train_split = float(train_split) if train_split is not None else float(train_cfg.train_split)

        self.dataset = self._prepare_dataset()
        self.num_train_samples = int(self.dataset.X_train.shape[0])

        input_size = len(sensor_cols)

        if "unit_id" in self.data.columns:
            self.model = TurbofanLSTM(
                input_size=input_size,
                hidden_size=train_cfg.hidden_size,
                num_classes=1,
                num_layers=2,
                dropout=0.3,
            ).to(device)
        else:
            self.model = PumpMaintenanceClassifier(
                input_size=input_size,
                hidden_size=train_cfg.hidden_size,
                num_classes=1,
                dropout=0.3,
            ).to(device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )

        # pos_weight는 Trainer에서도 동일하게 쓰는 값이므로 일관 유지
        pos_weight = torch.tensor([6.0]).to(device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def _prepare_dataset(self) -> NodeDataset:
        seq_len = self.train_cfg.sequence_length

        # NASA Turbofan
        if "unit_id" in self.data.columns:
            from fl.data_nasa import create_sequences_nasa
            X, y = create_sequences_nasa(self.data, self.sensor_cols, seq_len)
        else:
            # Pump
            from fl.data import create_sequences
            X, y = create_sequences(self.data, self.sensor_cols, seq_len)

        if len(X) == 0:
            raise ValueError(
                f"Node {self.node_id}: Not enough data to create sequences "
                f"(len={len(self.data)}, seq_len={seq_len})"
            )

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)

        # ✅ eval_only이면 전체를 evaluation set으로 사용 (split 없음)
        if self.eval_only:
            X_train = X_t[:0]  # empty
            y_train = y_t[:0]
            X_val = X_t
            y_val = y_t
            return NodeDataset(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

        split = int(len(X_t) * self._train_split)
        X_train, y_train = X_t[:split], y_t[:split]
        X_val, y_val = X_t[split:], y_t[split:]

        return NodeDataset(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )

    def _get_loaders(self):
        if self.eval_only:
            raise RuntimeError("This node is eval_only=True; no training loaders available.")

        train_ds = TensorDataset(self.dataset.X_train, self.dataset.y_train)
        val_ds = TensorDataset(self.dataset.X_val, self.dataset.y_val)

        return (
            DataLoader(train_ds, batch_size=self.train_cfg.batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=self.train_cfg.batch_size),
        )

    def get_state_dict(self) -> StateDict:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def set_state_dict(self, state: StateDict) -> None:
        self.model.load_state_dict(state)
        self.model.to(self.device)

    def train_local(self):
        if self.eval_only:
            raise RuntimeError("train_local() called on eval_only=True node")

        train_loader, val_loader = self._get_loaders()

        # ===== TRAIN =====
        self.model.train()
        train_loss_sum, train_total = 0.0, 0

        for epoch in range(self.train_cfg.local_epochs):
            for batch_idx, (xb, yb) in enumerate(train_loader):
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(xb).squeeze(1)
                loss = self.criterion(logits, yb)

                # 누적 (샘플 수 기준 가중 평균)
                bs = int(yb.size(0))
                train_loss_sum += float(loss.item()) * bs
                train_total += bs

                # (선택) 디버그: Node 1 첫 배치 1회만
                if self.debug_first_batch and self.node_id == 1 and epoch == 0 and batch_idx == 0:
                    with torch.no_grad():
                        lg = logits.detach().view(-1)
                        yy = yb.detach().view(-1)
                        print("\n[DEBUG][Node 1] first batch check")
                        print(f"  xb.shape={tuple(xb.shape)} yb.shape={tuple(yb.shape)} logits.shape={tuple(logits.shape)}")
                        print(f"  yb.mean(pos_ratio)={yy.mean().item():.6f} yb.sum={yy.sum().item():.0f}/{yy.numel()}")
                        print(f"  logits: min={lg.min().item():.6f} max={lg.max().item():.6f} mean={lg.mean().item():.6f}")
                        print(f"  loss={loss.item():.12f} isfinite={torch.isfinite(loss).item()}")

                loss.backward()
                self.optimizer.step()

        train_loss = train_loss_sum / max(train_total, 1)

        # ===== VALIDATION =====
        self.model.eval()
        val_loss_sum, val_total = 0.0, 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = self.model(xb).squeeze(1)
                loss = self.criterion(logits, yb)

                bs = int(yb.size(0))
                val_loss_sum += float(loss.item()) * bs
                val_total += bs

        val_loss = val_loss_sum / max(val_total, 1)
        return float(train_loss), float(val_loss), 0.0
