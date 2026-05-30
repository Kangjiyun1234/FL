"""
fl/edge_node.py — FEMTO PRONOSTIA Bearing 데이터셋용 Edge 노드

AEEdgeNode: Conv1DAE 기반 정상 신호 재구성 학습 (anomaly detection)
  - 입력: raw 시계열 신호 (N, seq_len) numpy array
  - 손실: MSE reconstruction loss (정상 샘플만)
  - FedAvg 집계: get/set_state_dict()

레거시:
  EdgeNode: 통계 피처 기반 MLP 분류기 (이전 버전, 미사용)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import TrainConfig, AEConfig
from model import Conv1DAE

StateDict = dict[str, torch.Tensor]


# ════════════════════════════════════════════════════════
# AE Edge Node
# ════════════════════════════════════════════════════════

class AEEdgeNode:
    """
    Conv1DAE 기반 Edge 노드

    학습: 정상 신호만 사용, MSE 재구성 손실 최소화
    평가: val 세트에서 anomaly score(재구성 오차) 기반 AUROC / threshold 결정
    """

    def __init__(
        self,
        node_id:    int,
        train_signals: np.ndarray,   # (N_train, seq_len) — 정상만
        val_signals:   np.ndarray,   # (N_val,   seq_len)
        val_labels:    np.ndarray,   # (N_val,)  0=정상, 1=이상
        ae_cfg:     AEConfig,
        train_cfg:  TrainConfig,
        device:     str = "cpu",
    ):
        self.node_id   = node_id
        self.device    = device
        self.ae_cfg    = ae_cfg
        self.train_cfg = train_cfg

        # ── 텐서 변환 ──
        X_train = torch.tensor(train_signals, dtype=torch.float32)
        X_val   = torch.tensor(val_signals,   dtype=torch.float32)
        y_val   = torch.tensor(val_labels,    dtype=torch.long)

        # shape: (N, seq_len) → DataLoader에서 unsqueeze
        self.train_ds = TensorDataset(X_train)
        self.val_ds   = TensorDataset(X_val, y_val)

        self.num_train_samples = len(X_train)

        # ── 모델 ──
        self.model = Conv1DAE(
            n_channels = ae_cfg.n_channels,
            latent_dim = ae_cfg.latent_dim,
            seq_len    = ae_cfg.seq_len,
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr           = train_cfg.lr,
            weight_decay = train_cfg.weight_decay,
        )

        self.criterion = nn.MSELoss()

        print(f"    [AEEdgeNode] node={node_id}  train={self.num_train_samples}"
              f"  val={len(X_val)}  latent={ae_cfg.latent_dim}  seq={ae_cfg.seq_len}")

    # ── 가중치 공유 ──

    def get_state_dict(self) -> StateDict:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def set_state_dict(self, state: StateDict) -> None:
        self.model.load_state_dict(state)
        self.model.to(self.device)

    # ── 로컬 학습 ──

    def train_local(
        self,
        dp_epsilon:       Optional[float] = None,
        dp_delta:         float = 1e-5,
        dp_max_grad_norm: float = 1.5,
    ) -> tuple[float, float, float]:
        """
        로컬 AE 학습 (정상 데이터만)

        반환: (train_loss, val_loss, val_auroc)
          val_auroc: 재구성 오차 기반 AUROC (0.5=랜덤, 1.0=완벽)
        """
        use_dp = dp_epsilon is not None
        if use_dp:
            sigma = math.sqrt(2 * math.log(1.25 / dp_delta)) / dp_epsilon

        train_loader = DataLoader(
            self.train_ds,
            batch_size = self.train_cfg.batch_size,
            shuffle    = True,
            drop_last  = False,
        )

        # ── Train ──
        self.model.train()
        train_loss_sum = 0.0
        train_total    = 0

        for _ in range(self.train_cfg.local_epochs):
            for (xb,) in train_loader:
                xb = xb.to(self.device)          # (B, L)
                xb_in = xb.unsqueeze(1)          # (B, 1, L)

                self.optimizer.zero_grad()
                recon = self.model(xb_in)        # (B, 1, L)
                loss  = self.criterion(recon, xb_in)

                bs = int(xb.size(0))
                train_loss_sum += float(loss.item()) * bs
                train_total    += bs

                loss.backward()

                if use_dp:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=dp_max_grad_norm
                    )
                    for param in self.model.parameters():
                        if param.grad is not None:
                            noise = (torch.randn_like(param.grad)
                                     * sigma * dp_max_grad_norm / bs)
                            param.grad += noise

                self.optimizer.step()

        train_loss = train_loss_sum / max(train_total, 1)

        # ── Validation ──
        val_loss, val_auroc = self._evaluate_val()

        return float(train_loss), float(val_loss), float(val_auroc)

    def _evaluate_val(self) -> tuple[float, float]:
        """val 세트 평가: (val_mse_normal, auroc)"""
        val_loader = DataLoader(
            self.val_ds,
            batch_size = self.train_cfg.batch_size,
            shuffle    = False,
        )

        self.model.eval()
        all_errors = []
        all_labels = []
        val_mse_sum   = 0.0
        val_normal_n  = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                xb_in  = xb.unsqueeze(1)           # (B, 1, L)
                recon  = self.model(xb_in)
                errors = ((xb_in - recon) ** 2).mean(dim=(1, 2))  # (B,)

                all_errors.append(errors.cpu())
                all_labels.append(yb.cpu())

                # val_loss = 정상 샘플의 MSE
                normal_mask = (yb == 0)
                if normal_mask.any():
                    val_mse_sum  += errors[normal_mask].sum().item()
                    val_normal_n += normal_mask.sum().item()

        val_loss = val_mse_sum / max(val_normal_n, 1)

        # AUROC 계산
        errors = torch.cat(all_errors).numpy()
        labels = torch.cat(all_labels).numpy()

        auroc = _compute_auroc(errors, labels)

        return val_loss, auroc

    def compute_anomaly_scores(self, signals: np.ndarray) -> np.ndarray:
        """
        signals: (N, seq_len) → anomaly scores (N,) — 재구성 MSE
        임계값 비교에 사용
        """
        X = torch.tensor(signals, dtype=torch.float32).to(self.device)
        X = X.unsqueeze(1)   # (N, 1, L)

        self.model.eval()
        scores = []
        with torch.no_grad():
            for i in range(0, len(X), self.train_cfg.batch_size):
                xb    = X[i: i + self.train_cfg.batch_size]
                recon = self.model(xb)
                err   = ((xb - recon) ** 2).mean(dim=(1, 2))
                scores.append(err.cpu().numpy())

        return np.concatenate(scores)


# ════════════════════════════════════════════════════════
# AUROC 헬퍼
# ════════════════════════════════════════════════════════

def _compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    scores가 높을수록 anomaly에 가깝다고 가정
    labels: 0=정상, 1=이상
    """
    if len(np.unique(labels)) < 2:
        return 0.5   # 단일 클래스면 의미 없음

    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except Exception:
        return 0.5


# ════════════════════════════════════════════════════════
# 레거시 EdgeNode (통계 피처 기반 MLP, 미사용)
# ════════════════════════════════════════════════════════

class EdgeNode:
    """레거시: 통계 피처 벡터 → 분류 (현재 미사용, 인터페이스 유지)"""

    def __init__(self, node_id, data, sensor_cols, train_cfg, device="cpu", **kwargs):
        raise NotImplementedError(
            "EdgeNode (통계피처 MLP) 는 더 이상 사용하지 않습니다. "
            "AEEdgeNode 를 사용하세요."
        )
