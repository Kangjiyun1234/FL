# fl/model.py
# FEMTO PRONOSTIA Bearing 데이터셋용 모델

import torch
import torch.nn as nn


# ════════════════════════════════════════════════════════
# Conv1D Autoencoder (raw 시계열 anomaly detection용)
# ════════════════════════════════════════════════════════

class Conv1DAE(nn.Module):
    """
    1D CNN Autoencoder — 정상 신호 재구성 학습, 재구성 오차로 anomaly 탐지

    입력:  (batch, n_channels, seq_len)
    출력:  (batch, n_channels, seq_len) — 재구성 신호

    seq_len에 무관하게 동작 (인코더 출력 크기를 동적으로 계산).

    동작 확인된 seq_len:
      seq_len=2000  (current  2kHz × 1sec)
      seq_len=12000 (vibration 4kHz × 3sec)
    """

    def __init__(
        self,
        n_channels: int = 1,
        latent_dim: int = 64,
        seq_len:    int = 12000,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.latent_dim = latent_dim
        self.seq_len    = seq_len

        # ── Encoder conv ──
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=16, stride=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=8, stride=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=4, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # 인코더 각 Conv1d 레이어의 출력 크기를 기록 (output_padding 역산용)
        enc_sizes = [seq_len]   # [seq_len, L1, L2, L3]
        with torch.no_grad():
            x = torch.zeros(1, n_channels, seq_len)
            for layer in self.encoder_conv:
                x = layer(x)
                if isinstance(layer, nn.Conv1d):
                    enc_sizes.append(int(x.shape[2]))

        self._enc_ch   = int(x.shape[1])   # 128
        self._enc_len  = int(x.shape[2])   # L3
        self._enc_flat = self._enc_ch * self._enc_len

        # ── Encoder FC ──
        self.encoder_fc = nn.Linear(self._enc_flat, latent_dim)

        # ── Decoder FC ──
        self.decoder_fc = nn.Linear(latent_dim, self._enc_flat)

        # ── Decoder ConvTranspose: output_padding = target - (in-1)*stride - kernel ──
        # enc_sizes = [seq_len, L1, L2, L3]
        # decoder layer1: L3 → L2,  layer2: L2 → L1,  layer3: L1 → seq_len
        targets = list(reversed(enc_sizes[:-1]))   # [L2, L1, seq_len]
        inputs  = list(reversed(enc_sizes[1:]))    # [L3, L2, L1]
        specs   = [(4, 2), (8, 4), (16, 4)]        # (kernel, stride) 각 레이어

        ops = []
        for (k, s), in_len, tgt in zip(specs, inputs, targets):
            op = tgt - (in_len - 1) * s - k
            assert 0 <= op < s, (
                f"output_padding={op} out of range for stride={s} "
                f"(in={in_len}, target={tgt}, k={k})"
            )
            ops.append(op)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4,  stride=2, output_padding=ops[0]),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.ConvTranspose1d(64, 32,  kernel_size=8,  stride=4, output_padding=ops[1]),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.ConvTranspose1d(32, n_channels, kernel_size=16, stride=4, output_padding=ops[2]),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, L) → latent: (B, latent_dim)"""
        h = self.encoder_conv(x)
        h = h.flatten(1)
        return self.encoder_fc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → (B, C, L)"""
        h = self.decoder_fc(z)
        h = h.view(-1, self._enc_ch, self._enc_len)
        return self.decoder_conv(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, L)  또는  (B, L) — 1채널 생략 허용
        반환: (B, C, L)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.decode(self.encode(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L) or (B, C, L) → 샘플별 MSE (B,)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        recon = self.forward(x)
        return ((x - recon) ** 2).mean(dim=(1, 2))


# ════════════════════════════════════════════════════════
# 레거시 MLP (미사용)
# ════════════════════════════════════════════════════════

class MachineryMLP(nn.Module):
    """레거시: 통계 피처 벡터 → 5클래스 분류 (현재 미사용)"""

    def __init__(self, input_size, hidden_size=128, num_classes=5, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2), nn.ReLU(), nn.Dropout(dropout / 2),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.network(x)
