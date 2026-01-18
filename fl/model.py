# fl/model.py
import torch
import torch.nn as nn


class PumpSensorModel(nn.Module):
    """시계열 센서 데이터 이상 탐지 모델"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # BatchNorm은 마지막 hidden state에 적용
        self.bn = nn.BatchNorm1d(hidden_size)

        self.fc = nn.Linear(hidden_size, num_classes)

        self._init_weights()

    def _init_weights(self):
        """가중치 초기화 (Xavier는 2D 이상만 적용)"""
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # LSTM / Linear weight
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)

            # bias는 0
            elif "bias" in name:
                nn.init.zeros_(param)

            # BatchNorm / LayerNorm weight (1D)
            elif "weight" in name and param.dim() == 1:
                nn.init.ones_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)

        # 마지막 timestep
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)

        # BatchNorm은 batch > 1일 때만
        if last_hidden.size(0) > 1:
            last_hidden = self.bn(last_hidden)

        last_hidden = self.dropout(last_hidden)
        out = self.fc(last_hidden)
        return out
