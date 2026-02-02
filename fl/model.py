# fl/model.py

import torch
import torch.nn as nn


class TurbofanLSTM(nn.Module):
    """
    NASA Turbofan용 LSTM 모델
    시계열 전체를 활용하여 RUL/Maintenance 예측
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_classes: int = 2,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

        self._init_weights()

    def _init_weights(self):
        """가중치 초기화"""
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # LSTM weight
            if "lstm" in name and "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "lstm" in name and "weight_hh" in name:
                nn.init.orthogonal_(param)
            
            # Linear weight
            elif "fc" in name and "weight" in name:
                nn.init.xavier_uniform_(param)

            # Bias
            elif "bias" in name:
                nn.init.constant_(param, 0.01)

            # BatchNorm
            elif "weight" in name and param.dim() == 1:
                nn.init.ones_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size) = (batch, 50, 14)
        
        🔥 전체 시퀀스 활용!
        """
        # LSTM: 모든 타임스텝 처리
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size)

        # 마지막 타임스텝의 hidden state 사용
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)

        # BatchNorm (batch > 1일 때만)
        if last_hidden.size(0) > 1:
            last_hidden = self.bn(last_hidden)

        last_hidden = self.dropout(last_hidden)
        out = self.fc(last_hidden)  # (batch, num_classes)
        
        return out


class PumpMaintenanceClassifier(nn.Module):
    """
    펌프 유지보수 예측 분류기 (MLP)
    입력: 6개 센서 값 (Temperature, Vibration, Pressure, Flow_Rate, RPM, Operational_Hours)
    출력: 2개 클래스 (정상=0, 유지보수 필요=1)
    """

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_size, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 2
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 3
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            
            # Output
            nn.Linear(hidden_size // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming 초기화 (ReLU에 최적)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size) 또는 (batch, input_size)
        """
        # 3D 텐서면 중간 차원 제거
        if x.dim() == 3:
            if x.size(1) == 1:
                x = x.squeeze(1)  # (batch, 1, 6) → (batch, 6)
            else:
                # seq_len > 1이면 마지막 시점만 사용 (backward compatibility)
                x = x[:, -1, :]
        
        return self.network(x)

class PumpSensorModel(nn.Module):
    """시계열용 LSTM 모델 (sequence_length > 1)"""

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
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if "lstm" in name and "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "lstm" in name and "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "fc" in name and "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.01)
            elif "weight" in name and param.dim() == 1:
                nn.init.ones_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]

        if last_hidden.size(0) > 1:
            last_hidden = self.bn(last_hidden)

        last_hidden = self.dropout(last_hidden)
        out = self.fc(last_hidden)
        return out


class SimpleMLP(nn.Module):
    """
    시퀀스가 아닌 단일 시점 데이터용 MLP
    sequence_length=1일 때 사용
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, input_size) → squeeze
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)  # (batch, input_size)
        
        return self.network(x)