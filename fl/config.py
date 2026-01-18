# fl/config.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    """
    데이터 설정.
    - synthetic: 기존 합성 회귀 데이터 생성용 파라미터 사용
    - pump_sensor: CSV 로드 + 시계열 윈도우 생성용 파라미터 사용
    """
    # ===== 데이터셋 선택/경로 =====
    dataset: str = "synthetic"          # "synthetic" | "pump_sensor"
    data_dir: str = "./data"            # pump_sensor일 때 사용 (예: ./data/pump-sensor-data)

    # ===== pump_sensor 시계열 옵션 =====
    sequence_length: int = 50           # 시계열 윈도우 길이
    train_split: float = 0.8            # train/val 분할 비율(노드 내부)

    # ===== synthetic(기존) 옵션 =====
    num_samples_per_node: int = 800     # 노드당 샘플 수
    num_features: int = 10              # 입력 feature 수
    noise_std: float = 0.1              # 노이즈 표준편차
    non_iid: bool = True                # 노드별 분포 차이 유무
    val_ratio: float = 0.2              # (synthetic에서) 검증 비율
    seed: int = 42                      # 데이터 생성 시드


@dataclass(frozen=True)
class TrainConfig:
    """
    학습 설정.
    - 공통 파라미터 + (pump_sensor 분류용) hidden_size/num_classes 포함
    """
    rounds: int = 5
    local_epochs: int = 1
    batch_size: int = 128
    lr: float = 0.01
    weight_decay: float = 0.0
    device: str = "cpu"
    seed: int = 42
    log_every: int = 1
    
    # ===== 데이터 분할 (edge_node에서 사용) =====
    train_split: float = 0.8            # train/val 분할 비율
    sequence_length: int = 50           # 시계열 윈도우 (edge_node._prepare_dataset에서 필요)

    # ===== pump_sensor 분류 모델 옵션 =====
    hidden_size: int = 64               # LSTM hidden size
    num_classes: int = 3                # NORMAL/BROKEN/RECOVERING 등


@dataclass(frozen=True)
class FLJobConfig:
    job_name: str
    num_nodes: int
    data: DataConfig
    training: TrainConfig