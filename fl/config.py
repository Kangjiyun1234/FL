"""
fl/config.py — FL + oneM2M 통합 설정
FEMTO PRONOSTIA Bearing 데이터셋 (Raw Signal AE 버전)

- 3 edge 노드
  - mn1: Condition 1 / 1800 rpm
  - mn2: Condition 2 / 1650 rpm
  - mn3: Condition 3 / 1500 rpm
- Conv1DAE 기반 anomaly detection
- 정상 데이터만 사용해 Autoencoder 학습
"""

import os
from dataclasses import dataclass


# ════════════════════════════════════════════════════════
# AE 모델 설정
# ════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AEConfig:
    # vibration: 25.6 kHz × 0.1 sec
    seq_len: int = 2560

    # 현재는 수평 진동 1채널 사용
    n_channels: int = 1

    # Autoencoder 병목 차원
    latent_dim: int = 32


# ════════════════════════════════════════════════════════
# 학습 설정
# ════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TrainConfig:
    rounds: int = 10
    local_epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    seed: int = 42
    log_every: int = 1

    # 레거시 필드
    train_split: float = 0.8
    hidden_size: int = 128
    num_classes: int = 2


# ════════════════════════════════════════════════════════
# 데이터 설정
# ════════════════════════════════════════════════════════

# 환경변수로 다른 PKL 디렉터리를 지정할 수 있다.
FEMTO_DATA_DIR = os.getenv(
    "FL_PKL_DIR",
    "/tmp/fl_data/femto",
)

NODE_DATA_FILES = {
    "mn1": os.path.join(FEMTO_DATA_DIR, "mn1.pkl"),
    "mn2": os.path.join(FEMTO_DATA_DIR, "mn2.pkl"),
    "mn3": os.path.join(FEMTO_DATA_DIR, "mn3.pkl"),
}

CLASS_NAMES = ["정상", "이상"]
NUM_CLASSES = 2

# RMS가 처음 이 값을 초과한 파일을 fault onset으로 사용
RMS_FAULT_THRESHOLD = 1.0


# ════════════════════════════════════════════════════════
# Anomaly Detection 설정
# ════════════════════════════════════════════════════════

# 재구성 오차가 threshold를 연속 K회 초과하면 fault 판정
ANOMALY_K_CONSECUTIVE = 3

# threshold = 정상 validation MSE 평균 + N × 표준편차
THRESHOLD_N_SIGMA = 3.0

# Isaac Sim 버전에서는 test_stream의 실제 시간 순서를 그대로 사용한다.
# 따라서 특정 FL Round부터 anomaly를 강제로 표시하는 설정은 사용하지 않는다.


# ════════════════════════════════════════════════════════
# oneM2M 연결 설정
# ════════════════════════════════════════════════════════

BASE_URL = os.getenv(
    "TINYIOT_BASE_URL",
    "http://127.0.0.1:3000",
)

CSE_NAME = os.getenv(
    "TINYIOT_CSE_NAME",
    "TinyIoT",
)

MN_AE_NAME = "MN-AE-1"
IN_AE_NAME = "IN-AE"

NOTIFY_HOST = os.getenv(
    "FL_NOTIFY_HOST",
    "127.0.0.1",
)

NUM_CLIENTS = 3
GLOBAL_ROUNDS = 10

ORIGINATOR = "CAdmin"

HEADERS = {
    "X-M2M-Origin": ORIGINATOR,
    "X-M2M-RVI": "2a",
    "Content-Type": "application/json;ty=4",
    "Accept": "application/json",
}


# ════════════════════════════════════════════════════════
# 모델 저장 경로
# ════════════════════════════════════════════════════════

MODEL_BASE_DIR = os.getenv(
    "FL_MODEL_BASE_DIR",
    "/tmp/fl_models",
)

LOCAL_MODEL_DIR = os.path.join(
    MODEL_BASE_DIR,
    "local",
)

GLOBAL_MODEL_DIR = os.path.join(
    MODEL_BASE_DIR,
    "global",
)

CACHE_MODEL_DIR = os.path.join(
    MODEL_BASE_DIR,
    "cache",
)

# clean_fl.sh가 새로운 데모 실행마다 이 파일의 수정 시각을 갱신한다.
#
# Dashboard는 이 시각보다 오래된 global_round*.pt 파일을
# 이전 실행에서 남은 stale model로 판단하고 사용하지 않는다.
DEMO_RUN_MARKER = os.getenv(
    "FL_DEMO_RUN_MARKER",
    os.path.join(
        MODEL_BASE_DIR,
        ".demo_run_started",
    ),
)

os.makedirs(
    LOCAL_MODEL_DIR,
    exist_ok=True,
)

os.makedirs(
    GLOBAL_MODEL_DIR,
    exist_ok=True,
)

os.makedirs(
    CACHE_MODEL_DIR,
    exist_ok=True,
)


# ════════════════════════════════════════════════════════
# DP-SGD 설정
# ════════════════════════════════════════════════════════

DP_EPSILON = 12.0
DP_DELTA = 5e-4
DP_MAX_GRAD_NORM = 1.5


# ════════════════════════════════════════════════════════
# 설정 인스턴스
# ════════════════════════════════════════════════════════

TRAIN_CFG = TrainConfig()
AE_CFG = AEConfig()