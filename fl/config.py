"""
fl/config.py — FL + oneM2M 통합 설정
FEMTO PRONOSTIA Bearing 데이터셋 (Raw Signal AE 버전)
- 3 edge 노드 (mn1=Condition1/1800rpm, mn2=Condition2/1650rpm, mn3=Condition3/1500rpm)
- Conv1DAE 기반 anomaly detection (정상만 학습)
"""
import os
from dataclasses import dataclass
from typing import Optional


# ════════════════════════════════════════════════════════
# AE 모델 설정
# ════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AEConfig:
    seq_len:    int   = 2560    # vibration: 25.6kHz × 0.1sec (FEMTO PRONOSTIA)
    n_channels: int   = 1       # 진동 1채널
    latent_dim: int   = 32      # AE 병목 차원


# ════════════════════════════════════════════════════════
# 학습 설정
# ════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TrainConfig:
    rounds:       int   = 10
    local_epochs: int   = 10
    batch_size:   int   = 32
    lr:           float = 1e-3
    weight_decay: float = 1e-4
    device:       str   = "cpu"
    seed:         int   = 42
    log_every:    int   = 1
    train_split:  float = 0.8    # 레거시 필드 (AEEdgeNode에서는 미사용)
    hidden_size:  int   = 128    # 레거시 필드
    num_classes:  int   = 2      # 레거시 필드


# ════════════════════════════════════════════════════════
# 데이터 설정
# ════════════════════════════════════════════════════════

FEMTO_DATA_DIR = "/tmp/fl_data/femto"

NODE_DATA_FILES = {
    "mn1": os.path.join(FEMTO_DATA_DIR, "mn1.pkl"),
    "mn2": os.path.join(FEMTO_DATA_DIR, "mn2.pkl"),
    "mn3": os.path.join(FEMTO_DATA_DIR, "mn3.pkl"),
}

CLASS_NAMES  = ["정상", "이상"]
NUM_CLASSES  = 2

RMS_FAULT_THRESHOLD = 1.0    # g — first file exceeding this RMS = fault onset


# ════════════════════════════════════════════════════════
# Anomaly Detection 설정
# ════════════════════════════════════════════════════════

# 탐지 결정: 재구성 오차가 threshold를 K회 연속 초과하면 alarm
ANOMALY_K_CONSECUTIVE = 3

# threshold 결정 방법: val 정상 데이터 MSE 의 N 표준편차 위
THRESHOLD_N_SIGMA = 3.0


# ════════════════════════════════════════════════════════
# oneM2M 연결 설정
# ════════════════════════════════════════════════════════

BASE_URL    = "http://127.0.0.1:3000"
CSE_NAME    = "TinyIoT"
MN_AE_NAME  = "MN-AE-1"
IN_AE_NAME  = "IN-AE"
NOTIFY_HOST = "127.0.0.1"

NUM_CLIENTS   = 3
GLOBAL_ROUNDS = 10    # TRAIN_CFG.rounds 와 동일하게 유지

ORIGINATOR = "CAdmin"
HEADERS = {
    "X-M2M-Origin": ORIGINATOR,
    "X-M2M-RVI":    "2a",
    "Content-Type": "application/json;ty=4",
    "Accept":       "application/json",
}

# ════════════════════════════════════════════════════════
# 모델 저장 경로
# ════════════════════════════════════════════════════════

MODEL_BASE_DIR   = "/tmp/fl_models"
LOCAL_MODEL_DIR  = os.path.join(MODEL_BASE_DIR, "local")
GLOBAL_MODEL_DIR = os.path.join(MODEL_BASE_DIR, "global")

os.makedirs(LOCAL_MODEL_DIR,  exist_ok=True)
os.makedirs(GLOBAL_MODEL_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════
# DP-SGD 설정
# ════════════════════════════════════════════════════════

DP_EPSILON       = 12.0
DP_DELTA         = 5e-4
DP_MAX_GRAD_NORM = 1.5

# ════════════════════════════════════════════════════════
# 인스턴스
# ════════════════════════════════════════════════════════

TRAIN_CFG = TrainConfig()
AE_CFG    = AEConfig()
