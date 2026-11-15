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

# 데모에서 이상 상태를 표시할 노드
ANOMALY_DEMO_NODE = os.getenv(
    "FL_ANOMALY_DEMO_NODE",
    "mn3",
)

# MN3는 기본적으로 Round 7부터 이상 샘플을 표시한다.
ANOMALY_START_ROUND = int(
    os.getenv(
        "FL_ANOMALY_START_ROUND",
        "7",
    )
)


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
# cold start 노드 설정
# ════════════════════════════════════════════════════════

# 여기에 적힌 노드는 '학습에 참여하지 않는 신규 설비'로 취급한다.
# 라운드 명령과 글로벌 모델은 똑같이 받지만 train_local을 돌리지 않고,
# 받은 글로벌 모델로 추론만 해서 가중치 없는 결과를 올린다.
# 따라서 FedAvg에는 들어가지 않는다.
#   예) FL_COLD_START_NODES=mn3
# 비워 두면(기본) 모든 노드가 학습에 참여하는 기존 동작 그대로다.
COLD_START_NODES = [
    n.strip()
    for n in os.getenv("FL_COLD_START_NODES", "").split(",")
    if n.strip()
]

ALL_NODES = [f"mn{i}" for i in range(1, NUM_CLIENTS + 1)]

TRAINING_NODES = [
    n for n in ALL_NODES if n not in COLD_START_NODES
]


# ════════════════════════════════════════════════════════
# oneM2M 가중치 전송 설정
# ════════════════════════════════════════════════════════

# True면 모델 가중치를 contentInstance의 con 안에 base64로 실어 보낸다.
# False면 기존처럼 로컬 파일 경로만 전달한다(같은 PC에서만 동작).
FL_EMBED_WEIGHTS = os.getenv(
    "FL_EMBED_WEIGHTS",
    "1",
) not in ("0", "false", "False")

# 가중치를 fp16으로 줄여 보낼지 여부. 전송량이 절반이 된다.
# 정확도가 민감하면 0으로 끄면 된다.
FL_WEIGHTS_FP16 = os.getenv(
    "FL_WEIGHTS_FP16",
    "1",
) not in ("0", "false", "False")

# contentInstance 하나에 담을 base64 가중치의 상한(바이트).
# 이 값을 넘으면 embed를 포기하고 경로 전달로 되돌아간다.
#
# TinyIoT 실측 기준(httpd.c의 BUF_SIZE / config.h의 MAX_PAYLOAD_SIZE = 65536):
#   60KB CIN -> 생성 성공
#   64KB CIN -> 400 "Invalid JSON format" (버퍼에서 잘려 JSON이 깨짐)
# 그래서 기본값을 48KB로 둔다. 다른 필드와 JSON 오버헤드까지 감안한 여유분이다.
# 이 값을 CSE 한계보다 크게 잡으면 CIN 생성이 실패하고 라운드 전체가 멈춘다.
FL_MAX_CIN_BYTES = int(
    os.getenv(
        "FL_MAX_CIN_BYTES",
        str(48 * 1024),
    )
)

# 가중치가 위 상한을 넘을 때 조각내어 보낼지 여부.
#
# 주의: 현재 기본값은 꺼짐(0)이다.
# 조각을 한 컨테이너에 쌓고 이름으로 되읽는 현재 구현은 TinyIoT에서 동작하지 않는다.
# TinyIoT는 컨테이너의 '최신' contentInstance만 조회할 수 있고, 그보다 오래된 것을
# 이름이나 ol로 조회하면 stack smashing으로 프로세스가 종료된다(2KB CIN 2개로도 재현).
# 조각마다 전용 컨테이너를 두고 각각의 la를 읽는 구조로 바꾸면 해결되며,
# 그 방식은 CSE 수준에서 검증했다(39조각 업로드 0.8s / 회수 0.1s, 라운드 재사용 가능).
# 재구현 전까지는 꺼둔다. 켜면 상한 초과 시 조각 전송을 시도하고,
# 꺼져 있으면 기존 model_path 방식으로 폴백한다.
FL_CHUNK_WEIGHTS = os.getenv(
    "FL_CHUNK_WEIGHTS",
    "0",
) not in ("0", "false", "False")


# ════════════════════════════════════════════════════════
# 설정 인스턴스
# ════════════════════════════════════════════════════════

TRAIN_CFG = TrainConfig()
AE_CFG = AEConfig()