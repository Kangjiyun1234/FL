"""
fl/prepare_data_femto.py — FEMTO PRONOSTIA Bearing 데이터셋 전처리

Kaggle 데이터셋: alanhabrony/ieee-phm-2012-data-challenge
  Learning_set/BearingX_Y/acc_XXXXX.csv  (5 cols, no header)
  Test_set/BearingX_Y/acc_XXXXX.csv      (5 cols, no header)

CSV 컬럼: [hour, min, sec, horizontal_acc, vertical_acc]
샘플/파일: 2560 (25.6kHz × 0.1sec)
단위: g (중력 가속도)

데이터 미존재 시 kaggle 다운로드 명령:
  kaggle datasets download -d alanhabrony/ieee-phm-2012-data-challenge --unzip -p /mnt/d/SESLab/FEMTO

FL 노드 매핑:
  mn1 = Condition 1 (1800rpm, 4000N) → Bearing1_*
  mn2 = Condition 2 (1650rpm, 4200N) → Bearing2_*
  mn3 = Condition 3 (1500rpm, 5000N) → Bearing3_*

Fault onset: 파일 시계열 순서에서 RMS > 1.0g 인 첫 파일 (Bearing2: > 2.0g)
"""
from __future__ import annotations

import os
import sys
import glob
import pickle
import random
import numpy as np
import pandas as pd
from pathlib import Path

# ── 경로 설정 ────────────────────────────────────────────
FEMTO_ROOT = os.environ.get("FEMTO_ROOT", "/mnt/d/SESLab/FEMTO/ieee-phm-2012-data-challenge-dataset-master")
OUT_DIR    = "/tmp/fl_data/femto"

# ── 하이퍼파라미터 ────────────────────────────────────────
RMS_FAULT_THRESHOLD = 1.0    # g  (정상~0.3-0.5g, Bearing_2 최대~2.4g → 1g를 fault onset 기준으로)
SEQ_LEN             = 2560
N_CHANNELS          = 1      # horizontal 채널만 사용 (col index 4)
HORIZONTAL_COL      = 4      # 0-indexed: [hour, min, sec, microsec, h_acc, v_acc]

MAX_NORMAL_TRAIN    = 800    # 노드당 train pool 최대 샘플 수
MAX_NORMAL_VAL      = 100    # 노드당 val_normal 최대 샘플 수
MAX_ANOM_VAL        = 100    # 노드당 val_anomaly 최대 샘플 수
MAX_TEST_NORMAL     = 100    # 노드당 test_stream normal 최대 샘플 수
MAX_TEST_ANOMALY    = 100    # 노드당 test_stream anomaly 최대 샘플 수

SEED = 42

# ── 노드 → Condition 매핑 ─────────────────────────────────
NODE_CONDITIONS = {
    "mn1": 1,   # Bearing1_*
    "mn2": 2,   # Bearing2_*
    "mn3": 3,   # Bearing3_*
}

NODE_INFO = {
    "mn1": "Condition 1 (1800rpm, 4000N)",
    "mn2": "Condition 2 (1650rpm, 4200N)",
    "mn3": "Condition 3 (1500rpm, 5000N)",
}

# ── 노드 역할 (교수님 피드백 반영) ───────────────────────────
# mn1, mn2: 순수 정상 학습 노드 — 글로벌 모델 안정화 담당
# mn3:      고장 전이 노드 — 정상 → 이상 → 고장 흐름을 온전히 보여주는 탐지 대상
NODE_ROLES = {
    "mn1": "normal_support",
    "mn2": "normal_support",
    "mn3": "fault_transition",
}

FULL_TEST_STREAM_NODES = set()  # 모든 노드 동일하게 캡 적용 (MAX_TEST_NORMAL/ANOMALY)


# ── 유틸 함수 ─────────────────────────────────────────────

def read_acc_csv(path: str) -> np.ndarray:
    """acc_XXXXX.csv 파일 읽기 → horizontal 채널 배열 (2560,)
    구분자가 콤마(,) 또는 세미콜론(;)인 파일 모두 지원.
    """
    with open(path) as fp:
        sep = ";" if ";" in fp.readline() else ","
    df = pd.read_csv(path, header=None, sep=sep, usecols=[HORIZONTAL_COL], dtype=np.float32)
    sig = df.iloc[:, 0].values
    if len(sig) >= SEQ_LEN:
        return sig[:SEQ_LEN]
    padded = np.zeros(SEQ_LEN, dtype=np.float32)
    padded[:len(sig)] = sig
    return padded


def compute_rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal ** 2)))


def load_bearing_files(bearing_dir: str) -> list[str]:
    """acc_*.csv 파일을 이름 순(=시계열 순)으로 정렬하여 반환"""
    pattern = os.path.join(bearing_dir, "acc_*.csv")
    return sorted(glob.glob(pattern))


def load_bearing_signals(bearing_dir: str) -> tuple[list[np.ndarray], list[float]]:
    """
    베어링 폴더의 모든 파일을 한 번만 읽어
    신호 리스트와 RMS 리스트를 동시에 반환.
    """
    files = load_bearing_files(bearing_dir)
    signals, rms_list = [], []
    for f in files:
        sig = read_acc_csv(f)
        signals.append(sig)
        rms_list.append(compute_rms(sig))
    return signals, rms_list


def split_normal_anomaly(signals: list[np.ndarray], rms_list: list[float],
                         threshold: float = None):
    """RMS > threshold 인 첫 인덱스를 fault onset으로 설정."""
    thr = threshold if threshold is not None else RMS_FAULT_THRESHOLD
    fault_idx = next(
        (i for i, r in enumerate(rms_list) if r > thr),
        len(signals)
    )
    return signals[:fault_idx], signals[fault_idx:]


def to_array(signals: list[np.ndarray]) -> np.ndarray:
    if not signals:
        return np.empty((0, SEQ_LEN), dtype=np.float32)
    return np.stack(signals, axis=0)


def normalize(signals: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std == 0.0:
        return signals - mean
    return (signals - mean) / std


# ── 메인 처리 ─────────────────────────────────────────────

def process_node(node: str) -> dict:
    condition = NODE_CONDITIONS[node]
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)

    print(f"\n{'='*60}")
    print(f"  노드: {node}  ({NODE_INFO[node]})")
    print(f"  Condition {condition} → Bearing{condition}_*")
    print(f"{'='*60}")

    learn_dir = os.path.join(FEMTO_ROOT, "Learning_set")

    # Bearing2_1은 정상 구간이 33개뿐(빠른 열화) → train+val 부적합
    # mn2만 역방향: Bearing2_2(정상=784) → train+val, Bearing2_1 → test_stream
    if condition == 2:
        train_val_dir = os.path.join(learn_dir, f"Bearing{condition}_2")
        test_src_dir  = os.path.join(learn_dir, f"Bearing{condition}_1")
    else:
        train_val_dir = os.path.join(learn_dir, f"Bearing{condition}_1")
        test_src_dir  = os.path.join(learn_dir, f"Bearing{condition}_2")

    for d in [train_val_dir, test_src_dir]:
        if not os.path.isdir(d):
            raise FileNotFoundError(
                f"디렉토리를 찾을 수 없습니다: {d}\n"
                f"  데이터 다운로드:\n"
                f"    kaggle datasets download -d alanhabrony/ieee-phm-2012-data-challenge "
                f"--unzip -p /mnt/d/SESLab/FEMTO"
            )

    # ── Train / Val ───────────────────────────────────────
    tv_name = os.path.basename(train_val_dir)
    ts_name = os.path.basename(test_src_dir)
    signals_1, rms_1 = load_bearing_signals(train_val_dir)
    normal_1, anomaly_1 = split_normal_anomaly(signals_1, rms_1)
    print(f"  {tv_name} (train+val): 전체={len(signals_1)}  정상={len(normal_1)}  이상={len(anomaly_1)}")

    n_train = int(len(normal_1) * 0.8)
    train_signals   = to_array(normal_1[:n_train])
    val_normal_sig  = to_array(normal_1[n_train:])
    val_anomaly_sig = to_array(anomaly_1)

    # ── 샘플 캡 적용 ─────────────────────────────────────
    def cap_shuffle(arr: np.ndarray, cap: int, rng_np) -> np.ndarray:
        if len(arr) == 0:
            return arr
        idx = np.arange(len(arr))
        rng_np.shuffle(idx)
        return arr[idx[:cap]]

    train_signals   = cap_shuffle(train_signals,   MAX_NORMAL_TRAIN, np_rng)
    val_normal_sig  = cap_shuffle(val_normal_sig,  MAX_NORMAL_VAL,   np_rng)
    val_anomaly_sig = cap_shuffle(val_anomaly_sig, MAX_ANOM_VAL,     np_rng)

    print(f"\n  [BearingX_1 → train/val 풀]")
    print(f"    train pool  : {len(train_signals)}")
    print(f"    val_normal  : {len(val_normal_sig)}")
    print(f"    val_anomaly : {len(val_anomaly_sig)}")

    # ── 정규화 (train 기준) ───────────────────────────────
    norm_mean = float(train_signals.mean()) if len(train_signals) > 0 else 0.0
    norm_std  = float(train_signals.std())  if len(train_signals) > 0 else 1.0
    if norm_std == 0.0:
        norm_std = 1.0

    train_signals   = normalize(train_signals,   norm_mean, norm_std)
    val_normal_sig  = normalize(val_normal_sig,  norm_mean, norm_std)
    val_anomaly_sig = normalize(val_anomaly_sig, norm_mean, norm_std)

    # ── val set 구성 (섞기) ───────────────────────────────
    val_signals = np.concatenate([val_normal_sig, val_anomaly_sig], axis=0)
    val_labels  = np.array(
        [0] * len(val_normal_sig) + [1] * len(val_anomaly_sig),
        dtype=np.int64
    )
    shuffle_idx = np.arange(len(val_signals))
    np_rng.shuffle(shuffle_idx)
    val_signals = val_signals[shuffle_idx]
    val_labels  = val_labels[shuffle_idx]

    # ── Test_stream ───────────────────────────────────────
    # Bearing2_1은 초기 열화 신호가 불규칙 → 명확한 고장 신호(RMS > 2.0g)만 이상으로 정의
    test_threshold = 2.0 if condition == 2 else None
    signals_2, rms_2 = load_bearing_signals(test_src_dir)
    normal_2, anomaly_2 = split_normal_anomaly(signals_2, rms_2, threshold=test_threshold)
    thr_str = f"{test_threshold}g" if test_threshold else f"{RMS_FAULT_THRESHOLD}g"
    print(f"  {ts_name} (test_stream, 이상기준={thr_str}): 전체={len(signals_2)}  정상={len(normal_2)}  이상={len(anomaly_2)}")

    if node in FULL_TEST_STREAM_NODES:
        # 고장 전이 노드: 전체 run-to-failure 시계열을 시간순으로 유지 (캡 없음)
        # 정상 → 이상 전체 흐름을 보여주는 것이 목적
        test_normal_sig  = normalize(to_array(normal_2),  norm_mean, norm_std)
        test_anomaly_sig = normalize(to_array(anomaly_2), norm_mean, norm_std)
        print(f"  → 고장 전이 노드: 전체 시계열 유지 (정상={len(normal_2)}, 이상={len(anomaly_2)})")
    else:
        # 정상 지원 노드: 랜덤 샘플링 (평가 참고용)
        test_normal_sig  = normalize(cap_shuffle(to_array(normal_2),  MAX_TEST_NORMAL,  np_rng), norm_mean, norm_std)
        test_anomaly_sig = normalize(cap_shuffle(to_array(anomaly_2), MAX_TEST_ANOMALY, np_rng), norm_mean, norm_std)

    test_stream_signals = np.concatenate([test_normal_sig, test_anomaly_sig], axis=0)
    test_stream_labels  = np.array(
        [0] * len(test_normal_sig) + [1] * len(test_anomaly_sig),
        dtype=np.int64
    )

    print(f"\n  [Test_set 합계]")
    print(f"    test_normal  : {len(test_normal_sig)}")
    print(f"    test_anomaly : {len(test_anomaly_sig)}")

    print(f"\n  [최종 shape]")
    print(f"    train_signals       : {train_signals.shape}")
    print(f"    val_signals         : {val_signals.shape}")
    print(f"    val_labels          : {val_labels.shape}  (0=정상, 1=이상)")
    print(f"    test_stream_signals : {test_stream_signals.shape}")
    print(f"    test_stream_labels  : {test_stream_labels.shape}")
    print(f"    norm_mean={norm_mean:.4f}  norm_std={norm_std:.4f}")

    return {
        "train_signals":       train_signals,
        "val_signals":         val_signals,
        "val_labels":          val_labels,
        "test_stream_signals": test_stream_signals,
        "test_stream_labels":  test_stream_labels,
        "norm_mean":           norm_mean,
        "norm_std":            norm_std,
        "seq_len":             SEQ_LEN,
        "n_channels":          N_CHANNELS,
        "node":                node,
        "node_role":           NODE_ROLES[node],
        "node_info":           NODE_INFO[node],
        "class_names":         ["정상", "이상"],
        "rms_fault_threshold": RMS_FAULT_THRESHOLD,
    }


def main():
    # 데이터 존재 여부 사전 확인
    if not os.path.isdir(FEMTO_ROOT):
        print(f"[ERROR] FEMTO_ROOT 경로를 찾을 수 없습니다: {FEMTO_ROOT}")
        print()
        print("  데이터 다운로드 명령:")
        print(f"    kaggle datasets download -d alanhabrony/ieee-phm-2012-data-challenge \\")
        print(f"      --unzip -p {FEMTO_ROOT}")
        print()
        print("  또는 환경 변수로 경로 재지정:")
        print(f"    FEMTO_ROOT=/your/path python3 fl/prepare_data_femto.py")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n=== FEMTO PRONOSTIA Bearing 데이터 전처리 ===")
    print(f"  FEMTO_ROOT : {FEMTO_ROOT}")
    print(f"  OUT_DIR    : {OUT_DIR}")
    print(f"  RMS 결함 임계값: {RMS_FAULT_THRESHOLD} g")
    print(f"  SEQ_LEN    : {SEQ_LEN}")

    for node in ["mn1", "mn2", "mn3"]:
        data = process_node(node)
        out_path = os.path.join(OUT_DIR, f"{node}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(data, f)
        print(f"\n  -> 저장 완료: {out_path}")

    print(f"\n{'='*60}")
    print(f"  전처리 완료! 저장 위치: {OUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
