"""
fl/prepare_data_femto_replay.py

FEMTO(PRONOSTIA) 베어링 데이터를 Isaac Sim gateway용 stream 패키지로 만든다.

목표:
- 기존 FEMTO 전체 데이터를 그대로 들고 다니지 않고,
  상태별(normal / degradation / fault) 패턴 pool을 만든다.
- 그 패턴을 변형하여 MN별로 충분한 양의 window stream을 생성한다.
- Isaac Sim은 생성된 mn*_stream.npz를 시간순으로 읽어 화면을 재생하고,
  같은 window_idx를 FL 쪽으로 전달한다.

출력:
  /tmp/fl_data/femto_replay/mn1_stream.npz
  /tmp/fl_data/femto_replay/mn2_stream.npz
  /tmp/fl_data/femto_replay/mn3_stream.npz

주의:
- 여기서는 FL용 pkl을 만들지 않는다.
- score / threshold / fault detected 같은 모델 결과도 만들지 않는다.
- FL에 들어갈 입력 window와 Isaac 시각화용 metadata만 만든다.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 0. Settings
# ============================================================

FEMTO_ROOT = Path(
    os.getenv(
        "FEMTO_ROOT",
        "/mnt/d/SESLab/FEMTO/ieee-phm-2012-data-challenge-dataset-master",
    )
)

OUT_DIR = Path(
    os.getenv(
        "FEMTO_REPLAY_OUT_DIR",
        "/tmp/fl_data/femto_replay",
    )
)

SEQ_LEN = 2_560
SAMPLE_RATE = 25_600
WINDOW_SEC = SEQ_LEN / SAMPLE_RATE

# FEMTO acc csv 기준: [hour, min, sec, microsec, horizontal_acc, vertical_acc]
# 기존 main 전처리와 맞추기 위해 horizontal_acc 사용
SIGNAL_COL = int(
    os.getenv(
        "FEMTO_SIGNAL_COL",
        "4",
    )
)

SEED = int(
    os.getenv(
        "FEMTO_REPLAY_SEED",
        "42",
    )
)

# 기존 main의 train/val 규모를 유지하고,
# dashboard/replay용 test stream은 800개로 크게 맞춘다.
TRAIN_N = int(
    os.getenv(
        "FEMTO_REPLAY_TRAIN_N",
        "800",
    )
)

VAL_NORMAL_N = int(
    os.getenv(
        "FEMTO_REPLAY_VAL_NORMAL_N",
        "100",
    )
)

VAL_ANOMALY_N = int(
    os.getenv(
        "FEMTO_REPLAY_VAL_ANOMALY_N",
        "100",
    )
)

TEST_STREAM_N = int(
    os.getenv(
        "FEMTO_REPLAY_TEST_N",
        "800",
    )
)

# MN3 test stream 구성
MN3_NORMAL_N = int(
    os.getenv(
        "FEMTO_REPLAY_MN3_NORMAL_N",
        "450",
    )
)

MN3_DEGRADATION_N = int(
    os.getenv(
        "FEMTO_REPLAY_MN3_DEGRADATION_N",
        "150",
    )
)

MN3_FAULT_N = TEST_STREAM_N - MN3_NORMAL_N - MN3_DEGRADATION_N

# MN2 test stream 구성: 고장 노드는 아니므로 label은 전부 0
MN2_NORMAL_N = int(
    os.getenv(
        "FEMTO_REPLAY_MN2_NORMAL_N",
        "650",
    )
)

MN2_DEGRADATION_N = TEST_STREAM_N - MN2_NORMAL_N

# Condition 별 RMS fault 기준
# 기존 main에서 Bearing2는 더 높은 기준을 사용하던 흐름을 반영
FAULT_RMS_THRESHOLD = {
    1: 1.0,
    2: 2.0,
    3: 1.0,
}

NODE_CONDITIONS = {
    "mn1": 1,
    "mn2": 2,
    "mn3": 3,
}

NODE_INFO = {
    "mn1": "Condition 1 (1800 rpm, 4000 N)",
    "mn2": "Condition 2 (1650 rpm, 4200 N)",
    "mn3": "Condition 3 (1500 rpm, 5000 N)",
}

NODE_RPM = {
    "mn1": 1800,
    "mn2": 1650,
    "mn3": 1500,
}

NODE_LOAD_N = {
    "mn1": 4000,
    "mn2": 4200,
    "mn3": 5000,
}

STATE_NORMAL = 0
STATE_DEGRADATION = 1
STATE_FAULT = 2

STATE_NAMES = np.array(
    ["NORMAL", "DEGRADATION", "FAULT"],
    dtype="<U16",
)


# ============================================================
# 1. FEMTO loading
# ============================================================

def read_acc_csv(path: str) -> np.ndarray | None:
    """
    FEMTO acc_XXXXX.csv를 1개 window로 읽는다.
    비어 있거나 깨진 csv는 None을 반환해서 skip한다.
    """
    try:
        if os.path.getsize(path) == 0:
            return None

        with open(path, "r", encoding="utf-8", errors="ignore") as fp:
            first = fp.readline()

        if not first.strip():
            return None

        sep = ";" if ";" in first else ","

        df = pd.read_csv(
            path,
            header=None,
            sep=sep,
            usecols=[SIGNAL_COL],
            dtype=np.float32,
            on_bad_lines="skip",
        )

        if df.empty:
            return None

        sig = df.iloc[:, 0].to_numpy(dtype=np.float32)

        if len(sig) == 0:
            return None

        if len(sig) >= SEQ_LEN:
            return sig[:SEQ_LEN].astype(np.float32, copy=False)

        padded = np.zeros(
            SEQ_LEN,
            dtype=np.float32,
        )
        padded[: len(sig)] = sig
        return padded

    except pd.errors.EmptyDataError:
        return None
    except Exception as e:
        print(f"    skip bad csv: {path} ({type(e).__name__}: {e})")
        return None

def compute_rms(signal: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.square(signal, dtype=np.float32),
                dtype=np.float64,
            )
        )
    )


def compute_peak(signal: np.ndarray) -> float:
    return float(
        np.max(
            np.abs(signal)
        )
    )


def bearing_dirs(condition: int) -> List[Path]:
    """
    Learning_set / Test_set에 존재하는 Bearing{condition}_* 폴더를 모두 모은다.
    """
    dirs: List[Path] = []

    for subset in ["Learning_set", "Test_set"]:
        base = FEMTO_ROOT / subset
        dirs.extend(
            Path(p)
            for p in sorted(
                glob.glob(
                    str(base / f"Bearing{condition}_*")
                )
            )
            if Path(p).is_dir()
        )

    return dirs


def load_bearing_dir(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    특정 Bearing 폴더를 시간순 window array로 읽는다.
    빈 csv / 깨진 csv는 건너뛴다.
    """
    files = sorted(
        glob.glob(
            str(path / "acc_*.csv")
        )
    )

    windows = []
    rms_values = []
    peak_values = []
    skipped = 0

    for file_path in files:
        sig = read_acc_csv(file_path)

        if sig is None:
            skipped += 1
            continue

        windows.append(sig)
        rms_values.append(compute_rms(sig))
        peak_values.append(compute_peak(sig))

    if skipped:
        print(f"    skipped empty/bad csv in {path.name}: {skipped}")

    if not windows:
        return (
            np.empty((0, SEQ_LEN), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    return (
        np.stack(windows, axis=0).astype(np.float32),
        np.asarray(rms_values, dtype=np.float32),
        np.asarray(peak_values, dtype=np.float32),
    )

def classify_sequence(
    windows: np.ndarray,
    rms_values: np.ndarray,
    condition: int,
) -> Dict[str, np.ndarray]:
    """
    한 Bearing sequence를 normal / degradation / fault 후보로 나눈다.

    - fault: RMS 기준을 처음 넘은 지점 이후
    - normal: fault 이전의 앞쪽 안정 구간
    - degradation: fault 이전의 후반부 또는 RMS가 상대적으로 커진 구간
    """
    if len(windows) == 0:
        return {
            "normal": windows,
            "degradation": windows,
            "fault": windows,
        }

    threshold = FAULT_RMS_THRESHOLD.get(
        condition,
        1.0,
    )

    fault_idx = next(
        (
            i
            for i, rms in enumerate(rms_values)
            if float(rms) > threshold
        ),
        len(windows),
    )

    pre_fault = windows[:fault_idx]
    post_fault = windows[fault_idx:]

    if len(pre_fault) == 0:
        # 전부 fault처럼 보이는 sequence면 초반 일부를 weak degradation 후보로 둔다.
        return {
            "normal": np.empty((0, SEQ_LEN), dtype=np.float32),
            "degradation": windows[: max(1, len(windows) // 5)],
            "fault": windows,
        }

    # fault 이전 구간에서 앞 70%는 normal, 뒤 30%는 degradation 후보로 둔다.
    split = max(
        1,
        int(len(pre_fault) * 0.70),
    )

    normal = pre_fault[:split]
    degradation = pre_fault[split:]

    # degradation 후보가 너무 적으면 normal 후반부를 조금 가져온다.
    if len(degradation) < 10 and len(pre_fault) >= 10:
        tail = max(
            1,
            int(len(pre_fault) * 0.20),
        )
        degradation = pre_fault[-tail:]

    return {
        "normal": normal.astype(np.float32),
        "degradation": degradation.astype(np.float32),
        "fault": post_fault.astype(np.float32),
    }


def build_condition_pools(condition: int) -> Dict[str, np.ndarray]:
    """
    Condition별 상태 pattern pool 생성.
    """
    normal_parts = []
    degradation_parts = []
    fault_parts = []

    dirs = bearing_dirs(condition)

    if not dirs:
        raise FileNotFoundError(
            f"FEMTO Bearing{condition}_* 폴더를 찾지 못했습니다: {FEMTO_ROOT}"
        )

    print(f"\n[Condition {condition}] pattern source dirs")
    for directory in dirs:
        windows, rms_values, _ = load_bearing_dir(directory)
        parts = classify_sequence(
            windows,
            rms_values,
            condition,
        )

        print(
            f"  {directory.name:14s} "
            f"normal={len(parts['normal']):4d}  "
            f"degradation={len(parts['degradation']):4d}  "
            f"fault={len(parts['fault']):4d}"
        )

        if len(parts["normal"]):
            normal_parts.append(parts["normal"])
        if len(parts["degradation"]):
            degradation_parts.append(parts["degradation"])
        if len(parts["fault"]):
            fault_parts.append(parts["fault"])

    def concat(parts: List[np.ndarray]) -> np.ndarray:
        if not parts:
            return np.empty(
                (0, SEQ_LEN),
                dtype=np.float32,
            )
        return np.concatenate(
            parts,
            axis=0,
        ).astype(np.float32)

    pools = {
        "normal": concat(normal_parts),
        "degradation": concat(degradation_parts),
        "fault": concat(fault_parts),
    }

    return pools


# ============================================================
# 2. Pattern sampling / augmentation
# ============================================================

def ensure_pool(
    pools: Dict[str, np.ndarray],
    key: str,
    fallback: np.ndarray,
) -> np.ndarray:
    arr = pools.get(
        key,
        np.empty((0, SEQ_LEN), dtype=np.float32),
    )

    if len(arr) > 0:
        return arr

    if len(fallback) > 0:
        return fallback

    raise ValueError(
        f"'{key}' pattern pool이 비어 있고 fallback도 없습니다."
    )


def augment_window(
    window: np.ndarray,
    rng: np.random.Generator,
    mode: str,
    weak: bool = False,
) -> np.ndarray:
    """
    같은 FEMTO pattern을 그대로 반복하지 않도록 약한 변형을 준다.
    """
    out = window.astype(
        np.float32,
        copy=True,
    )

    if mode == "normal":
        gain = rng.uniform(0.96, 1.04)
        noise_ratio = 0.008
        max_shift = 24
    elif mode == "degradation":
        if weak:
            gain = rng.uniform(0.98, 1.08)
            noise_ratio = 0.010
            max_shift = 32
        else:
            gain = rng.uniform(1.02, 1.16)
            noise_ratio = 0.014
            max_shift = 48
    elif mode == "fault":
        gain = rng.uniform(1.00, 1.22)
        noise_ratio = 0.012
        max_shift = 64
    else:
        gain = 1.0
        noise_ratio = 0.0
        max_shift = 0

    if max_shift > 0:
        shift = int(
            rng.integers(
                -max_shift,
                max_shift + 1,
            )
        )
        out = np.roll(
            out,
            shift,
        )

    sigma = max(
        float(np.std(out)),
        1e-6,
    )

    noise = rng.normal(
        0.0,
        sigma * noise_ratio,
        size=out.shape,
    ).astype(np.float32)

    out = (
        out * float(gain)
        + noise
    ).astype(np.float32)

    # DC drift 방지
    out = (
        out
        - np.mean(out, dtype=np.float64).astype(np.float32)
    ).astype(np.float32)

    return out


def sample_patterns(
    pool: np.ndarray,
    count: int,
    rng: np.random.Generator,
    mode: str,
    weak: bool = False,
) -> np.ndarray:
    if count <= 0:
        return np.empty(
            (0, SEQ_LEN),
            dtype=np.float32,
        )

    if len(pool) == 0:
        raise ValueError(
            f"{mode} pool이 비어 있어 {count}개를 만들 수 없습니다."
        )

    indices = rng.integers(
        0,
        len(pool),
        size=count,
    )

    return np.stack(
        [
            augment_window(
                pool[int(index)],
                rng,
                mode=mode,
                weak=weak,
            )
            for index in indices
        ],
        axis=0,
    ).astype(np.float32)


# ============================================================
# 3. Stream metadata
# ============================================================

def build_visual_metadata(
    windows: np.ndarray,
    labels: np.ndarray,
    state_codes: np.ndarray,
) -> Dict[str, np.ndarray]:
    rms = np.asarray(
        [
            compute_rms(w)
            for w in windows
        ],
        dtype=np.float32,
    )

    peak = np.asarray(
        [
            compute_peak(w)
            for w in windows
        ],
        dtype=np.float32,
    )

    if len(rms) == 0:
        scaled = np.empty((0,), dtype=np.float32)
    else:
        lo = float(
            np.percentile(
                rms,
                5,
            )
        )
        hi = float(
            np.percentile(
                rms,
                95,
            )
        )
        denom = max(
            hi - lo,
            1e-6,
        )

        scaled = np.clip(
            (rms - lo) / denom,
            0.0,
            1.0,
        ).astype(np.float32)

    # Isaac Sim 화면에서 housing 흔들림 등에 쓸 대표 amplitude
    visual_amp = (
        0.01
        + 0.09 * scaled
    ).astype(np.float32)

    # 고장 상태는 조금 더 눈에 띄게
    visual_amp = np.where(
        state_codes == STATE_FAULT,
        np.minimum(
            visual_amp * 1.25,
            0.14,
        ),
        visual_amp,
    ).astype(np.float32)

    times = (
        np.arange(
            len(windows),
            dtype=np.float32,
        )
        * WINDOW_SEC
    ).astype(np.float32)

    progress = np.zeros(
        len(windows),
        dtype=np.float32,
    )

    degradation_idx = np.where(
        state_codes == STATE_DEGRADATION
    )[0]

    fault_idx = np.where(
        state_codes == STATE_FAULT
    )[0]

    if len(degradation_idx):
        progress[degradation_idx] = np.linspace(
            0.15,
            0.65,
            len(degradation_idx),
            dtype=np.float32,
        )

    if len(fault_idx):
        progress[fault_idx] = np.linspace(
            0.75,
            1.0,
            len(fault_idx),
            dtype=np.float32,
        )

    return {
        "rms": rms,
        "peak": peak,
        "visual_amp": visual_amp,
        "times": times,
        "progress": progress,
        "labels": labels.astype(np.int64),
        "state_codes": state_codes.astype(np.int64),
    }


def make_val_set(
    normal_pool: np.ndarray,
    fault_pool: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    val_normal = sample_patterns(
        normal_pool,
        VAL_NORMAL_N,
        rng,
        mode="normal",
    )

    val_fault = sample_patterns(
        fault_pool,
        VAL_ANOMALY_N,
        rng,
        mode="fault",
    )

    val_windows = np.concatenate(
        [val_normal, val_fault],
        axis=0,
    ).astype(np.float32)

    val_labels = np.asarray(
        [0] * len(val_normal)
        + [1] * len(val_fault),
        dtype=np.int64,
    )

    order = rng.permutation(
        len(val_windows)
    )

    return (
        val_windows[order],
        val_labels[order],
    )


def build_node_stream(
    node: str,
    pools: Dict[str, np.ndarray],
    global_fault_pool: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    normal_pool = ensure_pool(
        pools,
        "normal",
        fallback=pools.get("degradation", global_fault_pool),
    )

    degradation_pool = ensure_pool(
        pools,
        "degradation",
        fallback=normal_pool,
    )

    fault_pool = ensure_pool(
        pools,
        "fault",
        fallback=global_fault_pool,
    )

    if node == "mn1":
        train_windows = sample_patterns(
            normal_pool,
            TRAIN_N,
            rng,
            mode="normal",
        )

        test_windows = sample_patterns(
            normal_pool,
            TEST_STREAM_N,
            rng,
            mode="normal",
        )

        test_labels = np.zeros(
            TEST_STREAM_N,
            dtype=np.int64,
        )

        state_codes = np.full(
            TEST_STREAM_N,
            STATE_NORMAL,
            dtype=np.int64,
        )

        node_role = "normal_reference"

    elif node == "mn2":
        # train에도 약한 열화 패턴을 조금 섞어
        # MN1과 완전히 같은 정상 node처럼 보이지 않게 한다.
        n_deg_train = max(
            1,
            int(TRAIN_N * 0.10),
        )
        n_norm_train = TRAIN_N - n_deg_train

        train_normal = sample_patterns(
            normal_pool,
            n_norm_train,
            rng,
            mode="normal",
        )

        train_degradation = sample_patterns(
            degradation_pool,
            n_deg_train,
            rng,
            mode="degradation",
            weak=True,
        )

        train_windows = np.concatenate(
            [train_normal, train_degradation],
            axis=0,
        ).astype(np.float32)

        order = rng.permutation(
            len(train_windows)
        )
        train_windows = train_windows[order]

        test_normal = sample_patterns(
            normal_pool,
            MN2_NORMAL_N,
            rng,
            mode="normal",
        )

        test_degradation = sample_patterns(
            degradation_pool,
            MN2_DEGRADATION_N,
            rng,
            mode="degradation",
            weak=True,
        )

        test_windows = np.concatenate(
            [test_normal, test_degradation],
            axis=0,
        ).astype(np.float32)

        # MN2는 고장 target이 아니므로 약한 열화도 label 0으로 둔다.
        test_labels = np.zeros(
            TEST_STREAM_N,
            dtype=np.int64,
        )

        state_codes = np.asarray(
            [STATE_NORMAL] * MN2_NORMAL_N
            + [STATE_DEGRADATION] * MN2_DEGRADATION_N,
            dtype=np.int64,
        )

        node_role = "normal_with_weak_degradation"

    elif node == "mn3":
        # AE가 고장을 정상처럼 학습하지 않도록 train은 정상만 사용한다.
        train_windows = sample_patterns(
            normal_pool,
            TRAIN_N,
            rng,
            mode="normal",
        )

        test_normal = sample_patterns(
            normal_pool,
            MN3_NORMAL_N,
            rng,
            mode="normal",
        )

        test_degradation = sample_patterns(
            degradation_pool,
            MN3_DEGRADATION_N,
            rng,
            mode="degradation",
            weak=False,
        )

        test_fault = sample_patterns(
            fault_pool,
            MN3_FAULT_N,
            rng,
            mode="fault",
        )

        test_windows = np.concatenate(
            [
                test_normal,
                test_degradation,
                test_fault,
            ],
            axis=0,
        ).astype(np.float32)

        test_labels = np.asarray(
            [0] * (MN3_NORMAL_N + MN3_DEGRADATION_N)
            + [1] * MN3_FAULT_N,
            dtype=np.int64,
        )

        state_codes = np.asarray(
            [STATE_NORMAL] * MN3_NORMAL_N
            + [STATE_DEGRADATION] * MN3_DEGRADATION_N
            + [STATE_FAULT] * MN3_FAULT_N,
            dtype=np.int64,
        )

        node_role = "fault_transition_target"

    else:
        raise ValueError(
            f"Unknown node: {node}"
        )

    val_windows, val_labels = make_val_set(
        normal_pool,
        fault_pool,
        rng,
    )

    # 정규화 기준은 train만으로 계산한다.
    # 실제 정규화 적용은 Isaac Sim 이후 MN-AE 쪽에서 수행할 수 있도록
    # 여기서는 mean/std만 같이 저장한다.
    norm_mean = np.float32(
        np.mean(
            train_windows,
            dtype=np.float64,
        )
    )

    norm_std = np.float32(
        max(
            float(
                np.std(
                    train_windows,
                    dtype=np.float64,
                )
            ),
            1e-6,
        )
    )

    visual = build_visual_metadata(
        test_windows,
        test_labels,
        state_codes,
    )

    fault_indices = np.where(
        test_labels == 1
    )[0]

    fault_onset_idx = (
        int(fault_indices[0])
        if len(fault_indices)
        else -1
    )

    return {
        "train_windows": train_windows.astype(np.float32),
        "val_windows": val_windows.astype(np.float32),
        "val_labels": val_labels.astype(np.int64),
        "test_stream_windows": test_windows.astype(np.float32),
        "test_stream_labels": visual["labels"].astype(np.int64),
        "test_stream_state_codes": visual["state_codes"].astype(np.int64),
        "test_stream_times": visual["times"].astype(np.float32),
        "test_stream_progress": visual["progress"].astype(np.float32),
        "test_stream_rms": visual["rms"].astype(np.float32),
        "test_stream_peak": visual["peak"].astype(np.float32),
        "test_stream_visual_amp": visual["visual_amp"].astype(np.float32),
        "norm_mean": np.asarray(norm_mean, dtype=np.float32),
        "norm_std": np.asarray(norm_std, dtype=np.float32),
        "seq_len": np.asarray(SEQ_LEN, dtype=np.int64),
        "sample_rate": np.asarray(SAMPLE_RATE, dtype=np.int64),
        "window_sec": np.asarray(WINDOW_SEC, dtype=np.float32),
        "node": np.asarray(node),
        "node_role": np.asarray(node_role),
        "node_info": np.asarray(NODE_INFO[node]),
        "rpm": np.asarray(NODE_RPM[node], dtype=np.int64),
        "load_N": np.asarray(NODE_LOAD_N[node], dtype=np.int64),
        "state_names": STATE_NAMES,
        "fault_onset_idx": np.asarray(fault_onset_idx, dtype=np.int64),
    }


# ============================================================
# 4. Save
# ============================================================

def save_stream_npz(
    node: str,
    data: Dict[str, np.ndarray],
) -> Path:
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path = OUT_DIR / f"{node}_stream.npz"

    np.savez_compressed(
        out_path,
        **data,
    )

    return out_path


def print_summary(
    node: str,
    data: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    labels = data["test_stream_labels"]
    states = data["test_stream_state_codes"]

    n_normal_state = int(
        np.sum(states == STATE_NORMAL)
    )
    n_deg_state = int(
        np.sum(states == STATE_DEGRADATION)
    )
    n_fault_state = int(
        np.sum(states == STATE_FAULT)
    )

    n_label0 = int(
        np.sum(labels == 0)
    )
    n_label1 = int(
        np.sum(labels == 1)
    )

    print(f"\n[{node.upper()}] saved: {out_path}")
    print(f"  role      : {str(data['node_role'])}")
    print(f"  train     : {data['train_windows'].shape}")
    print(
        f"  val       : {data['val_windows'].shape} "
        f"(normal={int(np.sum(data['val_labels'] == 0))}, "
        f"anomaly={int(np.sum(data['val_labels'] == 1))})"
    )
    print(
        f"  test      : {data['test_stream_windows'].shape} "
        f"(label0={n_label0}, label1={n_label1})"
    )
    print(
        f"  states    : "
        f"NORMAL={n_normal_state}, "
        f"DEGRADATION={n_deg_state}, "
        f"FAULT={n_fault_state}"
    )
    print(f"  onset_idx : {int(data['fault_onset_idx'])}")
    print(
        f"  norm      : mean={float(data['norm_mean']):.6f}, "
        f"std={float(data['norm_std']):.6f}"
    )


# ============================================================
# 5. Main
# ============================================================

def main() -> None:
    print("\n=== FEMTO Pattern Replay Preprocess ===")
    print(f"FEMTO_ROOT : {FEMTO_ROOT}")
    print(f"OUT_DIR    : {OUT_DIR}")
    print(f"SEQ_LEN    : {SEQ_LEN}")
    print(f"SAMPLE_RATE: {SAMPLE_RATE} Hz")
    print(f"TRAIN_N    : {TRAIN_N}")
    print(f"VAL        : normal={VAL_NORMAL_N}, anomaly={VAL_ANOMALY_N}")
    print(f"TEST_N     : {TEST_STREAM_N}")

    if not FEMTO_ROOT.exists():
        print(f"\n[ERROR] FEMTO_ROOT를 찾지 못했습니다: {FEMTO_ROOT}")
        print("환경변수로 경로를 지정하세요.")
        print(
            "예: FEMTO_ROOT=/your/femto/root "
            "python3 fl/prepare_data_femto_replay.py"
        )
        sys.exit(1)

    rng = np.random.default_rng(
        SEED,
    )

    condition_pools: Dict[int, Dict[str, np.ndarray]] = {}

    for condition in [1, 2, 3]:
        pools = build_condition_pools(
            condition,
        )
        condition_pools[condition] = pools

        print(
            f"  -> pool totals: "
            f"normal={len(pools['normal'])}, "
            f"degradation={len(pools['degradation'])}, "
            f"fault={len(pools['fault'])}"
        )

    # fallback용 전체 fault pool
    global_fault_parts = [
        condition_pools[c]["fault"]
        for c in condition_pools
        if len(condition_pools[c]["fault"]) > 0
    ]

    if global_fault_parts:
        global_fault_pool = np.concatenate(
            global_fault_parts,
            axis=0,
        ).astype(np.float32)
    else:
        raise ValueError(
            "전체 condition에서 fault pattern을 찾지 못했습니다."
        )

    print("\n=== Build node streams ===")

    for node in ["mn1", "mn2", "mn3"]:
        condition = NODE_CONDITIONS[node]

        data = build_node_stream(
            node,
            condition_pools[condition],
            global_fault_pool,
            rng,
        )

        out_path = save_stream_npz(
            node,
            data,
        )

        print_summary(
            node,
            data,
            out_path,
        )

    print("\n완료.")
    print("Isaac Sim gateway는 아래 파일을 읽어 시간순으로 재생/전달하면 됩니다.")
    print(f"  {OUT_DIR}/mn1_stream.npz")
    print(f"  {OUT_DIR}/mn2_stream.npz")
    print(f"  {OUT_DIR}/mn3_stream.npz\n")


if __name__ == "__main__":
    main()
