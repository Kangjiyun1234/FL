from __future__ import annotations

import os
import pickle
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 0. Settings
# ============================================================

RAW_DIR = Path(
    os.getenv(
        "ISAAC_DATA_DIR",
        "/mnt/c/Projects/bearing_testbed/data",
    )
)
OUT_DIR = Path(
    os.getenv(
        "FL_PKL_DIR",
        "/tmp/fl_data/femto",
    )
)
BACKUP_ROOT = Path(
    os.getenv(
        "FL_PKL_BACKUP_DIR",
        "/tmp/fl_data/femto_backups",
    )
)

SIGNAL_COL = os.getenv("ISAAC_SIGNAL_COL", "accel_y")

TARGET_FS = 25_600
SEQ_LEN = 2_560
TRAIN_STRIDE = 1_280       # 50% overlap
EVAL_STRIDE = 2_560        # non-overlap
TRIM_START_SEC = 0.5
TEST_STREAM_MAX = 160
SEED = 42

MN3_NORMAL_END = 20.0
MN3_FAULT_START = 40.0

# MN2: 정상 분포는 유지하되 MN1과 너무 똑같지 않도록 약한 차이 추가
MN2_BASE_GAIN = 1.06
MN2_MOD_DEPTH = 0.04
MN2_MOD_HZ = 1.2
MN2_NOISE_STD_RATIO = 0.05
MN2_TEST_ANOMALY_COUNT = 10
MN2_IMPULSE_MIN_SIGMA = 3.5
MN2_IMPULSE_MAX_SIGMA = 5.5

# MN3: 기존 90 Hz / 최대 약 6 sigma가 너무 크고 평평해지는 문제 완화
FAULT_IMPACT_HZ = 55.0
FAULT_RESONANCE_HZ = 2_500.0
FAULT_RINGDOWN_SEC = 0.025
FAULT_DECAY = 220.0
MN3_RINGDOWN_BASE_SIGMA = 0.8
MN3_RINGDOWN_GROWTH_SIGMA = 2.0   # p=1일 때 기본 최대 약 2.8 sigma
MN3_EVENT_INTERVAL_JITTER = 0.20
MN3_EVENT_AMPLITUDE_MIN = 0.70
MN3_EVENT_AMPLITUDE_MAX = 1.15
MN3_EVENT_SKIP_PROB = 0.12

REQUIRED_COLUMNS = {
    "time",
    "state",
    "fault_progress",
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
}

NODE_ROLES = {
    "mn1": "normal_reference",
    "mn2": "intermittent_anomaly",
    "mn3": "progressive_fault",
}


# ============================================================
# 1. Raw file discovery / loading
# ============================================================

def find_latest_run():
    pattern = re.compile(r"mn1_raw_(\d{8}_\d{6})\.csv$")
    run_ids = []

    for path in RAW_DIR.glob("mn1_raw_*.csv"):
        match = pattern.match(path.name)
        if match:
            run_ids.append(match.group(1))

    for run_id in sorted(run_ids, reverse=True):
        paths = {
            node: RAW_DIR / f"{node}_raw_{run_id}.csv"
            for node in ("mn1", "mn2", "mn3")
        }

        if all(path.exists() for path in paths.values()):
            return run_id, paths

    raise FileNotFoundError(
        f"완전한 MN1/MN2/MN3 raw CSV 세트를 찾지 못했습니다: {RAW_DIR}"
    )


def load_raw(path):
    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            f"{path.name}: missing columns={sorted(missing)}"
        )

    for col in REQUIRED_COLUMNS - {"state"}:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["time", SIGNAL_COL])
        .sort_values("time")
        .drop_duplicates(subset=["time"])
        .reset_index(drop=True)
    )

    df = df[df["time"] >= TRIM_START_SEC].reset_index(drop=True)

    if len(df) < 10:
        raise ValueError(
            f"{path.name}: 유효한 raw sample이 너무 적습니다."
        )

    return df


def estimate_raw_fs(df):
    t = df["time"].to_numpy(np.float64)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(1.0 / np.median(dt)) if len(dt) else 0.0


def resample_raw(df):
    old_t = df["time"].to_numpy(np.float64)
    old_signal = df[SIGNAL_COL].to_numpy(np.float64)
    old_progress = df["fault_progress"].to_numpy(np.float64)

    new_t = np.arange(
        float(old_t[0]),
        float(old_t[-1]),
        1.0 / TARGET_FS,
        dtype=np.float64,
    )

    signal = np.interp(new_t, old_t, old_signal).astype(np.float32)
    progress = np.interp(new_t, old_t, old_progress).astype(np.float32)
    progress = np.clip(progress, 0.0, 1.0)

    return new_t, signal, progress


# ============================================================
# 2. Hybrid signal helpers
# ============================================================

def ringdown_template(amplitude, rng):
    n = max(8, int(FAULT_RINGDOWN_SEC * TARGET_FS))
    t = np.arange(n, dtype=np.float64) / TARGET_FS

    phase = float(rng.uniform(-0.20, 0.20))

    return (
        amplitude
        * np.exp(-FAULT_DECAY * t)
        * np.sin(2.0 * np.pi * FAULT_RESONANCE_HZ * t + phase)
    ).astype(np.float32)


def add_progressive_fault_component(signal, progress, rng):
    """
    MN3:
    - Isaac 저주파 신호 보존
    - fault_progress에 따라 고주파 ring-down 증가
    - 고장 후반에도 pulse 간격/진폭을 랜덤하게 만들어 plateau 완화
    """
    out = signal.astype(np.float32, copy=True)

    normal_part = signal[progress <= 1e-6]
    sigma = max(
        float(np.std(normal_part if len(normal_part) else signal)),
        1e-6,
    )

    base_interval = max(
        1,
        int(TARGET_FS / FAULT_IMPACT_HZ),
    )

    start = 0

    while start < len(out):
        p = float(progress[start])

        if p > 0.0 and rng.random() >= MN3_EVENT_SKIP_PROB:
            local_scale = float(
                rng.uniform(
                    MN3_EVENT_AMPLITUDE_MIN,
                    MN3_EVENT_AMPLITUDE_MAX,
                )
            )

            # p=1에서 기본 약 2.8 sigma, local_scale까지 포함해도
            # 이전의 6 sigma보다 훨씬 낮고 매 이벤트마다 달라진다.
            amplitude = (
                sigma
                * p
                * (
                    MN3_RINGDOWN_BASE_SIGMA
                    + MN3_RINGDOWN_GROWTH_SIGMA * p
                )
                * local_scale
            )

            ring = ringdown_template(amplitude, rng)
            end = min(len(out), start + len(ring))
            out[start:end] += ring[: end - start]

        jitter = float(
            rng.uniform(
                1.0 - MN3_EVENT_INTERVAL_JITTER,
                1.0 + MN3_EVENT_INTERVAL_JITTER,
            )
        )
        start += max(1, int(base_interval * jitter))

    return out


def add_impulse_to_window(window, amplitude, rng):
    out = window.astype(np.float32, copy=True)
    ring = ringdown_template(amplitude, rng)

    max_start = max(1, len(out) - len(ring))
    start = int(rng.integers(0, max_start))
    end = min(len(out), start + len(ring))

    out[start:end] += ring[: end - start]
    return out


def make_mn2_variant(t, signal, rng):
    """
    MN2는 Isaac 단계에서는 정상 조건을 유지한다.
    후처리에서 정상 범위 내의 약한 domain shift만 추가한다.

    - 전체 gain 약 +6%
    - 1.2 Hz 저주파 amplitude modulation
    - 정상 sigma의 5% 수준 measurement noise
    """
    sigma = max(float(np.std(signal)), 1e-6)

    modulation = (
        1.0
        + MN2_MOD_DEPTH
        * np.sin(2.0 * np.pi * MN2_MOD_HZ * t)
    )

    noise = rng.normal(
        0.0,
        sigma * MN2_NOISE_STD_RATIO,
        size=len(signal),
    )

    return (
        signal * MN2_BASE_GAIN * modulation
        + noise
    ).astype(np.float32)


# ============================================================
# 3. Windowing / anomaly injection
# ============================================================

def window_range(t, signal, progress, start_sec, end_sec, stride):
    start_idx = int(np.searchsorted(t, start_sec, side="left"))
    end_idx = int(np.searchsorted(t, end_sec, side="left"))

    windows = []
    centers = []
    progresses = []

    last_start = end_idx - SEQ_LEN

    if last_start < start_idx:
        return (
            np.empty((0, SEQ_LEN), np.float32),
            np.empty((0,), np.float32),
            np.empty((0,), np.float32),
        )

    for index in range(start_idx, last_start + 1, stride):
        end = index + SEQ_LEN
        windows.append(signal[index:end])
        centers.append((t[index] + t[end - 1]) * 0.5)
        progresses.append(np.mean(progress[index:end]))

    return (
        np.asarray(windows, np.float32),
        np.asarray(centers, np.float32),
        np.asarray(progresses, np.float32),
    )


def build_synthetic_val(
    val_normal,
    rng,
    anomaly_ratio=0.5,
    strength_min=4.0,
    strength_max=5.0,
):
    n_anomaly = max(
        1,
        int(round(len(val_normal) * anomaly_ratio)),
    )

    selected = rng.choice(
        len(val_normal),
        size=min(n_anomaly, len(val_normal)),
        replace=False,
    )

    sigma = max(float(np.std(val_normal)), 1e-6)

    anomalies = np.stack([
        add_impulse_to_window(
            val_normal[index],
            amplitude=(
                sigma
                * float(rng.uniform(strength_min, strength_max))
            ),
            rng=rng,
        )
        for index in selected
    ]).astype(np.float32)

    signals = np.concatenate(
        [val_normal, anomalies],
        axis=0,
    )
    labels = np.concatenate([
        np.zeros(len(val_normal), np.int64),
        np.ones(len(anomalies), np.int64),
    ])

    order = rng.permutation(len(signals))
    return signals[order], labels[order]


def inject_sparse_test_anomalies(windows, rng):
    """
    MN2 test:
    기존 5개 -> 10개 간헐 이상 window.
    각 impulse 강도도 3.5~5.5 sigma 사이에서 다르게 한다.
    """
    out = windows.astype(np.float32, copy=True)
    labels = np.zeros(len(out), np.int64)

    if len(out) == 0:
        return out, labels

    count = min(MN2_TEST_ANOMALY_COUNT, len(out))

    fractions = np.linspace(
        0.08,
        0.92,
        count,
    )

    indices = sorted({
        min(
            len(out) - 1,
            max(
                0,
                int(round((len(out) - 1) * f)),
            ),
        )
        for f in fractions
    })

    sigma = max(float(np.std(out)), 1e-6)

    for index in indices:
        strength = float(
            rng.uniform(
                MN2_IMPULSE_MIN_SIGMA,
                MN2_IMPULSE_MAX_SIGMA,
            )
        )

        out[index] = add_impulse_to_window(
            out[index],
            amplitude=sigma * strength,
            rng=rng,
        )
        labels[index] = 1

    return out, labels


def cap_chronological(
    signals,
    labels,
    times,
    progresses,
    max_items,
):
    if len(signals) <= max_items:
        return signals, labels, times, progresses

    indices = np.unique(
        np.linspace(
            0,
            len(signals) - 1,
            max_items,
            dtype=np.int64,
        )
    )

    return (
        signals[indices],
        labels[indices],
        times[indices],
        progresses[indices],
    )


# ============================================================
# 4. Node-specific dataset construction
# ============================================================

def process_mn1(t, signal, progress, rng):
    train, _, _ = window_range(
        t,
        signal,
        progress,
        TRIM_START_SEC,
        36.0,
        TRAIN_STRIDE,
    )

    val_normal, _, _ = window_range(
        t,
        signal,
        progress,
        36.0,
        48.0,
        EVAL_STRIDE,
    )

    test, test_times, test_progress = window_range(
        t,
        signal,
        progress,
        48.0,
        60.0,
        EVAL_STRIDE,
    )

    val, val_labels = build_synthetic_val(
        val_normal,
        rng,
        anomaly_ratio=0.5,
        strength_min=4.0,
        strength_max=5.0,
    )

    test_labels = np.zeros(len(test), np.int64)

    return (
        train,
        val,
        val_labels,
        test,
        test_labels,
        test_times,
        test_progress,
    )


def process_mn2(t, signal, progress, rng):
    # 같은 정상 운전조건이지만 후처리에서 baseline 자체도 약간 다르게 만든다.
    signal = make_mn2_variant(t, signal, rng)

    train, _, _ = window_range(
        t,
        signal,
        progress,
        TRIM_START_SEC,
        36.0,
        TRAIN_STRIDE,
    )

    val_normal, _, _ = window_range(
        t,
        signal,
        progress,
        36.0,
        48.0,
        EVAL_STRIDE,
    )

    test_normal, test_times, test_progress = window_range(
        t,
        signal,
        progress,
        48.0,
        60.0,
        EVAL_STRIDE,
    )

    val, val_labels = build_synthetic_val(
        val_normal,
        rng,
        anomaly_ratio=0.6,
        strength_min=4.0,
        strength_max=5.5,
    )

    test, test_labels = inject_sparse_test_anomalies(
        test_normal,
        rng,
    )

    return (
        train,
        val,
        val_labels,
        test,
        test_labels,
        test_times,
        test_progress,
    )


def process_mn3(t, signal, progress, rng):
    hybrid = add_progressive_fault_component(
        signal,
        progress,
        rng,
    )

    # v3 수정:
    # test stream을 10초부터 시작하되 train/val과 겹치지 않게 분할한다.
    # 0.5~8.0 train / 8~10 normal val / 10~50 test / 50~52 fault val
    train, _, _ = window_range(
        t,
        hybrid,
        progress,
        TRIM_START_SEC,
        8.0,
        TRAIN_STRIDE,
    )

    val_normal, _, _ = window_range(
        t,
        hybrid,
        progress,
        8.0,
        10.0,
        EVAL_STRIDE,
    )

    val_anomaly, _, _ = window_range(
        t,
        hybrid,
        progress,
        50.0,
        52.0,
        EVAL_STRIDE,
    )

    val = np.concatenate(
        [val_normal, val_anomaly],
        axis=0,
    )
    val_labels = np.concatenate([
        np.zeros(len(val_normal), np.int64),
        np.ones(len(val_anomaly), np.int64),
    ])

    order = rng.permutation(len(val))
    val = val[order]
    val_labels = val_labels[order]

    # 10~20 NORMAL -> 20~40 TRANSITION -> 40~50 FAULT
    test, test_times, test_progress = window_range(
        t,
        hybrid,
        progress,
        10.0,
        50.0,
        EVAL_STRIDE,
    )

    # transition 시작부터 anomaly onset
    test_labels = (
        test_progress > 1e-6
    ).astype(np.int64)

    return (
        train,
        val,
        val_labels,
        *cap_chronological(
            test,
            test_labels,
            test_times,
            test_progress,
            TEST_STREAM_MAX,
        ),
    )


# ============================================================
# 5. Normalize / PKL
# ============================================================

def normalize(signals, mean, std):
    if len(signals) == 0:
        return signals.astype(np.float32)

    return (
        (signals - mean)
        / max(std, 1e-6)
    ).astype(np.float32)


def backup_existing_pkls():
    existing = [
        OUT_DIR / f"{node}.pkl"
        for node in ("mn1", "mn2", "mn3")
        if (OUT_DIR / f"{node}.pkl").exists()
    ]

    if not existing:
        return None

    backup_dir = (
        BACKUP_ROOT
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    backup_dir.mkdir(parents=True, exist_ok=True)

    for path in existing:
        shutil.copy2(path, backup_dir / path.name)

    return backup_dir


def build_dataset(
    node,
    run_id,
    raw_path,
    raw_fs,
    train_raw,
    val_raw,
    val_labels,
    test_raw,
    test_labels,
    test_times,
    test_progress,
):
    if min(
        len(train_raw),
        len(val_raw),
        len(test_raw),
    ) == 0:
        raise ValueError(
            f"{node}: train/val/test 중 빈 데이터가 있습니다."
        )

    norm_mean = float(train_raw.mean())
    norm_std = max(float(train_raw.std()), 1e-6)

    return {
        "train_signals": normalize(
            train_raw,
            norm_mean,
            norm_std,
        ),
        "val_signals": normalize(
            val_raw,
            norm_mean,
            norm_std,
        ),
        "val_labels": val_labels.astype(np.int64),
        "test_stream_signals": normalize(
            test_raw,
            norm_mean,
            norm_std,
        ),
        "test_stream_labels": test_labels.astype(np.int64),
        "test_stream_times": test_times.astype(np.float32),
        "test_stream_progress": test_progress.astype(np.float32),
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "seq_len": SEQ_LEN,
        "n_channels": 1,
        "node": node,
        "node_role": NODE_ROLES[node],
        "class_names": ["정상", "이상"],
        "source": "isaac-sim",
        "raw_run_id": run_id,
        "raw_path": str(raw_path),
        "raw_sampling_rate_measured": raw_fs,
        "target_sampling_rate": TARGET_FS,
        "signal_column": SIGNAL_COL,
        "mn3_normal_end_sec": MN3_NORMAL_END,
        "mn3_fault_start_sec": MN3_FAULT_START,
    }


# ============================================================
# 6. Main
# ============================================================

def main():
    print("\n=== Isaac Sim -> FL PKL v3 ===")
    print(f"RAW_DIR    : {RAW_DIR}")
    print(f"OUT_DIR    : {OUT_DIR}")
    print(f"SIGNAL_COL : {SIGNAL_COL}")
    print(f"TARGET_FS  : {TARGET_FS} Hz")
    print(f"SEQ_LEN    : {SEQ_LEN}")

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Isaac raw 폴더가 없습니다: {RAW_DIR}"
        )

    run_id, raw_paths = find_latest_run()
    print(f"\nlatest complete run: {run_id}")

    backup_dir = backup_existing_pkls()
    if backup_dir:
        print(f"기존 PKL 백업: {backup_dir}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    processors = {
        "mn1": process_mn1,
        "mn2": process_mn2,
        "mn3": process_mn3,
    }

    for i, node in enumerate(
        ("mn1", "mn2", "mn3"),
        start=1,
    ):
        raw_path = raw_paths[node]
        df = load_raw(raw_path)
        raw_fs = estimate_raw_fs(df)

        print(f"\n[{node.upper()}] {raw_path}")
        print(f"  measured raw fs : {raw_fs:.2f} Hz")

        t, signal, progress = resample_raw(df)
        rng = np.random.default_rng(SEED + i)

        result = processors[node](
            t,
            signal,
            progress,
            rng,
        )

        dataset = build_dataset(
            node,
            run_id,
            raw_path,
            raw_fs,
            *result,
        )

        out_path = OUT_DIR / f"{node}.pkl"

        with out_path.open("wb") as fp:
            pickle.dump(
                dataset,
                fp,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        print(f"  train : {dataset['train_signals'].shape}")
        print(
            f"  val   : {dataset['val_signals'].shape} "
            f"(normal={np.sum(dataset['val_labels'] == 0)}, "
            f"anomaly={np.sum(dataset['val_labels'] == 1)})"
        )
        print(
            f"  test  : {dataset['test_stream_signals'].shape} "
            f"(normal={np.sum(dataset['test_stream_labels'] == 0)}, "
            f"anomaly={np.sum(dataset['test_stream_labels'] == 1)})"
        )
        print(f"  saved : {out_path}")

    print(
        "\n완료. ./clean_fl.sh 실행 후 기존 FL 실행 절차로 진행하세요.\n"
    )


if __name__ == "__main__":
    main()
