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

RAW_DIR = Path(os.getenv("ISAAC_DATA_DIR", "/mnt/c/Projects/bearing_testbed/data"))
OUT_DIR = Path(os.getenv("FL_PKL_DIR", "/tmp/fl_data/femto"))
BACKUP_ROOT = Path(os.getenv("FL_PKL_BACKUP_DIR", "/tmp/fl_data/femto_backups"))

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

FAULT_IMPACT_HZ = 90.0
FAULT_RESONANCE_HZ = 2_500.0
FAULT_RINGDOWN_SEC = 0.025
FAULT_DECAY = 220.0

REQUIRED_COLUMNS = {
    "time", "state", "fault_progress",
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z",
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
        raise ValueError(f"{path.name}: missing columns={sorted(missing)}")

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
        raise ValueError(f"{path.name}: 유효한 raw sample이 너무 적습니다.")

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

    amp_jitter = float(rng.uniform(0.9, 1.1))
    phase = float(rng.uniform(-0.15, 0.15))

    return (
        amplitude
        * amp_jitter
        * np.exp(-FAULT_DECAY * t)
        * np.sin(2.0 * np.pi * FAULT_RESONANCE_HZ * t + phase)
    ).astype(np.float32)


def add_progressive_fault_component(signal, progress, rng):
    """
    MN3:
    Isaac 저주파 진동은 보존하고 fault_progress가 증가할수록
    고주파 ring-down 성분만 보강한다.
    """
    out = signal.copy()

    normal_part = signal[progress <= 1e-6]
    sigma = max(float(np.std(normal_part if len(normal_part) else signal)), 1e-6)

    interval = max(1, int(TARGET_FS / FAULT_IMPACT_HZ))

    for start in range(0, len(out), interval):
        p = float(progress[start])
        if p <= 0.0:
            continue

        amplitude = sigma * (1.0 + 5.0 * p) * p
        ring = ringdown_template(amplitude, rng)

        end = min(len(out), start + len(ring))
        out[start:end] += ring[: end - start]

    return out


def add_impulse_to_window(window, amplitude, rng):
    out = window.astype(np.float32, copy=True)
    ring = ringdown_template(amplitude, rng)

    max_start = max(1, len(out) - len(ring))
    start = int(rng.integers(0, max_start))
    end = min(len(out), start + len(ring))

    out[start:end] += ring[: end - start]
    return out


# ============================================================
# 3. Windowing / anomaly injection
# ============================================================

def window_range(t, signal, progress, start_sec, end_sec, stride):
    start_idx = int(np.searchsorted(t, start_sec, side="left"))
    end_idx = int(np.searchsorted(t, end_sec, side="left"))

    windows, centers, progresses = [], [], []
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


def build_synthetic_val(val_normal, rng, anomaly_ratio=0.5, strength=5.0):
    n_anomaly = max(1, int(round(len(val_normal) * anomaly_ratio)))
    selected = rng.choice(
        len(val_normal),
        size=min(n_anomaly, len(val_normal)),
        replace=False,
    )

    sigma = max(float(np.std(val_normal)), 1e-6)
    anomalies = np.stack([
        add_impulse_to_window(
            val_normal[index],
            amplitude=sigma * strength,
            rng=rng,
        )
        for index in selected
    ]).astype(np.float32)

    signals = np.concatenate([val_normal, anomalies], axis=0)
    labels = np.concatenate([
        np.zeros(len(val_normal), np.int64),
        np.ones(len(anomalies), np.int64),
    ])

    order = rng.permutation(len(signals))
    return signals[order], labels[order]


def inject_sparse_test_anomalies(windows, rng):
    """
    MN2:
    정상 test stream 중 떨어진 5개 window에만 순간 impulse 추가.
    """
    out = windows.astype(np.float32, copy=True)
    labels = np.zeros(len(out), np.int64)

    if len(out) == 0:
        return out, labels

    fractions = (0.15, 0.32, 0.50, 0.68, 0.85)
    indices = sorted({
        min(len(out) - 1, max(0, int(round((len(out) - 1) * f))))
        for f in fractions
    })

    sigma = max(float(np.std(out)), 1e-6)

    for index in indices:
        out[index] = add_impulse_to_window(
            out[index],
            amplitude=sigma * 6.0,
            rng=rng,
        )
        labels[index] = 1

    return out, labels


def cap_chronological(signals, labels, times, progresses, max_items):
    if len(signals) <= max_items:
        return signals, labels, times, progresses

    indices = np.unique(np.linspace(
        0,
        len(signals) - 1,
        max_items,
        dtype=np.int64,
    ))

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
        t, signal, progress, TRIM_START_SEC, 36.0, TRAIN_STRIDE
    )
    val_normal, _, _ = window_range(
        t, signal, progress, 36.0, 48.0, EVAL_STRIDE
    )
    test, test_times, test_progress = window_range(
        t, signal, progress, 48.0, 60.0, EVAL_STRIDE
    )

    val, val_labels = build_synthetic_val(
        val_normal, rng, anomaly_ratio=0.5, strength=5.0
    )
    test_labels = np.zeros(len(test), np.int64)

    return train, val, val_labels, test, test_labels, test_times, test_progress


def process_mn2(t, signal, progress, rng):
    # Isaac 단계에서는 MN1과 동일 정상.
    train, _, _ = window_range(
        t, signal, progress, TRIM_START_SEC, 36.0, TRAIN_STRIDE
    )
    val_normal, _, _ = window_range(
        t, signal, progress, 36.0, 48.0, EVAL_STRIDE
    )
    test_normal, test_times, test_progress = window_range(
        t, signal, progress, 48.0, 60.0, EVAL_STRIDE
    )

    val, val_labels = build_synthetic_val(
        val_normal, rng, anomaly_ratio=0.5, strength=5.5
    )
    test, test_labels = inject_sparse_test_anomalies(test_normal, rng)

    return train, val, val_labels, test, test_labels, test_times, test_progress


def process_mn3(t, signal, progress, rng):
    hybrid = add_progressive_fault_component(signal, progress, rng)

    # train: 정상 구간만
    train, _, _ = window_range(
        t, hybrid, progress, TRIM_START_SEC, 14.0, TRAIN_STRIDE
    )

    # validation: 정상(14~18s) + 명확한 고장(50~60s)
    val_normal, _, _ = window_range(
        t, hybrid, progress, 14.0, 18.0, EVAL_STRIDE
    )
    val_anomaly, _, _ = window_range(
        t, hybrid, progress, 50.0, 60.0, EVAL_STRIDE
    )

    val = np.concatenate([val_normal, val_anomaly], axis=0)
    val_labels = np.concatenate([
        np.zeros(len(val_normal), np.int64),
        np.ones(len(val_anomaly), np.int64),
    ])

    order = rng.permutation(len(val))
    val, val_labels = val[order], val_labels[order]

    # 실제 시간 순서:
    # 18~20s normal -> 20~40s transition -> 40~50s fault
    test, test_times, test_progress = window_range(
        t, hybrid, progress, 18.0, 50.0, EVAL_STRIDE
    )

    # 조건 변화가 시작되는 시점부터 anomaly onset.
    test_labels = (test_progress > 1e-6).astype(np.int64)

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
    return ((signals - mean) / max(std, 1e-6)).astype(np.float32)


def backup_existing_pkls():
    existing = [
        OUT_DIR / f"{node}.pkl"
        for node in ("mn1", "mn2", "mn3")
        if (OUT_DIR / f"{node}.pkl").exists()
    ]

    if not existing:
        return None

    backup_dir = BACKUP_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
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
    if min(len(train_raw), len(val_raw), len(test_raw)) == 0:
        raise ValueError(f"{node}: train/val/test 중 빈 데이터가 있습니다.")

    norm_mean = float(train_raw.mean())
    norm_std = max(float(train_raw.std()), 1e-6)

    return {
        "train_signals": normalize(train_raw, norm_mean, norm_std),
        "val_signals": normalize(val_raw, norm_mean, norm_std),
        "val_labels": val_labels.astype(np.int64),
        "test_stream_signals": normalize(test_raw, norm_mean, norm_std),
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
    print("\n=== Isaac Sim -> FL PKL ===")
    print(f"RAW_DIR    : {RAW_DIR}")
    print(f"OUT_DIR    : {OUT_DIR}")
    print(f"SIGNAL_COL : {SIGNAL_COL}")
    print(f"TARGET_FS  : {TARGET_FS} Hz")
    print(f"SEQ_LEN    : {SEQ_LEN}")

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Isaac raw 폴더가 없습니다: {RAW_DIR}")

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

    for i, node in enumerate(("mn1", "mn2", "mn3"), start=1):
        raw_path = raw_paths[node]
        df = load_raw(raw_path)
        raw_fs = estimate_raw_fs(df)

        print(f"\n[{node.upper()}] {raw_path}")
        print(f"  measured raw fs : {raw_fs:.2f} Hz")

        t, signal, progress = resample_raw(df)
        rng = np.random.default_rng(SEED + i)

        result = processors[node](t, signal, progress, rng)
        dataset = build_dataset(
            node,
            run_id,
            raw_path,
            raw_fs,
            *result,
        )

        out_path = OUT_DIR / f"{node}.pkl"
        with out_path.open("wb") as fp:
            pickle.dump(dataset, fp, protocol=pickle.HIGHEST_PROTOCOL)

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

    print("\n완료. 다음은 ./clean_fl.sh 실행 후 기존 FL 실행 절차로 진행하세요.\n")


if __name__ == "__main__":
    main()
