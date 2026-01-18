# fl/data.py (완전 수정)
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List

from fl.config import DataConfig


# fl/data.py

def load_pump_sensor_data(data_dir: str, sample_size: int = 10000):
    csv_path = Path(data_dir) / "sensor.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 1) label 먼저 확보 (label 없는 행만 제거)
    if 'machine_status' not in df.columns:
        raise ValueError("Column 'machine_status' not found in CSV")
    df['machine_status'] = df['machine_status'].fillna('NORMAL')
    df = df[df['machine_status'].notna()].reset_index(drop=True)

    # 2) timestamp 정렬
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.sort_values('timestamp').reset_index(drop=True)

    # 3) 센서 컬럼 NaN 처리 (핵심)
    sensor_cols = [c for c in df.columns if c.startswith('sensor_')]
    # forward-fill -> back-fill -> 0
    df[sensor_cols] = df[sensor_cols].ffill().bfill().fillna(0.0)

    print("Class distribution before encoding:")
    print(df['machine_status'].value_counts())

    le = LabelEncoder()
    df['status_encoded'] = le.fit_transform(df['machine_status'])

    # 4) (옵션) 샘플링은 그 다음에
    if len(df) > sample_size:
        n_classes = df['machine_status'].nunique()
        per_class = max(1, sample_size // n_classes)
        df = df.groupby('machine_status', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), per_class), random_state=42)
        ).reset_index(drop=True)
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    print(f"Loaded {len(df)} samples")
    print(f"Classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    return df, le


def create_sequences(
    data: pd.DataFrame, 
    sensor_cols: List[str], 
    seq_len: int,
    max_sequences: int = 5000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    시계열 시퀀스 생성 (메모리 효율적)
    """
    # 노드가 가진 센서만 필터링
    available_sensors = [col for col in sensor_cols if col in data.columns]
    
    if not available_sensors:
        raise ValueError("No matching sensor columns found in data")
    
    temps = data[available_sensors].values.astype(np.float32)
    labels = data['status_encoded'].values.astype(np.int64)
    
    # 🔴 NaN 체크
    if np.isnan(temps).any():
        print("Warning: NaN found in sensor data, filling with 0")
        temps = np.nan_to_num(temps, nan=0.0)
    
    total_possible = len(temps) - seq_len
    
    if total_possible <= 0:
        raise ValueError(
            f"Not enough data: len={len(temps)}, seq_len={seq_len}"
        )
    
    # 메모리 절약: 샘플링
    if total_possible > max_sequences:
        indices = np.linspace(0, total_possible - 1, max_sequences, dtype=int)
        print(f"  Creating {max_sequences} sequences (sampled from {total_possible})")
    else:
        indices = np.arange(total_possible)
        print(f"  Creating {total_possible} sequences")
    
    X = np.zeros((len(indices), seq_len, len(available_sensors)), dtype=np.float32)
    y = np.zeros(len(indices), dtype=np.int64)
    
    for idx, i in enumerate(indices):
        X[idx] = temps[i:i + seq_len]
        y[idx] = labels[i + seq_len]
    
    # 🔴 정규화 (gradient explosion 방지)
    X_mean = np.mean(X, axis=(0, 1), keepdims=True)
    X_std = np.std(X, axis=(0, 1), keepdims=True) + 1e-8
    X = (X - X_mean) / X_std
    
    return X, y


def split_sensors_by_nodes(df, num_nodes: int = 3):
    sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
    print(f"Total sensors: {len(sensor_cols)}")

    sensors_per_node = len(sensor_cols) // num_nodes

    node_datasets = []
    node_sensor_lists = []

    for i in range(num_nodes):
        start_idx = i * sensors_per_node
        end_idx = len(sensor_cols) if i == num_nodes - 1 else (i + 1) * sensors_per_node

        node_sensors = sensor_cols[start_idx:end_idx]
        node_sensor_lists.append(node_sensors)

        # ✅ 노드 DF에는 전체 센서 컬럼을 유지
        cols_to_keep = ['status_encoded'] + sensor_cols
        if 'timestamp' in df.columns:
            cols_to_keep = ['timestamp'] + cols_to_keep

        node_data = df[cols_to_keep].copy()

        # ✅ 노드가 "가지지 않은" 센서는 0으로 마스킹
        other_sensors = [c for c in sensor_cols if c not in node_sensors]
        node_data[other_sensors] = 0.0

        print(f"Node {i}: owns {len(node_sensors)} sensors, samples={len(node_data)}")
        node_datasets.append(node_data)

    return node_datasets, node_sensor_lists, sensor_cols



def make_node_dataset(cfg: DataConfig, num_nodes: int):
    """
    노드별 데이터셋 생성
    
    Returns:
        (node_datasets, sensor_cols, label_encoder)
    """
    if cfg.dataset == "pump_sensor":
        df, le = load_pump_sensor_data(
            cfg.data_dir, 
            sample_size=10000  # 🔴 메모리 절약
        )
        node_dfs, node_sensor_lists, all_sensors = split_sensors_by_nodes(df, num_nodes)
        return node_dfs, node_sensor_lists, all_sensors, le
    
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")