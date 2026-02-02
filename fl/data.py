# fl/data.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from fl.data_nasa import prepare_nasa_for_fl, create_sequences_nasa

# 이번 CSV에 맞춘 고정 컬럼
ID_COL = "Pump_ID"
TIME_COL = "Operational_Hours"
LABEL_COL = "Maintenance_Flag"
FEATURE_COLS = ["Temperature", "Vibration", "Pressure", "Flow_Rate", "RPM", "Operational_Hours"]


def make_node_dataset(data_cfg, num_nodes: int):
    """데이터셋 타입에 따라 분기"""
    dataset_type = getattr(data_cfg, "dataset", "pump_sensor")
    
    if dataset_type == "nasa_turbofan":
        # NASA 데이터
        data_path = f"{data_cfg.data_dir}/nasa_turbofan"
        
        # use_test는 기본 False (훈련용)
        node_dfs, sensor_cols, meta = prepare_nasa_for_fl(
            data_path=data_path,
            dataset=getattr(data_cfg, "nasa_dataset", "FD001"),
            num_nodes=num_nodes,
            use_test=False  # 훈련 단계
        )
        
        all_sensors = sensor_cols
        node_sensor_lists = [sensor_cols for _ in node_dfs]
        
        # create_sequences도 NASA 버전 사용
        global create_sequences
        create_sequences = create_sequences_nasa
        
        return node_dfs, node_sensor_lists, all_sensors, meta
        
    # ===============================
    # 2) Pump Sensor CSV 데이터
    # ===============================
    csv_path = getattr(data_cfg, "csv_path", None)
    if not csv_path:
        raise KeyError("data.csv_path is required in config (DataConfig).")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # 필수 컬럼 검증
    required = [ID_COL, TIME_COL, LABEL_COL] + FEATURE_COLS
    missing = [c for c in set(required) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}\nFound: {df.columns.tolist()}")

    # ===== 정규화 =====
    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
    print(f"[Data] Features normalized: mean≈0, std≈1")

    # Pump_ID 목록
    pump_ids = sorted(df[ID_COL].unique().tolist())

    if num_nodes != len(pump_ids):
        raise ValueError(
            f"num_nodes={num_nodes} but Pump_ID unique count={len(pump_ids)} ({pump_ids}).\n"
            f"Set num_nodes to {len(pump_ids)} to use '1 pump = 1 edge' split."
        )

    node_dfs: List[pd.DataFrame] = []
    for pid in pump_ids:
        sub = df[df[ID_COL] == pid].copy()
        sub = sub.sort_values(TIME_COL)

        pos = (sub[LABEL_COL] == 1).sum()
        neg = (sub[LABEL_COL] == 0).sum()
        total = len(sub)
        print(
            f"[Node {pid}] Samples={total}, "
            f"Positive={pos}({pos/total:.1%}), "
            f"Negative={neg}({neg/total:.1%})"
        )

        node_dfs.append(sub)

    all_sensors = FEATURE_COLS[:]
    node_sensor_lists = [FEATURE_COLS[:] for _ in node_dfs]

    meta = {"pump_ids": pump_ids, "scaler": scaler}
    return node_dfs, node_sensor_lists, all_sensors, meta


def create_sequences(df: pd.DataFrame, sensor_cols: List[str], seq_len: int):
    """
    시계열이 아니므로 그냥 각 row를 개별 샘플로 사용
    seq_len=1로 고정되어야 함
    """
    if seq_len != 1:
        print(f"[Warning] Pump data is not time-series. Forcing seq_len=1 (was {seq_len})")
        seq_len = 1
    
    df = df.sort_values(TIME_COL)  # 정렬은 유지 (일관성)

    X_values = df[sensor_cols].to_numpy(dtype=np.float32)
    y_values = df[LABEL_COL].to_numpy(dtype=np.int64)

    # 🔥 각 row가 독립적인 샘플
    # (N, 6) → (N, 1, 6) 형태로 reshape (기존 코드 호환)
    X = X_values[:, np.newaxis, :]  # (N, 1, 6)
    y = y_values
    
    print(f"[Data] Created {len(X)} samples (not sequences) with {len(sensor_cols)} features")
    return X, y