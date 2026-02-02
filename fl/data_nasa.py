# fl/data_nasa.py
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler

# NASA 데이터 컬럼명
INDEX_COLS = ['unit_id', 'time_cycles']
SETTING_COLS = ['setting_1', 'setting_2', 'setting_3']
SENSOR_COLS = [f'sensor_{i}' for i in range(1, 22)]  # sensor_1 ~ sensor_21
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS


def load_nasa_turbofan(data_path: str, dataset: str = 'FD001', mode: str = 'train') -> pd.DataFrame:
    """
    NASA Turbofan 데이터 로드
    
    Args:
        data_path: 데이터 디렉토리 경로
        dataset: 'FD001', 'FD002', 'FD003', 'FD004' 중 선택
        mode: 'train' 또는 'test'
    
    Returns:
        전처리된 DataFrame
    """
    if mode == 'train':
        file_path = f"{data_path}/train_{dataset}.txt"
    else:
        file_path = f"{data_path}/test_{dataset}.txt"
    
    # 공백으로 구분된 텍스트 파일 로드
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=ALL_COLS)
    
    print(f"[NASA Data] Loaded {mode.upper()} {dataset}: {len(df)} rows, {df['unit_id'].nunique()} engines")
    
    return df


def load_rul_labels(data_path: str, dataset: str = 'FD001') -> np.ndarray:
    """
    RUL 정답 레이블 로드
    
    Returns:
        각 테스트 엔진의 실제 RUL 값 (1D array)
    """
    rul_file = f"{data_path}/RUL_{dataset}.txt"
    rul_array = np.loadtxt(rul_file)
    print(f"[RUL Labels] Loaded {len(rul_array)} test engine RULs")
    return rul_array


def add_rul_labels(df: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
    """
    RUL (Remaining Useful Life) 레이블 추가
    
    - RUL: 고장까지 남은 사이클 수
    - Binary label: RUL < threshold면 1 (곧 고장), 아니면 0
    """
    # 각 엔진별로 최대 사이클 계산
    max_cycles = df.groupby('unit_id')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']
    
    df = df.merge(max_cycles, on='unit_id', how='left')
    
    # RUL 계산
    df['RUL'] = df['max_cycle'] - df['time_cycles']
    df['RUL'] = df['RUL'].clip(upper=max_rul)  # cap at max_rul
    
    # Binary classification: RUL < 30이면 곧 고장 (1), 아니면 정상 (0)
    df['Maintenance_Flag'] = (df['RUL'] < 30).astype(int)
    
    print(f"[RUL Labels] Positive ratio: {df['Maintenance_Flag'].mean():.2%}")
    
    return df


def add_rul_labels_for_test(df: pd.DataFrame, true_rul: np.ndarray, max_rul: int = 125) -> pd.DataFrame:
    max_cycles = df.groupby('unit_id')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']
    df = df.merge(max_cycles, on='unit_id', how='left')

    engine_ids = sorted(df['unit_id'].unique())
    if len(true_rul) < len(engine_ids):
        raise ValueError(f"true_rul length {len(true_rul)} < number of engines {len(engine_ids)}")

    rul_map = {engine_ids[i]: float(true_rul[i]) for i in range(len(engine_ids))}
    df['true_max_rul'] = df['unit_id'].map(rul_map)

    # 핵심 수정: row별 RUL = (마지막관측까지 남은 사이클) + (마지막관측 시점의 true RUL)
    df['RUL'] = (df['max_cycle'] - df['time_cycles']) + df['true_max_rul']

    df['RUL'] = df['RUL'].clip(upper=max_rul, lower=0)
    df['Maintenance_Flag'] = (df['RUL'] < 30).astype(int)

    print(f"[Test RUL Labels] Positive ratio: {df['Maintenance_Flag'].mean():.2%}")
    return df


def prepare_nasa_for_fl(
    data_path: str,
    dataset: str = 'FD001',
    num_nodes: int = 5,
    use_test: bool = False
) -> Tuple[List[pd.DataFrame], List[str], dict]:
    """
    NASA 데이터를 FL용으로 준비
    
    Args:
        data_path: 데이터 경로
        dataset: FD001, FD002, FD003, FD004
        num_nodes: 노드 개수
        use_test: True면 테스트 데이터 반환, False면 훈련 데이터 반환
    
    Returns:
        node_dfs: 노드별 DataFrame 리스트
        sensor_cols: 사용할 센서 컬럼 리스트
        meta: 메타데이터 (scaler, engine_ids, test_rul 등)
    """
    # Train 데이터로 scaler 학습
    train_df = load_nasa_turbofan(data_path, dataset, mode='train')
    train_df = add_rul_labels(train_df)
    
    # 상수값 센서 제거 (분산이 거의 0인 센서)
    sensor_stds = train_df[SENSOR_COLS].std()
    useful_sensors = sensor_stds[sensor_stds > 0.01].index.tolist()
    
    print(f"[Feature Selection] Using {len(useful_sensors)}/{len(SENSOR_COLS)} sensors")
    print(f"  Removed: {set(SENSOR_COLS) - set(useful_sensors)}")
    
    # Scaler 학습 (Train 데이터로)
    scaler = StandardScaler()
    scaler.fit(train_df[useful_sensors])
    print(f"[Normalization] Scaler fitted on train data")
    
    # 사용할 데이터 선택
    if use_test:
        # 테스트 모드: Test 데이터 + RUL 정답 사용
        df = load_nasa_turbofan(data_path, dataset, mode='test')
        true_rul = load_rul_labels(data_path, dataset)
        df = add_rul_labels_for_test(df, true_rul)
        print("[Mode] Using TEST data for evaluation")
    else:
        # 훈련 모드: Train 데이터 사용
        df = train_df
        print("[Mode] Using TRAIN data for federated learning")
    
    # 정규화 적용
    df[useful_sensors] = scaler.transform(df[useful_sensors])
    
    # 엔진 ID로 노드 분할
    engine_ids = sorted(df['unit_id'].unique())
    
    if len(engine_ids) < num_nodes:
        raise ValueError(f"Not enough engines ({len(engine_ids)}) for {num_nodes} nodes")
    
    # 엔진을 균등하게 노드에 배분
    engines_per_node = np.array_split(engine_ids, num_nodes)
    
    node_dfs = []
    for i, engine_group in enumerate(engines_per_node):
        node_df = df[df['unit_id'].isin(engine_group)].copy()
        
        pos = (node_df['Maintenance_Flag'] == 1).sum()
        neg = (node_df['Maintenance_Flag'] == 0).sum()
        total = len(node_df)
        
        print(f"[Node {i+1}] Engines={len(engine_group)}, Samples={total}, "
              f"Positive={pos}({pos/total:.1%}), Negative={neg}({neg/total:.1%})")
        
        node_dfs.append(node_df)
    
    meta = {
        'scaler': scaler,
        'engine_ids': engines_per_node,
        'mode': 'test' if use_test else 'train',
        'dataset': dataset,
    }
    
    # 테스트 모드면 RUL 정답도 포함
    if use_test:
        meta['test_rul'] = true_rul
    
    return node_dfs, useful_sensors, meta


def create_sequences_nasa(
    df: pd.DataFrame,
    sensor_cols: List[str],
    seq_len: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    엔진별 시계열 시퀀스 생성
    """
    X_list, y_list = [], []
    
    for engine_id in df['unit_id'].unique():
        engine_df = df[df['unit_id'] == engine_id].sort_values('time_cycles')
        
        X_values = engine_df[sensor_cols].values
        y_values = engine_df['Maintenance_Flag'].values
        
        # 슬라이딩 윈도우
        for i in range(len(engine_df) - seq_len + 1):
            X_list.append(X_values[i:i + seq_len])
            y_list.append(y_values[i + seq_len - 1])  # 윈도우 끝의 레이블
    
    if len(X_list) == 0:
        return np.empty((0, seq_len, len(sensor_cols)), dtype=np.float32), np.empty((0,), dtype=np.int64)
    
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    
    print(f"[Sequences] Created {len(X)} sequences (seq_len={seq_len})")
    
    return X, y