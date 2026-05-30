"""
fl/data_generator.py — FL 라운드별 센서 데이터 메타 업로드 (AE 버전)

AE 방식에서는 슬라이딩 윈도우가 필요 없음.
모든 라운드에서 동일한 base pkl을 사용하며,
라운드별로 cnt-sensor-data에 data_path만 publish.

Usage:
  python3 data_generator.py          # Round 1~20 전부 업로드
  python3 data_generator.py --reset  # 이전 캐시 삭제 후 업로드
"""
from __future__ import annotations

import os
import sys
import json
import time
import shutil

sys.path.append('/home/eunjin/federated-learning/fl')

import config
import onem2m_utils as om2m

BASE_DIR    = "/tmp/fl_data/femto"
TOTAL_ROUNDS = int(getattr(config, "GLOBAL_ROUNDS", 20))

# 노드 인덱스 → base pkl 경로
BASE_PKL_PATHS = {
    0: os.path.join(BASE_DIR, "mn1.pkl"),
    1: os.path.join(BASE_DIR, "mn2.pkl"),
    2: os.path.join(BASE_DIR, "mn3.pkl"),
}

NODE_NAMES = {0: "mn1", 1: "mn2", 2: "mn3"}


def get_sensor_data_container_path(node_idx: int) -> str:
    ae_name = f"MN-AE-{node_idx + 1}"
    return f"{config.CSE_NAME}/{ae_name}/cnt-sensor-data"


def upload_round_metadata(node_idx: int, round_num: int) -> bool:
    """
    해당 노드의 cnt-sensor-data 에 round_N 라벨로 CIN 업로드.
    data_path = base pkl 경로 (AE는 매 라운드 동일 데이터 사용).
    """
    pkl_path = BASE_PKL_PATHS[node_idx]
    if not os.path.exists(pkl_path):
        print(f"  ✗ pkl 없음: {pkl_path}  (먼저 prepare_data_femto.py 실행)")
        return False

    container_path = get_sensor_data_container_path(node_idx)
    node_name = NODE_NAMES[node_idx]

    payload = {
        "node":       node_name,
        "data_path":  pkl_path,
        "round":      round_num,
        "timestamp":  time.time(),
    }
    labels = [node_name, f"round_{round_num}", "type:ae-sensor-data"]

    result = om2m.create_content_instance(container_path, payload, labels=labels)
    ok = bool(result)
    print(
        f"  [{node_name}] round={round_num}  CIN -> {container_path}"
        f"  {'✓' if ok else '✗'}"
    )
    return ok


def reset_local_artifacts() -> None:
    print("\n[0] 이전 모델 파일 삭제...")
    for path in [
        "/tmp/fl_models/local",
        "/tmp/fl_models/global",
        "/tmp/fl_models/cache",
    ]:
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
    print("  ✓ 완료")


def main():
    reset = "--reset" in sys.argv

    if reset:
        reset_local_artifacts()

    print(f"\n=== FL Data Generator (FEMTO AE 버전) ===")
    print(f"  전체 라운드: {TOTAL_ROUNDS}")
    print(f"  base pkl 경로:")
    for idx, path in BASE_PKL_PATHS.items():
        exists = "✓" if os.path.exists(path) else "✗ 없음"
        print(f"    {NODE_NAMES[idx]}: {path}  {exists}")

    missing = [p for p in BASE_PKL_PATHS.values() if not os.path.exists(p)]
    if missing:
        print("\n  ✗ pkl 파일이 없습니다. 먼저 실행하세요:")
        print("      python3 fl/prepare_data_femto.py")
        sys.exit(1)

    print(f"\n[1] Round 1~{TOTAL_ROUNDS} 메타데이터 업로드...")
    for round_num in range(1, TOTAL_ROUNDS + 1):
        print(f"\n  --- Round {round_num} ---")
        for node_idx in range(getattr(config, "NUM_CLIENTS", 3)):
            upload_round_metadata(node_idx, round_num)

    print("\n✓ 전체 완료!")


if __name__ == "__main__":
    main()
