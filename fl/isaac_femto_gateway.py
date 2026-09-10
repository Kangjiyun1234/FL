"""
isaac_femto_gateway.py

역할:
1. FEMTO replay npz를 읽는다.
2. Isaac Sim에서 같은 stream을 시간순으로 보여준다. (Isaac 밖에서는 로그만 출력)
3. 기존 FL 코드가 읽을 수 있도록 runtime pkl buffer를 만든다.
4. oneM2M cnt-sensor-data에 data_path metadata를 publish한다.

핵심:
- 전처리 단계에서는 pkl을 만들지 않는다.
- 이 스크립트가 실행 중에 pkl buffer를 만들어 FL에 넘긴다.
- 기존 MN-AE / Dashboard와 최대한 호환되도록 pkl 형식은 기존 FEMTO pkl과 맞춘다.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np


# ============================================================
# 0. Settings
# ============================================================

NODES = ["mn1", "mn2", "mn3"]
NODE_INDEX = {"mn1": 0, "mn2": 1, "mn3": 2}
NODE_AE = {"mn1": "MN-AE-1", "mn2": "MN-AE-2", "mn3": "MN-AE-3"}

GLOBAL_ROUNDS = int(os.getenv("FL_GLOBAL_ROUNDS", "10"))
WINDOWS_PER_ROUND = int(os.getenv("ISAAC_WINDOWS_PER_ROUND", "50"))

# Isaac Sim이 Windows에서 돌 때 기본 경로
WIN_STREAM_DIR = Path(
    os.getenv(
        "ISAAC_FEMTO_STREAM_DIR_WIN",
        "C:/Projects/bearing_testbed/data/femto_replay",
    )
)

WIN_BUFFER_DIR = Path(
    os.getenv(
        "ISAAC_FL_BUFFER_DIR_WIN",
        "C:/Projects/bearing_testbed/data/fl_buffer",
    )
)

# WSL/FL 쪽에서 같은 Windows 폴더를 읽을 때 쓰는 경로
WSL_STREAM_DIR = Path(
    os.getenv(
        "ISAAC_FEMTO_STREAM_DIR_WSL",
        "/mnt/c/Projects/bearing_testbed/data/femto_replay",
    )
)

WSL_BUFFER_DIR = Path(
    os.getenv(
        "ISAAC_FL_BUFFER_DIR_WSL",
        "/mnt/c/Projects/bearing_testbed/data/fl_buffer",
    )
)

# Linux/WSL에서 직접 테스트할 때 fallback
LINUX_STREAM_DIR = Path(os.getenv("FEMTO_REPLAY_OUT_DIR", "/tmp/fl_data/femto_replay"))
LINUX_BUFFER_DIR = Path(os.getenv("FL_PKL_DIR", "/tmp/fl_data/femto"))

TINYIOT_BASE_URL = os.getenv("TINYIOT_BASE_URL", "http://127.0.0.1:3000")
TINYIOT_CSE_NAME = os.getenv("TINYIOT_CSE_NAME", "TinyIoT")
ORIGINATOR = os.getenv("ONEM2M_ORIGINATOR", "CAdmin")

# publish 없이 데이터/시각화만 확인하고 싶으면:
#   python3 isaac_femto_gateway.py --no-publish
NO_PUBLISH = "--no-publish" in sys.argv

# oneM2M round 상태를 기다리지 않고 pkl/publish만 먼저 할 때:
#   python3 isaac_femto_gateway.py --once
ONCE = "--once" in sys.argv

# Isaac 없이 WSL에서 확인할 때 로그 간격
ROUND_SLEEP_SEC = float(os.getenv("ISAAC_REPLAY_ROUND_SLEEP", "1.0"))


def running_on_windows() -> bool:
    return os.name == "nt"


def choose_stream_dir() -> Path:
    if running_on_windows():
        return WIN_STREAM_DIR
    if WSL_STREAM_DIR.exists():
        return WSL_STREAM_DIR
    return LINUX_STREAM_DIR


def choose_buffer_write_dir() -> Path:
    if running_on_windows():
        return WIN_BUFFER_DIR
    if WSL_STREAM_DIR.exists():
        return WSL_BUFFER_DIR
    return LINUX_BUFFER_DIR


def choose_buffer_path_for_fl(node: str) -> str:
    """
    oneM2M metadata에 들어갈 data_path.
    Windows Isaac에서 실행해도 MN-AE는 WSL에서 읽으므로 /mnt/c/... 경로를 보낸다.
    """
    if running_on_windows():
        return str(WSL_BUFFER_DIR / f"{node}.pkl")
    if WSL_STREAM_DIR.exists():
        return str(WSL_BUFFER_DIR / f"{node}.pkl")
    return str(LINUX_BUFFER_DIR / f"{node}.pkl")


# ============================================================
# 1. oneM2M publish
# ============================================================

def create_content_instance(container_path: str, content: Dict[str, Any], labels=None) -> bool:
    if NO_PUBLISH:
        print(f"  [NO_PUBLISH] {container_path} <- {content}")
        return True

    try:
        import requests
    except Exception as e:
        print(f"  ✗ requests import failed: {e}")
        return False

    url = f"{TINYIOT_BASE_URL}/{container_path}"
    headers = {
        "X-M2M-Origin": ORIGINATOR,
        "X-M2M-RVI": "2a",
        "Content-Type": "application/json;ty=4",
        "Accept": "application/json",
    }

    cin_obj = {
        "con": json.dumps(content, ensure_ascii=False, separators=(",", ":")),
    }

    if labels:
        cin_obj["lbl"] = labels

    payload = {"m2m:cin": cin_obj}

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=15)
        if r.status_code == 201:
            return True

        print(f"  ✗ CIN create failed: {r.status_code} - {r.text[:300]}")
        return False

    except Exception as e:
        print(f"  ✗ CIN create error: {type(e).__name__}: {e}")
        return False


def publish_round_metadata(node: str, round_num: int, start_idx: int, end_idx: int, stream_data) -> bool:
    ae_name = NODE_AE[node]
    container_path = f"{TINYIOT_CSE_NAME}/{ae_name}/cnt-sensor-data"

    labels = [
        node,
        f"round_{round_num}",
        "type:isaac-replay-buffer",
    ]

    state_codes = stream_data["test_stream_state_codes"][start_idx:end_idx]
    labels_arr = stream_data["test_stream_labels"][start_idx:end_idx]
    rms_arr = stream_data["test_stream_rms"][start_idx:end_idx]
    peak_arr = stream_data["test_stream_peak"][start_idx:end_idx]

    payload = {
        "type": "isaac-replay-buffer",
        "node": node,
        "round": int(round_num),
        "data_path": choose_buffer_path_for_fl(node),
        "stream_path": str(WSL_STREAM_DIR / f"{node}_stream.npz"),
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "window_count": int(end_idx - start_idx),
        "rms_mean": float(np.mean(rms_arr)) if len(rms_arr) else 0.0,
        "peak_max": float(np.max(peak_arr)) if len(peak_arr) else 0.0,
        "state_codes": sorted([int(x) for x in set(state_codes.tolist())]),
        "fault_count": int(np.sum(labels_arr == 1)),
        "timestamp": time.time(),
    }

    ok = create_content_instance(container_path, payload, labels=labels)
    mark = "✓" if ok else "✗"
    print(f"  {mark} publish {node} R{round_num} -> {container_path}")
    return ok


# ============================================================
# 2. stream npz -> FL pkl buffer
# ============================================================

def load_stream_npz(stream_dir: Path) -> Dict[str, Any]:
    streams = {}

    for node in NODES:
        path = stream_dir / f"{node}_stream.npz"
        if not path.exists():
            raise FileNotFoundError(f"stream npz 없음: {path}")

        streams[node] = np.load(path, allow_pickle=True)
        print(f"  ✓ load {node}: {path}")

    return streams


def norm_array(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    std = max(float(std), 1e-6)
    return ((arr.astype(np.float32) - float(mean)) / std).astype(np.float32)


def build_fl_dataset(node: str, stream_data) -> Dict[str, Any]:
    """
    기존 MN-AE / Dashboard가 읽는 pkl 구조로 맞춘다.
    """
    mean = float(np.asarray(stream_data["norm_mean"]).item())
    std = float(np.asarray(stream_data["norm_std"]).item())

    train_signals = norm_array(stream_data["train_windows"], mean, std)
    val_signals = norm_array(stream_data["val_windows"], mean, std)
    test_stream_signals = norm_array(stream_data["test_stream_windows"], mean, std)

    dataset = {
        "node": node,
        "source": "isaac_femto_replay_buffer",
        "motors": [str(np.asarray(stream_data["node_info"]).item())],
        "train_signals": train_signals,
        "val_signals": val_signals,
        "val_labels": stream_data["val_labels"].astype(np.int64),
        "test_stream_signals": test_stream_signals,
        "test_stream_labels": stream_data["test_stream_labels"].astype(np.int64),
        "test_stream_times": stream_data["test_stream_times"].astype(np.float32),
        "test_stream_progress": stream_data["test_stream_progress"].astype(np.float32),
        "test_stream_state_codes": stream_data["test_stream_state_codes"].astype(np.int64),
        "test_stream_rms": stream_data["test_stream_rms"].astype(np.float32),
        "test_stream_peak": stream_data["test_stream_peak"].astype(np.float32),
        "test_stream_visual_amp": stream_data["test_stream_visual_amp"].astype(np.float32),
        "norm_mean": np.float32(mean),
        "norm_std": np.float32(std),
        "seq_len": int(np.asarray(stream_data["seq_len"]).item()),
        "n_channels": 1,
        "sample_rate": int(np.asarray(stream_data["sample_rate"]).item()),
        "window_sec": float(np.asarray(stream_data["window_sec"]).item()),
        "fault_onset_idx": int(np.asarray(stream_data["fault_onset_idx"]).item()),
        "created_by": "isaac_femto_gateway.py",
    }

    return dataset


def write_fl_buffers(streams: Dict[str, Any], buffer_write_dir: Path) -> None:
    buffer_write_dir.mkdir(parents=True, exist_ok=True)

    for node in NODES:
        dataset = build_fl_dataset(node, streams[node])
        out_path = buffer_write_dir / f"{node}.pkl"

        with out_path.open("wb") as f:
            pickle.dump(dataset, f)

        print(
            f"  ✓ buffer {node}: {out_path} "
            f"train={dataset['train_signals'].shape} "
            f"test={dataset['test_stream_signals'].shape}"
        )


# ============================================================
# 3. Isaac visual hook
# ============================================================

def try_update_isaac_visual(streams: Dict[str, Any], round_num: int, start_idx: int, end_idx: int) -> None:
    """
    Isaac Sim 안에서 실행하면 간단한 prim 3개를 만들고 진동 크기만 반영한다.
    Isaac 밖에서 실행하면 그냥 return.
    """
    try:
        import omni.usd
        from pxr import Gf, UsdGeom
    except Exception:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    positions = {
        "mn1": (-1.5, 0.0, 0.5),
        "mn2": (0.0, 0.0, 0.5),
        "mn3": (1.5, 0.0, 0.5),
    }

    for node in NODES:
        prim_path = f"/World/{node.upper()}_Bearing"
        prim = stage.GetPrimAtPath(prim_path)

        if not prim.IsValid():
            cube = UsdGeom.Cube.Define(stage, prim_path)
            cube.CreateSizeAttr(0.45)
            prim = stage.GetPrimAtPath(prim_path)

        amp = float(np.mean(streams[node]["test_stream_visual_amp"][start_idx:end_idx]))
        state_max = int(np.max(streams[node]["test_stream_state_codes"][start_idx:end_idx]))

        x, y, z = positions[node]
        dz = math.sin(round_num * 1.7 + NODE_INDEX[node]) * amp

        xform = UsdGeom.Xformable(prim)
        try:
            xform.ClearXformOpOrder()
        except Exception:
            pass

        xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z + dz))

        # displayColor: normal=blue-ish, degradation=yellow-ish, fault=red-ish
        try:
            gprim = UsdGeom.Gprim(prim)
            if state_max == 2:
                color = [(1.0, 0.15, 0.10)]
            elif state_max == 1:
                color = [(1.0, 0.75, 0.15)]
            else:
                color = [(0.15, 0.45, 1.0)]
            gprim.CreateDisplayColorAttr(color)
        except Exception:
            pass


def print_round_preview(streams: Dict[str, Any], round_num: int, start_idx: int, end_idx: int) -> None:
    print(f"\n=== Isaac Replay Round {round_num} [{start_idx}:{end_idx}] ===")

    for node in NODES:
        labels = streams[node]["test_stream_labels"][start_idx:end_idx]
        states = streams[node]["test_stream_state_codes"][start_idx:end_idx]
        rms = streams[node]["test_stream_rms"][start_idx:end_idx]
        peak = streams[node]["test_stream_peak"][start_idx:end_idx]

        print(
            f"  {node}: "
            f"state={sorted([int(x) for x in set(states.tolist())])} "
            f"fault_count={int(np.sum(labels == 1)):3d} "
            f"rms_mean={float(np.mean(rms)):.4f} "
            f"peak_max={float(np.max(peak)):.4f}"
        )


# ============================================================
# 4. Main
# ============================================================

def main() -> None:
    stream_dir = choose_stream_dir()
    buffer_write_dir = choose_buffer_write_dir()

    print("\n=== Isaac FEMTO Gateway ===")
    print(f"stream_dir      : {stream_dir}")
    print(f"buffer_write_dir: {buffer_write_dir}")
    print(f"buffer path for FL:")
    for node in NODES:
        print(f"  {node}: {choose_buffer_path_for_fl(node)}")
    print(f"oneM2M          : {TINYIOT_BASE_URL}/{TINYIOT_CSE_NAME}")
    print(f"NO_PUBLISH      : {NO_PUBLISH}")
    print()

    streams = load_stream_npz(stream_dir)

    # 기존 MN-AE / Dashboard 호환용 runtime buffer pkl 생성
    write_fl_buffers(streams, buffer_write_dir)

    if ONCE:
        print("\n--once 모드: buffer 생성만 하고 종료합니다.")
        return

    # 10 round 기준으로 stream을 50개씩 보여주고 metadata publish
    for round_num in range(1, GLOBAL_ROUNDS + 1):
        start_idx = (round_num - 1) * WINDOWS_PER_ROUND
        end_idx = min(round_num * WINDOWS_PER_ROUND, int(streams["mn1"]["test_stream_windows"].shape[0]))

        print_round_preview(streams, round_num, start_idx, end_idx)
        try_update_isaac_visual(streams, round_num, start_idx, end_idx)

        for node in NODES:
            publish_round_metadata(node, round_num, start_idx, end_idx, streams[node])

        time.sleep(ROUND_SLEEP_SEC)

    print("\n완료: Isaac replay/gateway 종료")


if __name__ == "__main__":
    main()