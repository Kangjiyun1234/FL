# isaac_femto_demo_gateway_v7.py
# Isaac Sim 6.0.1 Script Editor에서 실행하는 최종 데모 연결 코드
#
# 실행:
# exec(open(r"C:\Projects\bearing_testbed\scripts\isaac_femto_demo_gateway_v7.py", encoding="utf-8").read())
#
# 중지:
# stop_femto_demo()
#
# 역할:
# 1. C:\Projects\bearing_testbed\data\femto_replay\mn*_stream.npz 읽기
# 2. Isaac Sim에서 MN1/MN2/MN3 시각화
# 3. C:\Projects\bearing_testbed\data\fl_buffer\mn*.pkl runtime buffer 생성
# 4. IN-AE의 cnt-fl-control/la를 감시
# 5. FL_TRAINING Round N이 감지되면 MN1/MN2/MN3 sensor-data metadata publish
# 6. 해당 Round 구간을 시각화하고, 다음 Round 전까지 최근 구간을 반복 재생
# 7. Dashboard에서 실제 fault alert가 감지되면 해당 node pin을 빨강으로 전환
#
# v7 최종 수정:
# - idle 대기 중 마지막 구간을 반복 재생해서 정상처럼 멈춰 보이지 않게 함
# - Round 10 시각화가 끝나면 Isaac Sim demo controller 자동 정지
# - pin 색은 기본 초록 유지
# - Dashboard /stream score 이벤트에서 score > threshold 조건을 직접 판정
# - 3회 연속 threshold 초과 시 해당 node pin을 빨강으로 전환
# - 이전 callback/invalid prim 오류 방어 유지

from __future__ import annotations

import os
import json
import math
import time
import pickle
import threading
import urllib.request
import urllib.error

import numpy as np

import omni.kit.app
import omni.usd
from pxr import UsdGeom, Gf


# =========================================================
# 경로 / oneM2M 설정
# =========================================================

STREAM_DIR = r"C:\Projects\bearing_testbed\data\femto_replay"
BUFFER_WRITE_DIR_WIN = r"C:\Projects\bearing_testbed\data\fl_buffer"
BUFFER_DIR_FOR_WSL = "/mnt/c/Projects/bearing_testbed/data/fl_buffer"

TINYIOT_BASE_URL = "http://127.0.0.1:3000"
TINYIOT_CSE_NAME = "TinyIoT"
ORIGINATOR = "CAdmin"

NODE_AE = {
    "mn1": "MN-AE-1",
    "mn2": "MN-AE-2",
    "mn3": "MN-AE-3",
}

ROOT_PATH = "/World/FEMTO_Replay_Test"

# 80 window = FL 1 round
WINDOWS_PER_ROUND = 80
TOTAL_ROUNDS = 10

# Round 내부 시각화 속도.
# 10.0이면 한 round 80개 window를 약 8초에 재생함.
VISUAL_SPEED_WPS = 10.0

# round 사이 대기 중 반복 재생할 최근 window 개수/속도.
# 예: 마지막 24개 window를 6 windows/sec로 반복 -> 약 4초 루프.
IDLE_LOOP_WINDOWS = 24
IDLE_LOOP_SPEED_WPS = 6.0

# fl-control polling 주기. 너무 낮추면 TinyIoT 요청이 많아짐.
FL_CONTROL_POLL_SEC = 0.5

# Dashboard fault alert 감지 설정.
# /stream SSE를 먼저 듣고, 실패하면 /health 계열 JSON도 보조로 확인함.
DASHBOARD_BASE_URL = "http://127.0.0.1:7000"
DASHBOARD_STREAM_PATH = "/stream"
DASHBOARD_POLL_SEC = 1.0

# Dashboard frontend의 fault toast는 dashboard_server.py가 직접 "fault detected" 문자열을
# 보내는 것이 아니라, /stream score 이벤트를 보고 브라우저에서 판단한다.
# 그래서 Isaac Sim도 score > threshold를 직접 계산한다.
DASHBOARD_ALERT_K_CONSECUTIVE = 3
DASHBOARD_ALERT_REQUIRE_LABEL = True

# False로 바꾸면 Isaac 시각화만 하고 TinyIoT publish는 안 함
ENABLE_PUBLISH = True


# =========================================================
# 상태 / 색상 / 배치
# =========================================================

STATE_NORMAL = 0
STATE_DEGRADATION = 1
STATE_FAULT = 2

STATE_NAME = {
    STATE_NORMAL: "NORMAL",
    STATE_DEGRADATION: "DEGRADATION",
    STATE_FAULT: "FAULT",
}

COLOR_COMMON_BASE = (0.88, 0.88, 0.88)
COLOR_NODE_BASE = (0.92, 0.92, 0.92)
COLOR_RAIL = (0.66, 0.66, 0.66)
COLOR_SUPPORT = (0.74, 0.74, 0.74)
COLOR_HOUSING = (0.82, 0.84, 0.86)
COLOR_SHAFT = (0.96, 0.96, 0.96)
COLOR_DARK_GAP = (0.10, 0.10, 0.10)

# pin 색상:
# 기본은 항상 초록.
# Isaac 내부 상태가 DEGRADATION/FAULT여도 노랑/빨강으로 바꾸지 않음.
# FL Dashboard에서 실제 fault alert가 감지된 node만 빨강으로 바꿈.
COLOR_PIN_NORMAL = (0.00, 0.80, 0.00)  # 초록
COLOR_PIN_FAULT = (1.00, 0.00, 0.00)   # 빨강

# 월드 좌표 기준: 왼쪽 -> 오른쪽 = MN1, MN2, MN3
NODE_LAYOUT = {
    "mn1": (-4.2, 0.0, 0.0),
    "mn2": (0.0, 0.0, 0.0),
    "mn3": (4.2, 0.0, 0.0),
}

NODE_ROTATION_SPEED = {
    "mn1": 680.0,
    "mn2": 720.0,
    "mn3": 780.0,
}


# =========================================================
# USD 생성 유틸
# =========================================================

def get_stage():
    return omni.usd.get_context().get_stage()


def remove_prim_if_exists(path: str):
    stage = get_stage()
    prim = stage.GetPrimAtPath(path)
    if prim and prim.IsValid():
        stage.RemovePrim(path)


def set_display_color(prim, rgb):
    """Prim이 stage remove 등으로 invalid가 되어도 데모가 죽지 않게 방어."""
    try:
        if prim is None:
            return
        try:
            if hasattr(prim, "IsValid") and not prim.IsValid():
                return
        except Exception:
            return

        gprim = UsdGeom.Gprim(prim)
        attr = gprim.GetDisplayColorAttr()
        if not attr:
            attr = gprim.CreateDisplayColorAttr()
        attr.Set([Gf.Vec3f(*rgb)])
    except Exception:
        # 재실행/stop 과정에서 old callback이 invalid prim을 만져도 무시
        pass

def clear_xform_ops(prim):
    xform = UsdGeom.Xformable(prim)
    try:
        xform.ClearXformOpOrder()
    except Exception:
        pass
    return xform


def make_xform(path: str, translate=(0, 0, 0), rotate=(0, 0, 0)):
    stage = get_stage()
    xf_prim = UsdGeom.Xform.Define(stage, path)
    prim = xf_prim.GetPrim()
    xform = clear_xform_ops(prim)

    t_op = xform.AddTranslateOp()
    r_op = xform.AddRotateXYZOp()

    t_op.Set(Gf.Vec3d(*translate))
    r_op.Set(Gf.Vec3f(*rotate))

    return xf_prim, t_op, r_op


def make_cube(
    path: str,
    size=(1, 1, 1),
    translate=(0, 0, 0),
    rotate=(0, 0, 0),
    color=(1, 1, 1),
):
    stage = get_stage()
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)

    prim = cube.GetPrim()
    xform = clear_xform_ops(prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotate))
    xform.AddScaleOp().Set(Gf.Vec3f(*size))

    set_display_color(prim, color)
    return cube


def make_cylinder(
    path: str,
    radius=0.1,
    height=1.0,
    translate=(0, 0, 0),
    rotate=(0, 0, 0),
    color=(1, 1, 1),
):
    stage = get_stage()
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)

    prim = cyl.GetPrim()
    xform = clear_xform_ops(prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotate))

    set_display_color(prim, color)
    return cyl



# =========================================================
# fl-control polling용 oneM2M GET 유틸
# =========================================================

def _http_json_get(resource_path: str, timeout: float = 2.0):
    url = f"{TINYIOT_BASE_URL}/{resource_path}"
    req = urllib.request.Request(
        url=url,
        method="GET",
        headers={
            "X-M2M-Origin": ORIGINATOR,
            "X-M2M-RVI": "2a",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
        return json.loads(raw)


def _parse_cin_con(cin_response):
    if not isinstance(cin_response, dict):
        return None
    cin = cin_response.get("m2m:cin")
    if not isinstance(cin, dict):
        return None
    con = cin.get("con")
    if con is None:
        return None
    try:
        data = json.loads(con) if isinstance(con, str) else con
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def get_latest_fl_command():
    try:
        resp = _http_json_get(f"{TINYIOT_CSE_NAME}/IN-AE/cnt-fl-control/la", timeout=2.0)
        return _parse_cin_con(resp)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"[fl-control poll] HTTP {e.code}")
        return None
    except Exception:
        # poll 실패는 데모를 죽이지 않음
        return None

def _walk_json(obj):
    """dashboard JSON/SSE event에서 문자열과 dict를 재귀적으로 훑기."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk_json(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_json(v)
    elif isinstance(obj, str):
        yield obj


def _normalize_node_name(value):
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"mn1", "mn2", "mn3"}:
        return s
    if s in {"1", "node1", "mn-1", "mn_1", "MN-AE-1".lower()}:
        return "mn1"
    if s in {"2", "node2", "mn-2", "mn_2", "MN-AE-2".lower()}:
        return "mn2"
    if s in {"3", "node3", "mn-3", "mn_3", "MN-AE-3".lower()}:
        return "mn3"
    return None


def _dashboard_event_to_fault_nodes(event):
    """Dashboard 응답/SSE 문구에서 실제 fault alert node를 추출."""
    nodes = set()

    # 1) dict 구조 우선 처리
    for item in _walk_json(event):
        if isinstance(item, dict):
            node = (
                _normalize_node_name(item.get("node"))
                or _normalize_node_name(item.get("nodeId"))
                or _normalize_node_name(item.get("target"))
                or _normalize_node_name(item.get("name"))
            )

            bool_alert = False
            for key in [
                "fault_detected",
                "faultDetected",
                "anomaly_detected",
                "anomalyDetected",
                "detected",
                "alert",
                "isAlert",
            ]:
                if item.get(key) is True:
                    bool_alert = True

            status_text = " ".join(
                str(item.get(k, ""))
                for k in ["status", "state", "type", "event", "message", "title"]
            ).lower()

            text_alert = (
                "bearing fault detected" in status_text
                or "fault detected" in status_text
                or "anomaly detected" in status_text
                or "fault alert" in status_text
            )

            if node and (bool_alert or text_alert):
                nodes.add(node)

        elif isinstance(item, str):
            s = item.lower()
            # 단순 "fault_transition" 같은 설명 텍스트는 제외하고,
            # 실제 alert/detected 표현만 fault로 인정함.
            has_alert_word = (
                "bearing fault detected" in s
                or "fault detected" in s
                or "anomaly detected" in s
                or "fault alert" in s
            )
            if has_alert_word:
                if "mn1" in s:
                    nodes.add("mn1")
                if "mn2" in s:
                    nodes.add("mn2")
                if "mn3" in s or not nodes:
                    # 현재 데모에서 실제 고장 대상은 mn3.
                    nodes.add("mn3")

    return nodes


def _to_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _dashboard_score_event_values(event):
    """
    dashboard_server.py의 /stream score 이벤트에서 node별 score/threshold/label을 뽑는다.

    예상 score event:
      {
        "type": "score",
        "round": 8,
        "anomaly_active": true,
        "mn3": 2.13,
        "mn3_label": 1,
        "mn3_thr": 1.91
      }
    """
    if not isinstance(event, dict):
        return {}

    if str(event.get("type", "")).lower() != "score":
        return {}

    result = {}
    fl_round = _to_int(event.get("round"), 0)
    anomaly_active = bool(event.get("anomaly_active", False))

    for node in ["mn1", "mn2", "mn3"]:
        if node not in event:
            continue

        score = _to_float(event.get(node), None)
        threshold = _to_float(event.get(f"{node}_thr"), None)
        label = _to_int(event.get(f"{node}_label"), None)

        if score is None or threshold is None:
            continue

        result[node] = {
            "score": score,
            "threshold": threshold,
            "label": label,
            "round": fl_round,
            "anomaly_active": anomaly_active,
            "is_over": score > threshold,
        }

    return result


def _read_dashboard_json(path: str, timeout: float = 1.5):
    url = DASHBOARD_BASE_URL.rstrip("/") + path
    req = urllib.request.Request(
        url=url,
        method="GET",
        headers={"Accept": "application/json,text/event-stream,*/*"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
        try:
            return json.loads(raw)
        except Exception:
            return raw



# =========================================================
# 데이터 로딩 / buffer pkl 생성
# =========================================================

def get_npz_array(npz_obj, candidates, default=None):
    for key in candidates:
        if key in npz_obj.files:
            return np.array(npz_obj[key])
    return default


def load_stream_npz(path: str):
    with np.load(path, allow_pickle=True) as d:
        train_windows = get_npz_array(d, ["train_windows"])
        val_windows = get_npz_array(d, ["val_windows"])
        val_labels = get_npz_array(d, ["val_labels"])

        test_x = get_npz_array(d, ["test_stream_windows"])
        test_y = get_npz_array(d, ["test_stream_labels"], default=None)
        test_states = get_npz_array(d, ["test_stream_state_codes"], default=None)
        test_times = get_npz_array(d, ["test_stream_times"], default=None)
        test_progress = get_npz_array(d, ["test_stream_progress"], default=None)
        rms = get_npz_array(d, ["test_stream_rms"], default=None)
        peak = get_npz_array(d, ["test_stream_peak"], default=None)
        visual_amp = get_npz_array(d, ["test_stream_visual_amp"], default=None)

        norm_mean = get_npz_array(d, ["norm_mean"], default=np.array(0.0, dtype=np.float32))
        norm_std = get_npz_array(d, ["norm_std"], default=np.array(1.0, dtype=np.float32))
        seq_len = get_npz_array(d, ["seq_len"], default=np.array(2560, dtype=np.int64))
        sample_rate = get_npz_array(d, ["sample_rate"], default=np.array(25600, dtype=np.int64))
        window_sec = get_npz_array(d, ["window_sec"], default=np.array(0.1, dtype=np.float32))
        fault_onset_idx = get_npz_array(d, ["fault_onset_idx"], default=np.array(-1, dtype=np.int64))
        node_info = get_npz_array(d, ["node_info"], default=np.array("FEMTO replay"))
        rpm = get_npz_array(d, ["rpm"], default=np.array(0, dtype=np.int64))
        load_N = get_npz_array(d, ["load_N"], default=np.array(0, dtype=np.int64))

    if test_x is None:
        raise RuntimeError(f"test_stream_windows not found in {path}")
    if train_windows is None:
        raise RuntimeError(f"train_windows not found in {path}")
    if val_windows is None:
        raise RuntimeError(f"val_windows not found in {path}")

    test_x = np.array(test_x, dtype=np.float32)
    train_windows = np.array(train_windows, dtype=np.float32)
    val_windows = np.array(val_windows, dtype=np.float32)

    n = len(test_x)

    if test_y is None:
        test_y = np.zeros(n, dtype=np.int64)
    else:
        test_y = np.array(test_y).astype(np.int64).reshape(-1)

    if test_states is None:
        test_states = np.where(test_y > 0, STATE_FAULT, STATE_NORMAL)
    else:
        test_states = np.array(test_states).astype(np.int64).reshape(-1)

    if val_labels is None:
        val_labels = np.zeros(len(val_windows), dtype=np.int64)
    else:
        val_labels = np.array(val_labels).astype(np.int64).reshape(-1)

    if rms is None:
        rms = np.sqrt(np.mean(np.square(test_x), axis=1)).astype(np.float32)
    else:
        rms = np.array(rms).astype(np.float32).reshape(-1)

    if peak is None:
        peak = np.max(np.abs(test_x), axis=1).astype(np.float32)
    else:
        peak = np.array(peak).astype(np.float32).reshape(-1)

    if visual_amp is None:
        visual_amp = np.clip(0.01 + rms * 0.03, 0.01, 0.12).astype(np.float32)
    else:
        visual_amp = np.array(visual_amp).astype(np.float32).reshape(-1)

    if test_times is None:
        test_times = np.arange(n, dtype=np.float32) * 0.1
    else:
        test_times = np.array(test_times).astype(np.float32).reshape(-1)

    if test_progress is None:
        test_progress = np.zeros(n, dtype=np.float32)
    else:
        test_progress = np.array(test_progress).astype(np.float32).reshape(-1)

    return {
        "train_windows": train_windows,
        "val_windows": val_windows,
        "val_labels": val_labels,
        "test_x": test_x,
        "test_y": test_y,
        "test_states": test_states,
        "test_times": test_times,
        "test_progress": test_progress,
        "rms": rms,
        "peak": peak,
        "visual_amp": visual_amp,
        "norm_mean": float(np.asarray(norm_mean).item()),
        "norm_std": float(np.asarray(norm_std).item()),
        "seq_len": int(np.asarray(seq_len).item()),
        "sample_rate": int(np.asarray(sample_rate).item()),
        "window_sec": float(np.asarray(window_sec).item()),
        "fault_onset_idx": int(np.asarray(fault_onset_idx).item()),
        "node_info": str(np.asarray(node_info).item()),
        "rpm": int(np.asarray(rpm).item()),
        "load_N": int(np.asarray(load_N).item()),
    }


def norm_array(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    std = max(float(std), 1e-6)
    return ((arr.astype(np.float32) - float(mean)) / std).astype(np.float32)


def build_fl_dataset(node: str, stream_data):
    mean = stream_data["norm_mean"]
    std = stream_data["norm_std"]

    return {
        "node": node,
        "source": "isaac_sim_femto_sync_gateway",
        "motors": [stream_data["node_info"]],
        "train_signals": norm_array(stream_data["train_windows"], mean, std),
        "val_signals": norm_array(stream_data["val_windows"], mean, std),
        "val_labels": stream_data["val_labels"].astype(np.int64),
        "test_stream_signals": norm_array(stream_data["test_x"], mean, std),
        "test_stream_labels": stream_data["test_y"].astype(np.int64),
        "test_stream_times": stream_data["test_times"].astype(np.float32),
        "test_stream_progress": stream_data["test_progress"].astype(np.float32),
        "test_stream_state_codes": stream_data["test_states"].astype(np.int64),
        "test_stream_rms": stream_data["rms"].astype(np.float32),
        "test_stream_peak": stream_data["peak"].astype(np.float32),
        "test_stream_visual_amp": stream_data["visual_amp"].astype(np.float32),
        "norm_mean": np.float32(mean),
        "norm_std": np.float32(std),
        "seq_len": int(stream_data["seq_len"]),
        "n_channels": 1,
        "sample_rate": int(stream_data["sample_rate"]),
        "window_sec": float(stream_data["window_sec"]),
        "fault_onset_idx": int(stream_data["fault_onset_idx"]),
        "created_by": "isaac_femto_demo_gateway_v7.py",
    }


def write_fl_buffers(streams):
    os.makedirs(BUFFER_WRITE_DIR_WIN, exist_ok=True)

    for node in ["mn1", "mn2", "mn3"]:
        dataset = build_fl_dataset(node, streams[node])
        out_path = os.path.join(BUFFER_WRITE_DIR_WIN, f"{node}.pkl")

        with open(out_path, "wb") as f:
            pickle.dump(dataset, f)

        print(
            f"[buffer] {node}: {out_path} "
            f"train={dataset['train_signals'].shape} "
            f"test={dataset['test_stream_signals'].shape}"
        )


# =========================================================
# oneM2M publish
# =========================================================

def create_content_instance(container_path: str, content, labels=None) -> bool:
    if not ENABLE_PUBLISH:
        print(f"[NO_PUBLISH] {container_path} <- {content}")
        return True

    url = f"{TINYIOT_BASE_URL}/{container_path}"

    cin_obj = {
        "con": json.dumps(content, ensure_ascii=False, separators=(",", ":")),
    }

    if labels:
        cin_obj["lbl"] = labels

    payload = json.dumps({"m2m:cin": cin_obj}).encode("utf-8")

    req = urllib.request.Request(
        url=url,
        data=payload,
        method="POST",
        headers={
            "X-M2M-Origin": ORIGINATOR,
            "X-M2M-RVI": "2a",
            "Content-Type": "application/json;ty=4",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.getcode()
            if status == 201:
                return True
            print(f"[publish fail] {container_path}: status={status}")
            return False

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        print(f"[publish fail] {container_path}: HTTP {e.code} {body[:300]}")
        return False

    except Exception as e:
        print(f"[publish error] {container_path}: {type(e).__name__}: {e}")
        return False


def publish_round_metadata(node: str, round_num: int, start_idx: int, end_idx: int, stream_data) -> bool:
    container_path = f"{TINYIOT_CSE_NAME}/{NODE_AE[node]}/cnt-sensor-data"

    labels = [
        node,
        f"round_{round_num}",
        "type:isaac-sim-sync-buffer",
    ]

    labels_arr = stream_data["test_y"][start_idx:end_idx]
    states_arr = stream_data["test_states"][start_idx:end_idx]
    rms_arr = stream_data["rms"][start_idx:end_idx]
    peak_arr = stream_data["peak"][start_idx:end_idx]

    payload = {
        "type": "isaac-sim-sync-buffer",
        "jobState": "FL_TRAINING",
        "node": node,
        "round": int(round_num),
        "currentRound": int(round_num),
        "data_path": f"{BUFFER_DIR_FOR_WSL}/{node}.pkl",
        "stream_path": f"/mnt/c/Projects/bearing_testbed/data/femto_replay/{node}_stream.npz",
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "window_count": int(end_idx - start_idx),
        "rms_mean": float(np.mean(rms_arr)) if len(rms_arr) else 0.0,
        "peak_max": float(np.max(peak_arr)) if len(peak_arr) else 0.0,
        "state_codes": sorted([int(x) for x in set(states_arr.tolist())]),
        "fault_count": int(np.sum(labels_arr == 1)),
        "timestamp": time.time(),
    }

    ok = create_content_instance(container_path, payload, labels=labels)
    mark = "OK" if ok else "FAIL"
    print(f"[publish {mark}] {node} R{round_num} -> {container_path}")
    return ok


# =========================================================
# 노드 비주얼
# =========================================================

class NodeVisual:
    def __init__(self, root_path: str, name: str, base_pos, stream_data):
        self.root_path = root_path
        self.name = name
        self.base_pos = base_pos
        self.stream = stream_data

        self.path = f"{root_path}/{name.upper()}"
        self.rotor_center_t = None
        self.rotor_group_r = None
        self.pin_prim = None

        self._build()

    def _build(self):
        x, y, z = self.base_pos

        make_xform(self.path, translate=(x, y, z), rotate=(0, 0, 0))

        make_cube(
            f"{self.path}/NodeBase",
            size=(2.35, 1.55, 0.10),
            translate=(0.0, 0.0, 0.10),
            color=COLOR_NODE_BASE,
        )

        make_cube(
            f"{self.path}/CenterGap",
            size=(1.35, 0.48, 0.025),
            translate=(0.08, 0.0, 0.165),
            color=COLOR_DARK_GAP,
        )

        make_cube(
            f"{self.path}/Rail",
            size=(1.55, 0.28, 0.08),
            translate=(0.0, 0.0, 0.24),
            color=COLOR_RAIL,
        )

        make_cube(
            f"{self.path}/Support",
            size=(0.92, 0.72, 0.18),
            translate=(0.0, 0.0, 0.38),
            color=COLOR_SUPPORT,
        )

        make_xform(
            f"{self.path}/HousingGroup",
            translate=(0.0, 0.0, 0.49),
            rotate=(0, 0, 0),
        )

        make_cube(
            f"{self.path}/HousingGroup/HousingBottom",
            size=(1.00, 0.74, 0.12),
            translate=(0.0, 0.0, 0.06),
            color=COLOR_HOUSING,
        )

        make_cube(
            f"{self.path}/HousingGroup/HousingLeftWall",
            size=(1.00, 0.10, 0.60),
            translate=(0.0, -0.34, 0.36),
            color=COLOR_HOUSING,
        )

        make_cube(
            f"{self.path}/HousingGroup/HousingRightWall",
            size=(1.00, 0.10, 0.60),
            translate=(0.0, 0.34, 0.36),
            color=COLOR_HOUSING,
        )

        make_cube(
            f"{self.path}/HousingGroup/HousingTopBridge",
            size=(1.00, 0.74, 0.10),
            translate=(0.0, 0.0, 0.66),
            color=COLOR_HOUSING,
        )

        make_cube(
            f"{self.path}/HousingGroup/HousingBackPlate",
            size=(0.10, 0.74, 0.44),
            translate=(-0.38, 0.0, 0.34),
            color=COLOR_HOUSING,
        )

        _, self.rotor_center_t, _ = make_xform(
            f"{self.path}/RotorCenter",
            translate=(0.42, 0.0, 0.89),
            rotate=(0, 0, 0),
        )

        _, _, self.rotor_group_r = make_xform(
            f"{self.path}/RotorCenter/RotorGroup",
            translate=(0.0, 0.0, 0.0),
            rotate=(0, 0, 0),
        )

        make_cylinder(
            f"{self.path}/RotorCenter/RotorGroup/Shaft",
            radius=0.095,
            height=1.80,
            rotate=(0.0, 90.0, 0.0),
            translate=(0.0, 0.0, 0.0),
            color=COLOR_SHAFT,
        )

        make_cylinder(
            f"{self.path}/RotorCenter/RotorGroup/ShaftNose",
            radius=0.13,
            height=0.34,
            rotate=(0.0, 90.0, 0.0),
            translate=(0.98, 0.0, 0.0),
            color=COLOR_SHAFT,
        )

        # 원통 shaft에 붙어 도는 상태 pin
        pin = make_cube(
            f"{self.path}/RotorCenter/RotorGroup/StatePin",
            size=(0.24, 0.08, 0.08),
            translate=(0.62, 0.0, 0.135),
            rotate=(0.0, 0.0, 0.0),
            color=COLOR_PIN_NORMAL,
        )
        self.pin_prim = pin.GetPrim()

    def get_state_at(self, idx: int):
        max_i = len(self.stream["test_x"]) - 1
        idx = max(0, min(idx, max_i))

        label = int(self.stream["test_y"][idx])
        state = int(self.stream["test_states"][idx])
        rms = float(self.stream["rms"][idx])
        peak = float(self.stream["peak"][idx])

        return {"label": label, "state": state, "rms": rms, "peak": peak}

    def update_visual(self, idx: int, elapsed: float, idle: bool = False, fault_alert: bool = False):
        info = self.get_state_at(idx)

        # 이전 실행의 callback이 남아 있거나 stage가 재생성된 경우,
        # old prim/xform op가 invalid가 될 수 있음.
        # 이때 RuntimeError로 데모 전체가 멈추지 않게 방어함.
        try:
            if self.pin_prim is not None:
                try:
                    if hasattr(self.pin_prim, "IsValid") and not self.pin_prim.IsValid():
                        return info
                except Exception:
                    return info

            angle = (elapsed * NODE_ROTATION_SPEED[self.name]) % 360.0
            self.rotor_group_r.Set(Gf.Vec3f(angle, 0.0, 0.0))

            rms = info["rms"]

            # idle 상태에서도 최근 구간을 반복하므로 RMS 기반 흔들림은 유지함.
            # 단 idle은 active play보다 약간만 낮춰서 화면이 너무 과하지 않게 함.
            if idle:
                amp = min(max(rms * 0.012, 0.002), 0.030)
            else:
                amp = min(max(rms * 0.018, 0.002), 0.045)

            wobble = amp * math.sin(elapsed * 24.0)
            self.rotor_center_t.Set(Gf.Vec3d(0.42, 0.0, 0.89 + wobble))

            # 핵심 수정:
            # Isaac 내부 상태가 DEGRADATION/FAULT여도 pin 색은 바꾸지 않음.
            # Dashboard에서 실제 fault alert를 받은 node만 빨강으로 바꿈.
            pin_color = COLOR_PIN_FAULT if fault_alert else COLOR_PIN_NORMAL
            set_display_color(self.pin_prim, pin_color)

        except RuntimeError as e:
            msg = str(e).lower()
            if "invalid prim" in msg or "expired" in msg or "accessed schema" in msg:
                return info
            raise
        except Exception:
            # 시각화 callback 문제로 FL publish/데모 전체가 죽지 않게 함.
            return info

        return info


# =========================================================
# 메인 컨트롤러
# =========================================================

class FEMTODemoController:
    def __init__(self, stream_dir=STREAM_DIR, visual_speed=VISUAL_SPEED_WPS):
        self.stream_dir = stream_dir
        self.visual_speed = visual_speed

        self.nodes = {}
        self.streams = {}
        self.subscription = None

        self.elapsed = 0.0
        self.last_time = time.time()
        self.accum = 0.0
        self.window_interval = 1.0 / self.visual_speed

        self.current_display_idx = 0
        self.visual_idx = 0
        self.round_start_idx = 0
        self.round_end_idx = 0
        self.max_windows = 0

        self.running = False
        self.round_active = False
        self.current_round = 0
        self.expected_next_round = 1
        self.published_rounds = set()

        self.next_poll_time = 0.0
        self.completed_logged = False

        # idle loop: round 사이에 최근 구간을 반복해서 "정상처럼 멈춰 보이는" 문제 방지
        self.idle_idx = 0
        self.idle_loop_start = 0
        self.idle_loop_end = max(1, IDLE_LOOP_WINDOWS)
        self.idle_accum = 0.0
        self.idle_window_interval = 1.0 / max(IDLE_LOOP_SPEED_WPS, 1e-6)

        # Dashboard fault alert 수신 후 pin 전환
        self.fault_alert_nodes = set()
        self.dashboard_score_consecutive = {"mn1": 0, "mn2": 0, "mn3": 0}
        self.dashboard_last_score_debug = {"mn1": 0.0, "mn2": 0.0, "mn3": 0.0}
        self.dashboard_thread = None
        self.dashboard_thread_stop = False
        self.next_dashboard_poll_time = 0.0

    def load_streams(self):
        loaded = {}
        for name in ["mn1", "mn2", "mn3"]:
            path = os.path.join(self.stream_dir, f"{name}_stream.npz")
            if not os.path.exists(path):
                raise FileNotFoundError(f"stream file not found: {path}")
            print(f"[load] {name}: {path}")
            loaded[name] = load_stream_npz(path)
        return loaded

    def build_scene(self):
        stage = get_stage()
        if stage is None:
            raise RuntimeError("USD stage가 없습니다. Isaac Sim에서 새 stage를 연 뒤 실행하세요.")

        remove_prim_if_exists(ROOT_PATH)

        make_xform(
            ROOT_PATH,
            translate=(0, 0, 0),
            rotate=(0, 0, 0),
        )

        make_cube(
            f"{ROOT_PATH}/CommonBase",
            size=(11.4, 2.5, 0.07),
            translate=(0.0, 0.0, 0.035),
            color=COLOR_COMMON_BASE,
        )

        make_cube(
            f"{ROOT_PATH}/CommonRailFront",
            size=(10.6, 0.13, 0.07),
            translate=(0.0, 0.92, 0.095),
            color=COLOR_RAIL,
        )

        make_cube(
            f"{ROOT_PATH}/CommonRailBack",
            size=(10.6, 0.13, 0.07),
            translate=(0.0, -0.92, 0.095),
            color=COLOR_RAIL,
        )

        self.streams = self.load_streams()
        write_fl_buffers(self.streams)

        self.nodes = {}
        for name in ["mn1", "mn2", "mn3"]:
            self.nodes[name] = NodeVisual(
                ROOT_PATH,
                name,
                NODE_LAYOUT[name],
                self.streams[name],
            )

        self.max_windows = min(
            len(node.stream["test_x"])
            for node in self.nodes.values()
        )

    def publish_round(self, round_num: int):
        if round_num in self.published_rounds:
            return

        start_idx = (round_num - 1) * WINDOWS_PER_ROUND
        end_idx = min(round_num * WINDOWS_PER_ROUND, self.max_windows)

        print(f"\n[sync] FL Round {round_num} detected")
        print(f"[round publish] R{round_num} [{start_idx}:{end_idx}]")

        ok_all = True
        for node in ["mn1", "mn2", "mn3"]:
            ok = publish_round_metadata(
                node,
                round_num,
                start_idx,
                end_idx,
                self.streams[node],
            )
            ok_all = ok_all and ok

        if ok_all:
            self.published_rounds.add(round_num)

        self.current_round = round_num
        self.round_start_idx = start_idx
        self.round_end_idx = end_idx
        self.visual_idx = start_idx
        self.current_display_idx = start_idx
        self.round_active = True
        self.accum = 0.0
        self.idle_accum = 0.0

        self.expected_next_round = round_num + 1

        print(
            f"[visual play] R{round_num} "
            f"{start_idx}->{end_idx} at {self.visual_speed} windows/sec "
            f"(about {(end_idx - start_idx) / self.visual_speed:.1f}s)"
        )

    def handle_fl_command(self, command):
        if not isinstance(command, dict):
            return

        state = str(command.get("jobState", ""))
        try:
            round_num = int(command.get("currentRound", command.get("round", 0)))
        except Exception:
            round_num = 0

        if state == "FL_COMPLETED":
            if not self.completed_logged:
                print("\n[sync] FL_COMPLETED detected.")
                self.completed_logged = True

            # R10 시각화까지 끝났으면 controller 정지.
            # R10 시각화 중이면 끝난 뒤 on_update에서 정지함.
            if self.current_round >= TOTAL_ROUNDS and not self.round_active:
                print("[stop] FL completed and R10 visual already finished.")
                self.stop(silent=True)
            return

        if state != "FL_TRAINING":
            return

        if round_num < 1 or round_num > TOTAL_ROUNDS:
            return

        if round_num in self.published_rounds:
            return

        # stale R10 같은 것을 잘못 잡지 않도록 순서대로만 허용
        if round_num != self.expected_next_round:
            print(
                f"[sync wait] observed FL round={round_num}, "
                f"expected={self.expected_next_round}; ignore stale/out-of-order command"
            )
            return

        self.publish_round(round_num)

    def poll_fl_control_if_needed(self, now: float):
        if now < self.next_poll_time:
            return

        self.next_poll_time = now + FL_CONTROL_POLL_SEC
        command = get_latest_fl_command()
        if command:
            self.handle_fl_command(command)

    def update_all_nodes(self, idle: bool):
        for name in ["mn1", "mn2", "mn3"]:
            node_visual = self.nodes.get(name)
            if node_visual is not None:
                node_visual.update_visual(
                    self.current_display_idx,
                    self.elapsed,
                    idle=idle,
                    fault_alert=(name in self.fault_alert_nodes),
                )

    def _set_idle_loop_for_round(self):
        if self.current_round <= 0:
            self.idle_loop_start = 0
            self.idle_loop_end = min(IDLE_LOOP_WINDOWS, self.max_windows)
        else:
            self.idle_loop_start = max(self.round_start_idx, self.round_end_idx - IDLE_LOOP_WINDOWS)
            self.idle_loop_end = max(self.idle_loop_start + 1, self.round_end_idx)

        self.idle_idx = self.idle_loop_start
        self.idle_accum = 0.0

        print(
            f"[visual idle-loop] repeat idx "
            f"{self.idle_loop_start}:{self.idle_loop_end} "
            f"at {IDLE_LOOP_SPEED_WPS} windows/sec"
        )

    def _advance_idle_loop_if_needed(self, dt: float):
        if self.round_active:
            return

        if self.max_windows <= 0:
            self.current_display_idx = 0
            return

        self.idle_accum += dt
        if self.idle_accum < self.idle_window_interval:
            return

        self.idle_accum = 0.0

        if self.idle_loop_end <= self.idle_loop_start:
            self.idle_loop_start = 0
            self.idle_loop_end = min(IDLE_LOOP_WINDOWS, self.max_windows)

        self.current_display_idx = self.idle_idx
        self.idle_idx += 1

        if self.idle_idx >= self.idle_loop_end:
            self.idle_idx = self.idle_loop_start

    def _register_dashboard_fault_nodes(self, nodes):
        new_nodes = set(nodes) - self.fault_alert_nodes
        if not new_nodes:
            return

        self.fault_alert_nodes.update(new_nodes)
        print(f"[dashboard alert] fault detected by dashboard: {sorted(new_nodes)}")
        print(f"[pin] red nodes now: {sorted(self.fault_alert_nodes)}")

    def _process_dashboard_event(self, obj):
        """
        Dashboard /stream 이벤트 처리.

        v6 문제:
        - Dashboard 서버가 "Bearing Fault Detected" 문자열을 SSE로 직접 보내는 줄 알고
          문자열/boolean alert만 찾았음.
        - 실제로는 score event가 오고, 브라우저 대시보드가 score > threshold를 보고
          toast를 띄우는 구조라 Isaac 쪽에서는 감지를 못 했음.

        v7 수정:
        - score event의 mn*_score, mn*_thr, mn*_label 값을 직접 비교.
        - label==1인 fault stream에서 score > threshold가 3회 연속이면 pin 빨강.
        """
        explicit_nodes = _dashboard_event_to_fault_nodes(obj)
        if explicit_nodes:
            self._register_dashboard_fault_nodes(explicit_nodes)
            return

        score_values = _dashboard_score_event_values(obj)
        if not score_values:
            return

        now = time.time()

        for node in ["mn1", "mn2", "mn3"]:
            values = score_values.get(node)
            if not values:
                continue

            score = values["score"]
            threshold = values["threshold"]
            label = values["label"]
            fl_round = values["round"]
            anomaly_active = values["anomaly_active"]
            is_over = values["is_over"]

            # label이 있는 경우에는 실제 fault label 구간에서만 alert 후보로 인정.
            # 이렇게 해야 MN1/MN2 false spike 때문에 빨개지는 걸 막을 수 있음.
            if DASHBOARD_ALERT_REQUIRE_LABEL and label is not None:
                label_ok = (label == 1)
            else:
                label_ok = True

            # 현재 dashboard demo 설정상 fault stream은 anomaly_active 이후에 의미 있음.
            active_ok = anomaly_active or (label == 1)

            if is_over and label_ok and active_ok:
                self.dashboard_score_consecutive[node] += 1

                # 너무 많이 찍히지 않게 처음/확정 순간 위주로 로그.
                if (
                    self.dashboard_score_consecutive[node] == 1
                    or self.dashboard_score_consecutive[node] == DASHBOARD_ALERT_K_CONSECUTIVE
                    or now - self.dashboard_last_score_debug[node] > 3.0
                ):
                    print(
                        f"[dashboard score] {node} "
                        f"round={fl_round} "
                        f"score={score:.4f} > thr={threshold:.4f} "
                        f"label={label} "
                        f"count={self.dashboard_score_consecutive[node]}/"
                        f"{DASHBOARD_ALERT_K_CONSECUTIVE}"
                    )
                    self.dashboard_last_score_debug[node] = now

                if self.dashboard_score_consecutive[node] >= DASHBOARD_ALERT_K_CONSECUTIVE:
                    self._register_dashboard_fault_nodes({node})

            else:
                if self.dashboard_score_consecutive.get(node, 0) > 0:
                    print(
                        f"[dashboard score] {node} reset "
                        f"score={score:.4f}, thr={threshold:.4f}, label={label}"
                    )
                self.dashboard_score_consecutive[node] = 0

    def _poll_dashboard_snapshot_if_needed(self, now: float):
        if now < self.next_dashboard_poll_time:
            return

        self.next_dashboard_poll_time = now + DASHBOARD_POLL_SEC

        for path in ["/health", "/status", "/api/status", "/api/state"]:
            try:
                obj = _read_dashboard_json(path, timeout=0.8)
                before = set(self.fault_alert_nodes)
                self._process_dashboard_event(obj)
                if self.fault_alert_nodes != before:
                    return
            except Exception:
                pass

    def _dashboard_stream_worker(self):
        """Dashboard /stream SSE에서 fault alert 이벤트를 들음."""
        stream_url = DASHBOARD_BASE_URL.rstrip("/") + DASHBOARD_STREAM_PATH

        while not self.dashboard_thread_stop:
            try:
                req = urllib.request.Request(
                    url=stream_url,
                    method="GET",
                    headers={"Accept": "text/event-stream"},
                )

                with urllib.request.urlopen(req, timeout=10.0) as resp:
                    for raw_line in resp:
                        if self.dashboard_thread_stop:
                            return

                        try:
                            line = raw_line.decode("utf-8", errors="ignore").strip()
                        except Exception:
                            continue

                        if not line:
                            continue

                        payload = line
                        if line.startswith("data:"):
                            payload = line[5:].strip()

                        obj = payload
                        try:
                            obj = json.loads(payload)
                        except Exception:
                            pass

                        self._process_dashboard_event(obj)

            except Exception:
                # dashboard가 아직 안 켜졌거나 SSE 연결이 끊기면 잠깐 기다렸다 재연결
                time.sleep(1.0)

    def _start_dashboard_listener(self):
        self.dashboard_thread_stop = False
        self.dashboard_thread = threading.Thread(
            target=self._dashboard_stream_worker,
            name="femto_dashboard_stream_listener",
            daemon=True,
        )
        self.dashboard_thread.start()
        print(f"[dashboard] listening {DASHBOARD_BASE_URL}{DASHBOARD_STREAM_PATH}")

    def on_update(self, event):
        if not self.running:
            return

        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        self.elapsed += dt
        self.accum += dt

        self.poll_fl_control_if_needed(now)
        self._poll_dashboard_snapshot_if_needed(now)

        idle = not self.round_active

        if idle:
            # round 사이 대기 중에도 마지막 구간을 반복 재생해서
            # 상태/진동이 완전히 정상처럼 멈춰 보이지 않게 함.
            self._advance_idle_loop_if_needed(dt)
            self.update_all_nodes(idle=True)
            return

        # active round 시각화
        if self.accum < self.window_interval:
            self.update_all_nodes(idle=False)
            return

        self.accum = 0.0

        if self.visual_idx % 20 == 0:
            print(f"\n[visual] R{self.current_round} idx={self.visual_idx}")
            for name in ["mn1", "mn2", "mn3"]:
                info = self.nodes[name].update_visual(
                    self.visual_idx,
                    self.elapsed,
                    idle=False,
                    fault_alert=(name in self.fault_alert_nodes),
                )
                print(
                    f"  {name}: {STATE_NAME[info['state']]:<12} "
                    f"label={info['label']} "
                    f"rms={info['rms']:.4f} "
                    f"peak={info['peak']:.4f} "
                    f"pin={'RED' if name in self.fault_alert_nodes else 'GREEN'}"
                )

        self.current_display_idx = self.visual_idx
        self.update_all_nodes(idle=False)
        self.visual_idx += 1

        if self.visual_idx >= self.round_end_idx:
            self.current_display_idx = max(self.round_end_idx - 1, 0)
            self.round_active = False

            if self.current_round >= TOTAL_ROUNDS:
                print("[visual done] R10 visual finished.")
                print("[stop] FEMTO Isaac demo stopped after final round.")
                self.stop(silent=True)
                return

            print(
                f"[visual idle] R{self.current_round} visual finished. "
                "Waiting for next FL round command..."
            )
            self._set_idle_loop_for_round()

    def start(self):
        self.stop(silent=True)

        self.elapsed = 0.0
        self.last_time = time.time()
        self.accum = 0.0

        self.current_display_idx = 0
        self.visual_idx = 0
        self.round_start_idx = 0
        self.round_end_idx = 0

        self.round_active = False
        self.current_round = 0
        self.expected_next_round = 1
        self.published_rounds = set()
        self.next_poll_time = 0.0
        self.completed_logged = False

        self.idle_idx = 0
        self.idle_loop_start = 0
        self.idle_loop_end = max(1, IDLE_LOOP_WINDOWS)
        self.idle_accum = 0.0
        self.fault_alert_nodes = set()
        self.dashboard_score_consecutive = {"mn1": 0, "mn2": 0, "mn3": 0}
        self.dashboard_last_score_debug = {"mn1": 0.0, "mn2": 0.0, "mn3": 0.0}
        self.next_dashboard_poll_time = 0.0
        self.dashboard_thread_stop = False

        self.build_scene()

        app = omni.kit.app.get_app()
        self.subscription = app.get_update_event_stream().create_subscription_to_pop(
            self.on_update,
            name="femto_isaac_demo_gateway_v7",
        )
        self.running = True
        self._start_dashboard_listener()
        self._set_idle_loop_for_round()

        print("[start] FEMTO Isaac synced demo started - v7")
        print(f"        stream_dir={self.stream_dir}")
        print(f"        buffer_dir_win={BUFFER_WRITE_DIR_WIN}")
        print(f"        buffer_dir_wsl={BUFFER_DIR_FOR_WSL}")
        print(f"        visual_speed={self.visual_speed} windows/sec")
        print(f"        round visual duration ≈ {WINDOWS_PER_ROUND / self.visual_speed:.1f} sec")
        print(f"        fl-control poll={FL_CONTROL_POLL_SEC} sec")
        print(f"        publish={ENABLE_PUBLISH}")
        print("        world order: left=MN1, center=MN2, right=MN3")
        print("        pin color: green until dashboard score alert, then red")
        print("        Waiting for IN-AE FL_TRAINING round command...")
        print("        stop command: stop_femto_demo()")

    def stop(self, silent=False):
        if self.subscription is not None:
            try:
                self.subscription = None
            except Exception:
                pass

        self.running = False
        self.dashboard_thread_stop = True

        if not silent:
            print("[stop] FEMTO Isaac synced demo stopped")


# =========================================================
# 전역 함수
# =========================================================

try:
    _FEMTO_DEMO_CONTROLLER
except NameError:
    _FEMTO_DEMO_CONTROLLER = None


def start_femto_demo(stream_dir=STREAM_DIR, speed=VISUAL_SPEED_WPS):
    global _FEMTO_DEMO_CONTROLLER

    # 재실행 시 이전 subscription/callback을 먼저 정리해야
    # "Accessed schema on invalid prim" 오류가 반복되지 않음.
    old_controller = _FEMTO_DEMO_CONTROLLER
    if old_controller is not None:
        try:
            old_controller.stop(silent=True)
            print("[restart] previous FEMTO Isaac demo stopped")
        except Exception:
            pass

    _FEMTO_DEMO_CONTROLLER = FEMTODemoController(
        stream_dir=stream_dir,
        visual_speed=speed,
    )
    _FEMTO_DEMO_CONTROLLER.start()


def stop_femto_demo():
    global _FEMTO_DEMO_CONTROLLER
    if _FEMTO_DEMO_CONTROLLER is not None:
        try:
            _FEMTO_DEMO_CONTROLLER.stop()
        finally:
            _FEMTO_DEMO_CONTROLLER = None
    else:
        print("[stop] no FEMTO Isaac demo controller")


start_femto_demo()
