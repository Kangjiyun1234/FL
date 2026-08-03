"""
fl/dashboard_server.py — FL 대시보드 백엔드

백그라운드 스레드:

1. fl-poll
   - oneM2M 상태 폴링
   - 새로운 글로벌 모델 감지
   - validation threshold 재계산
   - test stream reconstruction score 재계산

2. score-stream
   - 테스트 샘플의 재구성 오차를 SSE로 전송
   - score_idx는 글로벌 모델이 갱신돼도 유지
   - MN3는 설정된 라운드 전에는 정상 데이터만 표시
   - 설정된 라운드부터 이상 데이터만 표시

SSE 이벤트:

  init
    접속 직후 현재 상태

  round
    FL 상태 및 라운드 변경

  threshold
    글로벌 모델에 따른 threshold 재계산

  score
    노드별 reconstruction error

  summary
    최종 평가 결과

실행:

  python3 fl/dashboard_server.py

주소:

  http://localhost:7000
"""

from __future__ import annotations

import json
import os
import pickle
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch
from flask import Flask, Response
from sklearn.metrics import roc_auc_score

import config
import onem2m_utils as om2m
from model import Conv1DAE


# ════════════════════════════════════════════════════════
# 기본 설정
# ════════════════════════════════════════════════════════

PORT = int(
    os.getenv(
        "FL_DASHBOARD_PORT",
        "7000",
    )
)
POLL_INTERVAL = float(
    os.getenv(
        "FL_DASHBOARD_POLL_INTERVAL",
        "3.0",
    )
)
SCORE_INTERVAL = float(
    os.getenv(
        "FL_DASHBOARD_SCORE_INTERVAL",
        "0.5",
    )
)
N_SIGMA = float(
    getattr(
        config,
        "THRESHOLD_N_SIGMA",
        3.0,
    )
)
K_CONSECUTIVE = int(
    getattr(
        config,
        "ANOMALY_K_CONSECUTIVE",
        3,
    )
)
ANOMALY_NODE = str(
    getattr(
        config,
        "ANOMALY_DEMO_NODE",
        "mn3",
    )
)
ANOMALY_START_ROUND = int(
    getattr(
        config,
        "ANOMALY_START_ROUND",
        7,
    )
)
PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]
HTML_PATH = (
    PROJECT_ROOT
    / "fl_bearing_dashboard.html"
)
GLOBAL_MODEL_DIR = Path(
    getattr(
        config,
        "GLOBAL_MODEL_DIR",
        "/tmp/fl_models/global",
    )
)
PKL_DIR = Path(
    getattr(
        config,
        "FEMTO_DATA_DIR",
        "/tmp/fl_data/femto",
    )
)
RUN_MARKER_PATH = Path(
    getattr(
        config,
        "DEMO_RUN_MARKER",
        "/tmp/fl_models/.demo_run_started",
    )
)
NODES = [
    "mn1",
    "mn2",
    "mn3",
]

app = Flask(
    __name__,
)


# ════════════════════════════════════════════════════════
# SSE 구독자
# ════════════════════════════════════════════════════════

_subscribers: list[queue.Queue] = []
_subscribers_lock = threading.Lock()


# ════════════════════════════════════════════════════════
# 공유 상태
# ════════════════════════════════════════════════════════

_shared: dict = {
    "fl_state": "FL_READY",
    "round": 0,
    "max_rounds": config.GLOBAL_ROUNDS,

    # node → local training metrics
    "nodes": {},

    # node → threshold
    "thresholds": {},

    # node → reconstruction scores
    "scores": {},

    # node → test labels
    "labels": {},

    # 글로벌 모델이 변경돼도 유지되는 스트림 위치
    "score_idx": 0,

    # 현재 Dashboard가 반영한 global model round
    "model_round": -1,

    "summary_sent": False,

    "anomaly_start_round": ANOMALY_START_ROUND,
}

_shared_lock = threading.Lock()


# ════════════════════════════════════════════════════════
# SSE 헬퍼
# ════════════════════════════════════════════════════════

def _broadcast(event: dict) -> None:
    """
    현재 연결된 모든 SSE 클라이언트에 이벤트를 전송한다.
    """

    message = (
        "data: "
        + json.dumps(
            event,
            ensure_ascii=False,
        )
        + "\n\n"
    )

    with _subscribers_lock:
        disconnected = []

        for subscriber in _subscribers:
            try:
                subscriber.put_nowait(
                    message,
                )
            except queue.Full:
                disconnected.append(
                    subscriber,
                )
        for subscriber in disconnected:
            if subscriber in _subscribers:
                _subscribers.remove(
                    subscriber,
                )


# ════════════════════════════════════════════════════════
# HTTP Endpoint
# ════════════════════════════════════════════════════════

@app.route("/")
def index():
    if not HTML_PATH.exists():
        return (
            f"Dashboard HTML not found: {HTML_PATH}<br>"
            "프로젝트 루트에 "
            "fl_bearing_dashboard.html 파일이 있는지 확인하세요.",
            500,
        )

    return HTML_PATH.read_text(
        encoding="utf-8",
    )


@app.route("/health")
def health():
    try:
        global_model_files = sorted(
            path.name
            for path in GLOBAL_MODEL_DIR.glob(
                "global_round*.pt"
            )
        )
    except OSError:
        global_model_files = []

    pkl_files = {
        node: (
            PKL_DIR
            / f"{node}.pkl"
        ).exists()
        for node in NODES
    }

    with _shared_lock:
        state = {
            "ok": True,

            "oneM2M": config.BASE_URL,

            "state": _shared["fl_state"],
            "round": _shared["round"],
            "max_rounds": _shared["max_rounds"],
            "model_round": _shared["model_round"],

            "html_path": str(
                HTML_PATH,
            ),
            "html_exists": HTML_PATH.exists(),

            "global_model_dir": str(
                GLOBAL_MODEL_DIR,
            ),
            "global_model_dir_exists": (
                GLOBAL_MODEL_DIR.exists()
            ),
            "global_model_files": global_model_files,

            "pkl_dir": str(
                PKL_DIR,
            ),
            "pkl_dir_exists": PKL_DIR.exists(),
            "pkl_files": pkl_files,

            "nodes": _shared["nodes"],
            "thresholds": _shared["thresholds"],
            "score_idx": _shared["score_idx"],
            "summary_sent": _shared["summary_sent"],

            "anomaly_node": ANOMALY_NODE,
            "anomaly_start_round": (
                ANOMALY_START_ROUND
            ),

            "run_marker": str(
                RUN_MARKER_PATH,
            ),
            "run_marker_exists": (
                RUN_MARKER_PATH.exists()
            ),
        }

    return state


@app.route("/stream")
def stream():
    subscriber: queue.Queue = queue.Queue(
        maxsize=150,
    )
    with _subscribers_lock:
        _subscribers.append(
            subscriber,
        )

    # 접속 직후 현재 상태 전송
    with _shared_lock:
        snapshot = {
            "type": "init",
            "fl_state": _shared["fl_state"],
            "round": _shared["round"],
            "max_rounds": _shared["max_rounds"],
            "nodes": _shared["nodes"],
            "thresholds": _shared["thresholds"],
            "model_round": _shared["model_round"],
        }

    try:
        subscriber.put_nowait(
            "data: "
            + json.dumps(
                snapshot,
                ensure_ascii=False,
            )
            + "\n\n"
        )
    except queue.Full:
        pass

    def generate():
        try:
            while True:
                try:
                    yield subscriber.get(
                        timeout=25,
                    )
                except queue.Empty:
                    yield ": heartbeat\n\n"

        except GeneratorExit:
            pass

        finally:
            with _subscribers_lock:
                if subscriber in _subscribers:
                    _subscribers.remove(
                        subscriber,
                    )

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ════════════════════════════════════════════════════════
# 현재 데모 실행 시작 시각
# ════════════════════════════════════════════════════════

def _current_run_started_at() -> float:
    """
    clean_fl.sh가 생성한 실행 표식의 수정 시각을 반환한다.

    이 시각보다 오래된 global model은 이전 데모 실행에서
    남은 stale model로 간주한다.
    """

    try:
        return RUN_MARKER_PATH.stat().st_mtime
    except OSError:
        return 0.0


# ════════════════════════════════════════════════════════
# 최신 글로벌 모델 탐색
# ════════════════════════════════════════════════════════

def _find_latest_model(
    max_round: int | None = None,
) -> tuple[int, Path | None]:
    """
    현재 데모 실행에서 생성된 글로벌 모델 중
    가장 높은 라운드의 파일을 반환한다.

    다음 파일은 제외한다.

    1. 현재 FL round보다 미래 라운드인 모델
    2. 데모 실행 표식보다 오래된 모델
    3. global_roundN.pt 형식이 아닌 모델
    """

    if not GLOBAL_MODEL_DIR.exists():
        return -1, None

    run_started_at = _current_run_started_at()

    latest_round = -1
    latest_path: Path | None = None

    for model_path in GLOBAL_MODEL_DIR.glob(
        "global_round*.pt"
    ):
        try:
            model_round = int(
                model_path.stem.replace(
                    "global_round",
                    "",
                )
            )
        except ValueError:
            continue

        if (
            max_round is not None
            and model_round > max_round
        ):
            continue

        try:
            model_modified_at = (
                model_path.stat().st_mtime
            )
        except OSError:
            continue

        if (
            run_started_at
            and model_modified_at < run_started_at
        ):
            continue
        if model_round > latest_round:
            latest_round = model_round
            latest_path = model_path

    return latest_round, latest_path


# ════════════════════════════════════════════════════════
# 모델 로드
# ════════════════════════════════════════════════════════

def _load_model(
    model_path: str | Path,
) -> Conv1DAE:
    ae_config = config.AE_CFG

    model = Conv1DAE(
        n_channels=ae_config.n_channels,
        latent_dim=ae_config.latent_dim,
        seq_len=ae_config.seq_len,
    )

    state_dict = torch.load(
        model_path,
        map_location="cpu",
    )

    model.load_state_dict(
        state_dict,
    )

    model.eval()

    return model


# ════════════════════════════════════════════════════════
# Reconstruction score 계산
# ════════════════════════════════════════════════════════

def _compute_scores(
    model: Conv1DAE,
    signals: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    """
    signals shape:
      (N, sequence_length)

    return shape:
      (N,)
    """

    if len(signals) == 0:
        return np.array(
            [],
            dtype=np.float32,
        )

    score_batches: list[np.ndarray] = []

    tensor = torch.tensor(
        signals,
        dtype=torch.float32,
    )

    with torch.no_grad():
        for start in range(
            0,
            len(tensor),
            batch_size,
        ):
            batch = tensor[
                start:start + batch_size
            ].unsqueeze(1)

            reconstruction = model(
                batch,
            )

            error = (
                (
                    batch
                    - reconstruction
                )
                ** 2
            ).mean(
                dim=(1, 2),
            )
            score_batches.append(
                error.cpu().numpy(),
            )

    return np.concatenate(
        score_batches,
    )


# ════════════════════════════════════════════════════════
# Threshold 계산
# ════════════════════════════════════════════════════════

def _compute_threshold(
    model: Conv1DAE,
    validation_signals: np.ndarray,
    validation_labels: np.ndarray,
) -> float:
    normal_signals = validation_signals[
        validation_labels == 0
    ]

    if len(normal_signals) == 0:
        return 1.0

    scores = _compute_scores(
        model,
        normal_signals,
    )

    return float(
        scores.mean()
        + N_SIGMA * scores.std()
    )


# ════════════════════════════════════════════════════════
# 글로벌 모델 변경 시 score 재계산
# ════════════════════════════════════════════════════════

def _reload_scores_if_needed() -> None:
    """
    새로운 글로벌 모델이 생성되면:

    1. validation threshold 재계산
    2. test stream reconstruction score 재계산
    3. 기존 score_idx 유지

    기존 코드처럼 score_idx를 0으로 되돌리지 않는다.
    """

    with _shared_lock:
        current_model_round = _shared[
            "model_round"
        ]
        current_fl_round = _shared[
            "round"
        ]

    model_round, model_path = _find_latest_model(
        max_round=current_fl_round,
    )

    if (
        model_round <= current_model_round
        or model_path is None
    ):
        return

    print(
        "  [Scores] "
        f"새 글로벌 모델 R{model_round} 감지"
    )

    try:
        model = _load_model(
            model_path,
        )
    except Exception as error:
        print(
            "  ✗ 글로벌 모델 로드 실패: "
            f"{error}"
        )
        return

    new_thresholds: dict[str, float] = {}
    new_scores: dict[str, np.ndarray] = {}
    new_labels: dict[str, np.ndarray] = {}

    for node in NODES:
        pkl_path = (
            PKL_DIR
            / f"{node}.pkl"
        )

        if not pkl_path.exists():
            print(
                f"  ⚠ PKL 파일 없음: {pkl_path}"
            )
            continue

        try:
            with pkl_path.open(
                "rb",
            ) as file:
                dataset = pickle.load(
                    file,
                )
        except Exception as error:
            print(
                f"  ✗ {node} PKL 로드 실패: "
                f"{error}"
            )
            continue

        sequence_length = (
            config.AE_CFG.seq_len
        )
        empty_signals = np.empty(
            (
                0,
                sequence_length,
            ),
            dtype=np.float32,
        )
        empty_labels = np.empty(
            (0,),
            dtype=np.int64,
        )
        validation_signals = dataset.get(
            "val_signals",
            empty_signals,
        )
        validation_labels = dataset.get(
            "val_labels",
            empty_labels,
        )
        test_signals = dataset.get(
            "test_stream_signals",
            empty_signals,
        )
        test_labels = dataset.get(
            "test_stream_labels",
            empty_labels,
        )
        threshold = _compute_threshold(
            model,
            validation_signals,
            validation_labels,
        )
        reconstruction_scores = _compute_scores(
            model,
            test_signals,
        )
        new_thresholds[node] = round(
            threshold,
            4,
        )
        new_scores[node] = (
            reconstruction_scores
        )
        new_labels[node] = test_labels

        print(
            f"    {node}: "
            f"n={len(reconstruction_scores)}, "
            f"threshold={threshold:.4f}"
        )

    with _shared_lock:
        # 핵심 수정:
        # 글로벌 모델이 바뀌더라도 기존 스트림 진행 위치를 유지한다.
        current_score_idx = _shared[
            "score_idx"
        ]

        _shared["thresholds"] = (
            new_thresholds
        )

        _shared["scores"] = new_scores
        _shared["labels"] = new_labels

        _shared["score_idx"] = (
            current_score_idx
        )

        _shared["model_round"] = (
            model_round
        )

        _shared["summary_sent"] = False

    _broadcast(
        {
            "type": "threshold",
            "model_round": model_round,
            "thresholds": new_thresholds,
        }
    )

    print(
        f"  ✓ R{model_round} score 재계산 완료"
    )


# ════════════════════════════════════════════════════════
# 최종 Summary 계산
# ════════════════════════════════════════════════════════

def _compute_summary() -> dict | None:
    with _shared_lock:
        scores = {
            key: value.copy()
            for key, value
            in _shared["scores"].items()
        }
        labels = {
            key: value.copy()
            for key, value
            in _shared["labels"].items()
        }
        thresholds = dict(
            _shared["thresholds"],
        )
        model_round = _shared[
            "model_round"
        ]
        max_rounds = _shared[
            "max_rounds"
        ]

    if (
        model_round < max_rounds
        or not scores
    ):
        return None

    result = {
        "type": "summary",
        "model_round": model_round,
        "nodes": {},
    }

    for node in NODES:
        node_scores = scores.get(
            node,
        )
        node_labels = labels.get(
            node,
        )
        threshold = thresholds.get(
            node,
            1.0,
        )

        if (
            node_scores is None
            or node_labels is None
            or len(node_scores) == 0
        ):
            continue

        try:
            if len(
                np.unique(
                    node_labels,
                )
            ) > 1:
                auroc = float(
                    roc_auc_score(
                        node_labels,
                        node_scores,
                    )
                )
            else:
                auroc = 0.0
        except Exception:
            auroc = 0.0

        onset_index = next(
            (
                int(index)
                for index, label
                in enumerate(node_labels)
                if label == 1
            ),
            None,
        )

        detection_index = None
        consecutive = 0

        for index, score in enumerate(
            node_scores,
        ):
            if score > threshold:
                consecutive += 1

                if consecutive >= K_CONSECUTIVE:
                    detection_index = int(
                        index
                        - K_CONSECUTIVE
                        + 1
                    )
                    break
            else:
                consecutive = 0

        if (
            detection_index is not None
            and onset_index is not None
        ):
            detection_delay = int(
                detection_index
                - onset_index
            )
        else:
            detection_delay = None

        false_alarms = 0
        false_alarm_consecutive = 0

        for score, label in zip(
            node_scores,
            node_labels,
        ):
            if label != 0:
                continue

            if score > threshold:
                false_alarm_consecutive += 1
                if (
                    false_alarm_consecutive
                    >= K_CONSECUTIVE
                ):
                    false_alarms += 1
                    false_alarm_consecutive = 0
            else:
                false_alarm_consecutive = 0

        result["nodes"][node] = {
            "auroc": round(
                auroc,
                4,
            ),
            "threshold": round(
                float(threshold),
                4,
            ),
            "onset_idx": onset_index,
            "detect_idx": detection_index,
            "delay": detection_delay,
            "false_alarms": false_alarms,

            "n_normal": int(
                np.sum(
                    node_labels == 0
                )
            ),
            "n_anomaly": int(
                np.sum(
                    node_labels == 1
                )
            ),
        }

    return result


# ════════════════════════════════════════════════════════
# 표시할 테스트 스트림 인덱스 선택
# ════════════════════════════════════════════════════════

def _select_stream_index(
    node: str,
    node_labels: np.ndarray,
    stream_index: int,
    fl_round: int,
) -> int | None:
    """
    노드와 현재 FL round에 따라 표시할 샘플 인덱스를 선택한다.
    기본 동작:
      MN1:
        정상 샘플 반복
      MN2:
        정상 샘플 반복
      MN3 Round 1~6:
        정상 샘플 반복
      MN3 Round 7 이상:
        이상 샘플 반복
    """

    if len(node_labels) == 0:
        return None

    if node == ANOMALY_NODE:
        if fl_round >= ANOMALY_START_ROUND:
            target_label = 1
        else:
            target_label = 0
    else:
        target_label = 0

    candidate_indices = np.flatnonzero(
        node_labels == target_label
    )

    # 해당 라벨이 없다면 전체 데이터 사용
    if len(candidate_indices) == 0:
        candidate_indices = np.arange(
            len(node_labels),
        )
    if len(candidate_indices) == 0:
        return None

    selected_position = (
        stream_index
        % len(candidate_indices)
    )

    return int(
        candidate_indices[
            selected_position
        ]
    )


# ════════════════════════════════════════════════════════
# Score streaming thread
# ════════════════════════════════════════════════════════

def _score_thread() -> None:
    """
    테스트 reconstruction score를 주기적으로 전송한다.
    score_idx는 글로벌 모델이 변경돼도 초기화하지 않는다.
    """

    while True:
        time.sleep(
            SCORE_INTERVAL,
        )

        with _shared_lock:
            scores = _shared["scores"]
            labels = _shared["labels"]
            thresholds = _shared["thresholds"]
            stream_index = _shared["score_idx"]
            fl_state = _shared["fl_state"]
            fl_round = _shared["round"]

        if (
            not scores
            or fl_state == "FL_COMPLETED"
        ):
            continue

        event: dict = {
            "type": "score",
            "round": fl_round,
            "anomaly_active": (
                fl_round
                >= ANOMALY_START_ROUND
            ),
        }

        for node in NODES:
            node_scores = scores.get(
                node,
            )
            node_labels = labels.get(
                node,
            )

            if (
                node_scores is None
                or node_labels is None
                or len(node_scores) == 0
                or len(node_labels) == 0
            ):
                continue

            source_index = _select_stream_index(
                node=node,
                node_labels=node_labels,
                stream_index=stream_index,
                fl_round=fl_round,
            )

            if source_index is None:
                continue

            if source_index >= len(node_scores):
                continue

            event[node] = round(
                float(
                    node_scores[source_index]
                ),
                5,
            )
            event[f"{node}_label"] = int(
                node_labels[source_index]
            )
            event[f"{node}_thr"] = (
                thresholds.get(
                    node,
                    1.0,
                )
            )

        # type, round, anomaly_active 이외에
        # 실제 노드 데이터가 하나 이상 들어간 경우만 전송
        if len(event) > 3:
            with _shared_lock:
                _shared["score_idx"] = (
                    stream_index + 1
                )
            _broadcast(
                event,
            )


# ════════════════════════════════════════════════════════
# oneM2M CIN 파싱
# ════════════════════════════════════════════════════════

def _parse_content_instance(
    content_instance: dict | None,
) -> dict | None:
    if (
        not content_instance
        or "m2m:cin" not in content_instance
    ):
        return None

    content = (
        content_instance["m2m:cin"]
        .get("con")
    )

    try:
        if isinstance(
            content,
            str,
        ):
            return json.loads(
                content,
            )
        if isinstance(
            content,
            dict,
        ):
            return content

    except (
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None

    return None


# ════════════════════════════════════════════════════════
# FL 상태 polling thread
# ════════════════════════════════════════════════════════

def _poll_thread() -> None:
    fl_control_path = (
        f"{config.CSE_NAME}/"
        f"{config.IN_AE_NAME}/"
        "cnt-fl-control"
    )

    local_updates_path = (
        f"{config.CSE_NAME}/"
        f"{config.IN_AE_NAME}/"
        "cnt-local-updates"
    )

    previous_snapshot: dict = {}

    while True:
        try:
            control_content = (
                _parse_content_instance(
                    om2m.get_latest_content_instance(
                        fl_control_path,
                    )
                )
            )

            if control_content:
                fl_state = control_content.get(
                    "jobState",
                    "FL_READY",
                )
                round_number = int(
                    control_content.get(
                        "currentRound",
                        0,
                    )
                )
                max_rounds = int(
                    control_content.get(
                        "maxRounds",
                        config.GLOBAL_ROUNDS,
                    )
                )

                nodes: dict = {}

                for node in NODES:
                    node_content = (
                        _parse_content_instance(
                            om2m.get_latest_content_instance(
                                (
                                    f"{local_updates_path}/"
                                    f"cnt-{node}"
                                )
                            )
                        )
                    )

                    if not node_content:
                        continue

                    nodes[node] = {
                        "train_loss": round(
                            float(
                                node_content.get(
                                    "train_loss",
                                    0,
                                )
                            ),
                            5,
                        ),
                        "val_loss": round(
                            float(
                                node_content.get(
                                    "val_loss",
                                    0,
                                )
                            ),
                            5,
                        ),
                        "val_auroc": round(
                            float(
                                node_content.get(
                                    "val_auroc",
                                    0,
                                )
                            ),
                            4,
                        ),
                        "num_samples": int(
                            node_content.get(
                                "num_samples",
                                0,
                            )
                        ),
                        "round": int(
                            node_content.get(
                                "round",
                                0,
                            )
                        ),
                    }

                snapshot = {
                    "state": fl_state,
                    "round": round_number,

                    "node_keys": sorted(
                        nodes.keys(),
                    ),

                    "losses": {
                        node: data[
                            "train_loss"
                        ]
                        for node, data
                        in nodes.items()
                    },
                }

                if snapshot != previous_snapshot:
                    previous_snapshot = snapshot

                    with _shared_lock:
                        _shared.update(
                            {
                                "fl_state": fl_state,
                                "round": round_number,
                                "max_rounds": max_rounds,
                                "nodes": nodes,
                            }
                        )

                    _broadcast(
                        {
                            "type": "round",
                            "fl_state": fl_state,
                            "round": round_number,
                            "max_rounds": max_rounds,
                            "nodes": nodes,
                        }
                    )
                    print(
                        "  [Poll] "
                        f"R{round_number}/{max_rounds} "
                        f"{fl_state} "
                        f"nodes={list(nodes.keys())}"
                    )

        except Exception as error:
            print(
                "  ⚠ oneM2M poll 오류: "
                f"{error}"
            )

        # 새로운 글로벌 모델이 있는지 확인
        _reload_scores_if_needed()

        # FL 완료 후 최종 모델까지 준비되면
        # summary를 한 번만 전송한다.
        with _shared_lock:
            fl_completed = (
                _shared["fl_state"]
                == "FL_COMPLETED"
            )
            summary_already_sent = (
                _shared["summary_sent"]
            )
            final_model_ready = (
                _shared["model_round"]
                >= _shared["max_rounds"]
            )

        if (
            fl_completed
            and final_model_ready
            and not summary_already_sent
        ):
            summary = _compute_summary()

            if summary:
                with _shared_lock:
                    _shared["summary_sent"] = True

                _broadcast(
                    summary,
                )
                print(
                    "  ✓ Summary 전송 완료 "
                    f"(R{summary['model_round']})"
                )
        time.sleep(
            POLL_INTERVAL,
        )


# ════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════

def main() -> None:
    GLOBAL_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=== FL Dashboard Server ===")
    print(f"  oneM2M: {config.BASE_URL}")
    print(f"  PKL: {PKL_DIR}")
    print(f"  Global model: {GLOBAL_MODEL_DIR}")
    print(f"  Run marker: {RUN_MARKER_PATH}")
    print(
        "  이상 표시: "
        f"{ANOMALY_NODE}, "
        f"Round {ANOMALY_START_ROUND}부터"
    )
    print(f"  Port: {PORT}")
    print(f"  http://localhost:{PORT}")
    print()

    polling_thread = threading.Thread(
        target=_poll_thread,
        daemon=True,
        name="fl-poll",
    )

    score_thread = threading.Thread(
        target=_score_thread,
        daemon=True,
        name="score-stream",
    )

    polling_thread.start()
    score_thread.start()

    app.run(
        host="0.0.0.0",
        port=PORT,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()