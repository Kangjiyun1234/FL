"""
fl/dashboard_server.py — FL 대시보드 백엔드 (SSE)

백그라운드 스레드 2개:
  fl-poll      : oneM2M 폴링 (3s) + 새 글로벌 모델 감지 → test_stream 스코어 재계산
  score-stream : test_stream 샘플별 재구성 오차 순환 스트리밍 (0.5s)

SSE 이벤트 종류:
  type=init      : 접속 직후 현재 상태 스냅샷
  type=round     : FL 상태 + 라운드별 노드 메트릭 변경 시
  type=threshold : 새 글로벌 모델로 threshold 재계산 완료 시
  type=score     : 샘플별 재구성 오차 (0.5s 주기)
  type=summary   : FL 완료 시 최종 평가 결과 (AUROC, 탐지 지연, 오경보 수)

Usage:
  python3 fl/dashboard_server.py
  → http://localhost:7000
"""
from __future__ import annotations

import os
import sys
import json
import time
import pickle
import threading
import queue
import glob
from pathlib import Path

import numpy as np
import torch
from flask import Flask, Response
from sklearn.metrics import roc_auc_score

import config
import onem2m_utils as om2m
from model import Conv1DAE

# ─── 파라미터 ──────────────────────────────────────────────────────────────
PORT             = int(os.getenv("FL_DASHBOARD_PORT", "7000"))
POLL_INTERVAL    = float(os.getenv("FL_DASHBOARD_POLL_INTERVAL", "3.0"))    # oneM2M 폴링 주기 (초)
SCORE_INTERVAL   = float(os.getenv("FL_DASHBOARD_SCORE_INTERVAL", "0.5"))   # 샘플 스트리밍 주기 (초)

N_SIGMA          = 3.0    # threshold = val 정상 MSE mean + N*sigma
K_CONSECUTIVE    = 3      # 연속 N회 threshold 초과 시 fault 판정

PROJECT_ROOT     = Path(__file__).resolve().parents[1]
HTML_PATH        = PROJECT_ROOT / "fl_bearing_dashboard.html"

GLOBAL_MODEL_DIR = Path(os.getenv("FL_GLOBAL_MODEL_DIR", "/tmp/fl_models/global"))
PKL_DIR          = Path(os.getenv("FL_PKL_DIR", "/tmp/fl_data/femto"))

NODES            = ["mn1", "mn2", "mn3"]

app = Flask(__name__)

# ─── SSE 구독자 ────────────────────────────────────────────────────────────
_subs: list[queue.Queue] = []
_subs_lock = threading.Lock()

# ─── 공유 상태 ─────────────────────────────────────────────────────────────
_shared: dict = {
    "fl_state":    "FL_READY",
    "round":       0,
    "max_rounds":  config.GLOBAL_ROUNDS,
    "nodes":       {},   # node → {train_loss, val_loss, val_auroc, num_samples, round}
    "thresholds":  {},   # node → float
    "scores":      {},   # node → np.ndarray  (test_stream 재구성 오차)
    "labels":      {},   # node → np.ndarray  (test_stream 라벨 0/1)
    "score_idx":    0,
    "model_round":  -1,
    "summary_sent": False,
}
_shared_lock = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════
# SSE 헬퍼
# ══════════════════════════════════════════════════════════════════════════

def _broadcast(evt: dict) -> None:
    """모든 SSE 구독자에게 이벤트 전송 (끊어진 구독자는 제거)"""
    msg = "data: " + json.dumps(evt, ensure_ascii=False) + "\n\n"
    with _subs_lock:
        dead = []
        for q in _subs:
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _subs.remove(q)


@app.route("/")
def index():
    if not HTML_PATH.exists():
        return (
            f"Dashboard HTML not found: {HTML_PATH}<br>"
            "프로젝트 루트에 fl_bearing_dashboard.html이 있는지 확인하세요.",
            500,
        )
    return HTML_PATH.read_text(encoding="utf-8")


@app.route("/health")
def health():
    global_files = []
    try:
        global_files = sorted(
            [p.name for p in GLOBAL_MODEL_DIR.glob("global_round*.pt")]
        )
    except Exception:
        global_files = []

    pkl_status = {
        node: (PKL_DIR / f"{node}.pkl").exists()
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

            "html_path": str(HTML_PATH),
            "html_exists": HTML_PATH.exists(),

            "global_model_dir": str(GLOBAL_MODEL_DIR),
            "global_model_dir_exists": GLOBAL_MODEL_DIR.exists(),
            "global_model_files": global_files,

            "pkl_dir": str(PKL_DIR),
            "pkl_dir_exists": PKL_DIR.exists(),
            "pkl_files": pkl_status,

            "nodes": _shared["nodes"],
            "thresholds": _shared["thresholds"],
            "score_idx": _shared["score_idx"],
            "summary_sent": _shared["summary_sent"],
        }

    return state


@app.route("/stream")
def stream():
    q: queue.Queue = queue.Queue(maxsize=150)
    with _subs_lock:
        _subs.append(q)

    # 접속 직후: 현재 상태 스냅샷 즉시 전송
    with _shared_lock:
        snap = {
            "type":       "init",
            "fl_state":   _shared["fl_state"],
            "round":      _shared["round"],
            "max_rounds": _shared["max_rounds"],
            "nodes":      _shared["nodes"],
            "thresholds": _shared["thresholds"],
            "model_round": _shared["model_round"],
        }
    try:
        q.put_nowait("data: " + json.dumps(snap, ensure_ascii=False) + "\n\n")
    except queue.Full:
        pass

    def generate():
        try:
            while True:
                try:
                    yield q.get(timeout=25)
                except queue.Empty:
                    yield ": heartbeat\n\n"   # keep-alive
        except GeneratorExit:
            pass
        finally:
            with _subs_lock:
                if q in _subs:
                    _subs.remove(q)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ══════════════════════════════════════════════════════════════════════════
# 모델 / 스코어 계산
# ══════════════════════════════════════════════════════════════════════════

def _find_latest_model() -> tuple[int, Path | None]:
    """가장 최신 global_roundN.pt 파일의 (round, path) 반환"""
    if not GLOBAL_MODEL_DIR.exists():
        return -1, None

    best_r, best_f = -1, None
    for f in GLOBAL_MODEL_DIR.glob("global_round*.pt"):
        try:
            r = int(f.stem.replace("global_round", ""))
            if r > best_r:
                best_r, best_f = r, f
        except ValueError:
            pass
    return best_r, best_f


def _load_model(path: str | Path) -> Conv1DAE:
    ae = config.AE_CFG
    m = Conv1DAE(n_channels=ae.n_channels, latent_dim=ae.latent_dim, seq_len=ae.seq_len)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m


def _compute_scores(model: Conv1DAE, sigs: np.ndarray, batch: int = 64) -> np.ndarray:
    """signals (N, seq_len) → MSE scores (N,)"""
    if len(sigs) == 0:
        return np.array([], dtype=np.float32)
    out = []
    X = torch.tensor(sigs, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = X[i:i + batch].unsqueeze(1)
            err = ((xb - model(xb)) ** 2).mean(dim=(1, 2))
            out.append(err.numpy())
    return np.concatenate(out)


def _compute_threshold(model: Conv1DAE, val_sigs: np.ndarray, val_labels: np.ndarray) -> float:
    normal = val_sigs[val_labels == 0]
    if len(normal) == 0:
        return 1.0
    sc = _compute_scores(model, normal)
    return float(sc.mean() + N_SIGMA * sc.std())


def _reload_scores_if_needed() -> None:
    """새 글로벌 모델 발견 시 모든 노드 test_stream 스코어 + threshold 재계산"""
    model_r, model_path = _find_latest_model()
    with _shared_lock:
        cur      = _shared["model_round"]
        fl_round = _shared["round"]

    # Stale model guard: 디스크에 이전 FL 실행의 모델이 남아있으면 무시.
    # 현재 FL 라운드보다 2 이상 앞선 모델은 이전 실행 잔재로 판단.
    if model_r > fl_round + 1:
        if cur > fl_round:          # 이미 stale 모델로 스코어링됐다면 리셋
            with _shared_lock:
                _shared["model_round"] = -1
        return

    if model_r <= cur or not model_path:
        return

    print(f"  [Scores] 새 모델 R{model_r} 감지 → 스코어 재계산...")
    try:
        model = _load_model(model_path)
    except Exception as e:
        print(f"  ✗ 모델 로드 실패: {e}")
        return

    new_thr: dict[str, float]      = {}
    new_scores: dict[str, np.ndarray] = {}
    new_labels: dict[str, np.ndarray] = {}

    for node in NODES:
        pkl = PKL_DIR / f"{node}.pkl"
        if not pkl.exists():
            print(f"  ⚠ {pkl} 없음, 스킵")
            continue
        with pkl.open("rb") as f:
            ds = pickle.load(f)

        seq_len   = config.AE_CFG.seq_len
        empty_sig = np.empty((0, seq_len), dtype=np.float32)
        empty_lbl = np.empty((0,), dtype=np.int64)

        val_sigs   = ds.get("val_signals",          empty_sig)
        val_labels = ds.get("val_labels",            empty_lbl)
        test_sigs  = ds.get("test_stream_signals",   empty_sig)
        test_labels= ds.get("test_stream_labels",    empty_lbl)

        thr = _compute_threshold(model, val_sigs, val_labels)
        sc  = _compute_scores(model, test_sigs)

        new_thr[node]    = round(thr, 4)
        new_scores[node] = sc
        new_labels[node] = test_labels
        print(f"    {node}: n={len(sc)}  threshold={thr:.4f}")

    with _shared_lock:
        _shared["thresholds"]   = new_thr
        _shared["scores"]       = new_scores
        _shared["labels"]       = new_labels
        _shared["score_idx"]    = 0
        _shared["model_round"]  = model_r
        _shared["summary_sent"] = False

    _broadcast({
        "type":        "threshold",
        "model_round": model_r,
        "thresholds":  new_thr,
    })
    print(f"  ✓ 재계산 완료 (R{model_r})")


def _compute_summary() -> dict | None:
    """FL 완료 후 노드별 최종 평가 결과 계산."""
    with _shared_lock:
        scores     = {k: v.copy() for k, v in _shared["scores"].items()}
        labels     = {k: v.copy() for k, v in _shared["labels"].items()}
        thrs       = dict(_shared["thresholds"])
        model_r    = _shared["model_round"]
        max_rounds = _shared["max_rounds"]

    if model_r < max_rounds or not scores:
        return None

    result: dict = {"type": "summary", "model_round": model_r, "nodes": {}}

    for node in NODES:
        sc  = scores.get(node)
        lb  = labels.get(node)
        thr = thrs.get(node, 1.0)
        if sc is None or lb is None or len(sc) == 0:
            continue

        try:
            auroc = float(roc_auc_score(lb, sc)) if len(np.unique(lb)) > 1 else 0.0
        except Exception:
            auroc = 0.0

        onset_idx = next((int(i) for i, l in enumerate(lb) if l == 1), None)

        detect_idx, consec = None, 0
        for i, s in enumerate(sc):
            if s > thr:
                consec += 1
                if consec >= K_CONSECUTIVE:
                    detect_idx = int(i - K_CONSECUTIVE + 1)
                    break
            else:
                consec = 0

        delay = int(detect_idx - onset_idx) if (detect_idx is not None and onset_idx is not None) else None

        # False alarms: 정상 구간에서 K회 연속 threshold 초과 횟수 (evaluate 스크립트 기준)
        false_alarms, fa_consec = 0, 0
        for s, l in zip(sc, lb):
            if l == 0:
                if s > thr:
                    fa_consec += 1
                    if fa_consec >= K_CONSECUTIVE:
                        false_alarms += 1
                        fa_consec = 0
                else:
                    fa_consec = 0

        result["nodes"][node] = {
            "auroc":        round(auroc, 4),
            "threshold":    round(float(thr), 4),
            "onset_idx":    onset_idx,
            "detect_idx":   detect_idx,
            "delay":        delay,
            "false_alarms": false_alarms,
            "n_normal":     int(np.sum(lb == 0)),
            "n_anomaly":    int(np.sum(lb == 1)),
        }

    return result


# ══════════════════════════════════════════════════════════════════════════
# 스코어 스트리밍 스레드
# ══════════════════════════════════════════════════════════════════════════

def _score_thread() -> None:
    """test_stream 샘플을 순서대로 브로드캐스트.
    끝에 도달하거나 FL_COMPLETED 상태가 되면 멈춤.
    새 글로벌 모델 로드 시에만 idx가 0으로 리셋됨.
    """
    while True:
        time.sleep(SCORE_INTERVAL)

        with _shared_lock:
            scores   = _shared["scores"]
            labels   = _shared["labels"]
            thrs     = _shared["thresholds"]
            idx      = _shared["score_idx"]
            fl_state = _shared["fl_state"]

        if not scores:
            continue

        # 모든 노드의 test_stream 최대 길이
        max_len = max((len(v) for v in scores.values()), default=0)
        if idx >= max_len:
            # 스트림 종료 — 새 모델이 로드되면 _reload_scores_if_needed()가 idx=0으로 리셋
            continue

        evt: dict = {"type": "score"}
        for node in NODES:
            if node not in scores or idx >= len(scores[node]):
                continue
            evt[node]            = round(float(scores[node][idx]), 5)
            evt[f"{node}_label"] = int(labels[node][idx]) if node in labels else 0
            evt[f"{node}_thr"]   = thrs.get(node, 1.0)

        with _shared_lock:
            _shared["score_idx"] = idx + 1

        if len(evt) > 1:
            _broadcast(evt)


# ══════════════════════════════════════════════════════════════════════════
# FL 폴링 스레드
# ══════════════════════════════════════════════════════════════════════════

def _parse_con(cin: dict | None) -> dict | None:
    if not cin or "m2m:cin" not in cin:
        return None
    con = cin["m2m:cin"]["con"]
    try:
        return json.loads(con) if isinstance(con, str) else con
    except Exception:
        return None


def _poll_thread() -> None:
    fl_ctrl = f"{config.CSE_NAME}/{config.IN_AE_NAME}/cnt-fl-control"
    db_root = f"{config.CSE_NAME}/{config.IN_AE_NAME}/cnt-local-updates"
    prev_snap: dict = {}

    while True:
        try:
            ctrl = _parse_con(om2m.get_latest_content_instance(fl_ctrl))
            if ctrl:
                state   = ctrl.get("jobState", "FL_READY")
                round_n = int(ctrl.get("currentRound", 0))
                max_r   = int(ctrl.get("maxRounds", config.GLOBAL_ROUNDS))

                # 노드별 최신 메트릭 (dropbox 최신 CIN)
                nodes: dict = {}
                for node in NODES:
                    data = _parse_con(
                        om2m.get_latest_content_instance(f"{db_root}/cnt-{node}")
                    )
                    if data:
                        nodes[node] = {
                            "train_loss":  round(float(data.get("train_loss",  0)), 5),
                            "val_loss":    round(float(data.get("val_loss",    0)), 5),
                            "val_auroc":   round(float(data.get("val_auroc",   0)), 4),
                            "num_samples": int(data.get("num_samples", 0)),
                            "round":       int(data.get("round", 0)),
                        }

                snap = {
                    "state":  state,
                    "round":  round_n,
                    "n_keys": sorted(nodes.keys()),
                    # 노드 메트릭이 바뀌었는지도 비교
                    "losses": {n: d["train_loss"] for n, d in nodes.items()},
                }
                if snap != prev_snap:
                    prev_snap = snap
                    with _shared_lock:
                        _shared.update({
                            "fl_state":   state,
                            "round":      round_n,
                            "max_rounds": max_r,
                            "nodes":      nodes,
                        })
                    _broadcast({
                        "type":       "round",
                        "fl_state":   state,
                        "round":      round_n,
                        "max_rounds": max_r,
                        "nodes":      nodes,
                    })
                    print(f"  [Poll] R{round_n}/{max_r} {state}  "
                          f"nodes={list(nodes.keys())}")

        except Exception as e:
            print(f"  ⚠ poll error: {e}")

        _reload_scores_if_needed()

        # FL 완료 + score 스트리밍 종료 후 summary 한 번만 브로드캐스트
        with _shared_lock:
            _fl_done    = _shared["fl_state"] == "FL_COMPLETED"
            _sent       = _shared["summary_sent"]
            _mready     = _shared["model_round"] >= _shared["max_rounds"]
            _score_idx  = _shared["score_idx"]
            _scores     = _shared["scores"]

        _max_len = max((len(v) for v in _scores.values()), default=0)
        _stream_done = _score_idx >= _max_len > 0

        if _fl_done and _mready and _stream_done and not _sent:
            summary = _compute_summary()
            if summary:
                with _shared_lock:
                    _shared["summary_sent"] = True
                _broadcast(summary)
                print(f"  ✓ Summary broadcast (R{summary['model_round']})")

        time.sleep(POLL_INTERVAL)


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    GLOBAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== FL Dashboard Server ===")
    print(f"  oneM2M : {config.BASE_URL}")
    print(f"  PKL    : {PKL_DIR}")
    print(f"  모델   : {GLOBAL_MODEL_DIR}")
    print(f"  포트   : {PORT}")
    print(f"  → http://localhost:{PORT}\n")

    threading.Thread(target=_poll_thread,  daemon=True, name="fl-poll").start()
    threading.Thread(target=_score_thread, daemon=True, name="score-stream").start()

    app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False)
