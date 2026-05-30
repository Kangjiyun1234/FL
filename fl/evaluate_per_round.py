"""
fl/evaluate_per_round.py — FL 라운드별 global model 탐지 성능 추이

교수님 피드백 반영: "정상 데이터로 먼저 학습한 뒤, FL이 언제 anomaly를 탐지하는가"
  → /tmp/fl_models/global/global_round{N}.pt 파일을 라운드 순서로 스캔
  → 각 라운드 모델로 mn1.pkl test_stream 평가 (정상→이상 순서)
  → 라운드별 AUROC, F1, detection_idx, onset_idx, delay, false_alarms 출력

Usage:
  python3 fl/evaluate_per_round.py
  python3 fl/evaluate_per_round.py \\
      --data /tmp/fl_data/femto/mn1.pkl \\
      --models-dir /tmp/fl_models/global \\
      --n-sigma 3.0 --k 3
"""
import sys
import os
import re
import argparse
import pickle

import numpy as np
import torch

sys.path.append('/home/eunjin/federated-learning/fl')
from model import Conv1DAE
import config


# ════════════════════════════════════════════════════════
# 유틸 (evaluate_test_stream.py 와 동일 로직)
# ════════════════════════════════════════════════════════

def load_model(model_path: str, ae_cfg) -> Conv1DAE:
    model = Conv1DAE(
        n_channels=ae_cfg.n_channels,
        latent_dim=ae_cfg.latent_dim,
        seq_len=ae_cfg.seq_len,
    )
    sd = torch.load(model_path, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    return model


def compute_scores(model: Conv1DAE, signals: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """signals: (N, seq_len) → anomaly scores (N,)"""
    scores = []
    X = torch.tensor(signals, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size].unsqueeze(1)  # (B, 1, L)
            recon = model(xb)
            err = ((xb - recon) ** 2).mean(dim=(1, 2))
            scores.append(err.numpy())
    return np.concatenate(scores)


def compute_threshold(val_scores_normal: np.ndarray, n_sigma: float) -> float:
    return float(val_scores_normal.mean() + n_sigma * val_scores_normal.std())


def online_detection(scores: np.ndarray, labels: np.ndarray,
                     threshold: float, k: int) -> dict:
    """
    정상→이상 스트림에서 K회 연속 threshold 초과 시 탐지.
    """
    onset_idx = int(np.where(labels == 1)[0][0]) if (labels == 1).any() else None

    detection_idx = None
    consecutive = 0
    false_alarms = 0

    for i, (score, label) in enumerate(zip(scores, labels)):
        if score > threshold:
            consecutive += 1
            if consecutive >= k:
                detection_idx = i - k + 1
                if onset_idx is not None and detection_idx < onset_idx:
                    false_alarms += 1
                    consecutive = 0
                    detection_idx = None
                else:
                    break
        else:
            consecutive = 0

    delay = None
    if detection_idx is not None and onset_idx is not None:
        delay = max(0, detection_idx - onset_idx)

    return {
        "detection_idx": detection_idx,
        "onset_idx":     onset_idx,
        "delay":         delay,
        "false_alarms":  false_alarms,
    }


def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")


def compute_pr(scores: np.ndarray, labels: np.ndarray, threshold: float):
    preds = (scores > threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


# ════════════════════════════════════════════════════════
# 라운드 파일 탐색
# ════════════════════════════════════════════════════════

def find_round_models(models_dir: str) -> list:
    """
    models_dir 안의 global_round{N}.pt 파일을 찾아 (round_num, path) 목록으로 반환.
    라운드 번호 순 정렬.
    """
    pattern = re.compile(r"global_round(\d+)\.pt$")
    results = []
    if not os.path.isdir(models_dir):
        return results
    for fname in os.listdir(models_dir):
        m = pattern.match(fname)
        if m:
            round_num = int(m.group(1))
            results.append((round_num, os.path.join(models_dir, fname)))
    results.sort(key=lambda x: x[0])
    return results


# ════════════════════════════════════════════════════════
# 단일 라운드 평가
# ════════════════════════════════════════════════════════

def evaluate_round(round_num: int, model_path: str,
                   val_sigs: np.ndarray, val_labels: np.ndarray,
                   test_sigs: np.ndarray, test_labels: np.ndarray,
                   ae_cfg, n_sigma: float, k: int, batch_size: int) -> dict:
    """한 라운드 모델을 로드해 test_stream 평가 결과를 반환."""
    model = load_model(model_path, ae_cfg)

    # threshold: val 정상 MSE
    val_scores      = compute_scores(model, val_sigs, batch_size)
    val_norm_scores = val_scores[val_labels == 0]
    threshold       = compute_threshold(val_norm_scores, n_sigma)

    # test stream 스코어
    test_scores = compute_scores(model, test_sigs, batch_size)

    auroc            = compute_auroc(test_scores, test_labels)
    prec, rec, f1    = compute_pr(test_scores, test_labels, threshold)
    det              = online_detection(test_scores, test_labels, threshold, k)

    return {
        "round":         round_num,
        "auroc":         auroc,
        "f1":            f1,
        "precision":     prec,
        "recall":        rec,
        "threshold":     threshold,
        "detection_idx": det["detection_idx"],
        "onset_idx":     det["onset_idx"],
        "delay":         det["delay"],
        "false_alarms":  det["false_alarms"],
    }


# ════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="FL 라운드별 global model anomaly 탐지 성능 추이"
    )
    parser.add_argument("--data",       default="/tmp/fl_data/femto/mn1.pkl",
                        help="평가 대상 pkl 파일 (기본: mn1.pkl)")
    parser.add_argument("--models-dir", default="/tmp/fl_models/global",
                        help="global_round*.pt 파일이 있는 디렉터리")
    parser.add_argument("--n-sigma",    type=float, default=3.0,
                        help="threshold = val 정상 MSE mean + N*sigma (기본 3.0)")
    parser.add_argument("--k",          type=int,   default=3,
                        help="K회 연속 초과 시 탐지 (기본 3)")
    parser.add_argument("--batch",      type=int,   default=32,
                        help="배치 크기 (기본 32)")
    args = parser.parse_args()

    print("=" * 70)
    print("  FL Round-by-Round Anomaly Detection Evaluation")
    print("=" * 70)
    print(f"  데이터      : {args.data}")
    print(f"  모델 디렉터 : {args.models_dir}")
    print(f"  임계값      : val 정상 MSE mean + {args.n_sigma}sigma")
    print(f"  탐지        : {args.k}회 연속 초과")

    # ── 데이터 로드 ──
    if not os.path.exists(args.data):
        print(f"\n[ERROR] pkl 파일 없음: {args.data}")
        sys.exit(1)

    with open(args.data, "rb") as f:
        ds = pickle.load(f)

    val_sigs    = ds.get("val_signals")
    val_labels  = ds.get("val_labels")
    test_sigs   = ds.get("test_stream_signals")
    test_labels = ds.get("test_stream_labels")

    if val_sigs is None or test_sigs is None:
        print("[ERROR] pkl에 val_signals 또는 test_stream_signals 없음")
        sys.exit(1)

    motors = ds.get("motors", [])
    n_val_normal  = int((val_labels == 0).sum())
    n_test_normal = int((test_labels == 0).sum())
    n_test_anom   = int((test_labels == 1).sum())

    print(f"\n  모터    : {motors}")
    print(f"  val     : 정상={n_val_normal}")
    print(f"  test    : 정상={n_test_normal}  이상={n_test_anom}  (정상→이상 순서)")

    # ── 라운드 파일 탐색 ──
    round_list = find_round_models(args.models_dir)
    if not round_list:
        print(f"\n[ERROR] {args.models_dir} 에서 global_round*.pt 파일을 찾을 수 없음")
        sys.exit(1)

    print(f"\n  발견된 라운드 : {[r for r, _ in round_list]}")

    ae_cfg = config.AE_CFG
    results = []

    print(f"\n  평가 중...", flush=True)
    for round_num, model_path in round_list:
        try:
            r = evaluate_round(
                round_num, model_path,
                val_sigs, val_labels,
                test_sigs, test_labels,
                ae_cfg,
                n_sigma=args.n_sigma,
                k=args.k,
                batch_size=args.batch,
            )
            results.append(r)
            # 진행 표시
            det_str = str(r["detection_idx"]) if r["detection_idx"] is not None else "miss"
            print(f"    Round {round_num:3d}  AUROC={r['auroc']:.4f}  F1={r['f1']:.3f}"
                  f"  det={det_str}", flush=True)
        except Exception as e:
            print(f"    Round {round_num:3d}  [ERROR] {e}", flush=True)

    if not results:
        print("\n[ERROR] 평가 결과 없음")
        sys.exit(1)

    # ── 테이블 출력 ──
    onset_idx = results[0]["onset_idx"]  # 데이터마다 고정

    header = (f"\n{'Round':>6}  {'AUROC':>6}  {'F1':>6}  "
              f"{'detection_idx':>14}  {'onset_idx':>10}  {'delay':>7}  {'false_alarms':>12}")
    sep    = "─" * 75

    print(f"\n{'='*75}")
    print("  Round-by-Round Detection Table")
    print(f"{'='*75}")
    print(header)
    print(f"  {sep}")

    first_detect_round = None
    for r in results:
        det_str   = str(r["detection_idx"]) if r["detection_idx"] is not None else "miss"
        onset_str = str(r["onset_idx"])     if r["onset_idx"]     is not None else "N/A"
        delay_str = str(r["delay"])         if r["delay"]         is not None else "N/A"

        print(f"  {r['round']:>6}  {r['auroc']:>6.4f}  {r['f1']:>6.3f}  "
              f"{det_str:>14}  {onset_str:>10}  {delay_str:>7}  {r['false_alarms']:>12}")

        if first_detect_round is None and r["detection_idx"] is not None:
            first_detect_round = r["round"]

    print(f"  {sep}")

    # ── 요약 ──
    print(f"\n  [요약]")
    if onset_idx is not None:
        print(f"    첫 이상 샘플 위치 (onset_idx) : {onset_idx}")
    if first_detect_round is not None:
        r_first = next(r for r in results if r["round"] == first_detect_round)
        print(f"    최초 탐지 라운드              : Round {first_detect_round}"
              f"  (detection_idx={r_first['detection_idx']}, delay={r_first['delay']})")
    else:
        print(f"    최초 탐지 라운드              : 없음 (전 라운드 miss)")

    best_auroc = max(results, key=lambda r: r["auroc"] if not np.isnan(r["auroc"]) else -1)
    best_f1    = max(results, key=lambda r: r["f1"])
    print(f"    최고 AUROC                    : {best_auroc['auroc']:.4f}  "
          f"(Round {best_auroc['round']})")
    print(f"    최고 F1                       : {best_f1['f1']:.3f}  "
          f"(Round {best_f1['round']})")

    print(f"\n{'='*75}")
    print("  완료")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
