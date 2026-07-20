"""
fl/evaluate_mn1_hidden_test.py — mn1 hidden test stream 단독 평가

mn1.pkl 의 test_stream (정상→이상 순서, 학습 중 미공개) 에 대해:
  1. val 정상 MSE로 threshold 결정 (mean + N*sigma)
  2. test_stream 각 샘플의 재구성 오차 계산
  3. K회 연속 초과 시 탐지 → 탐지 시점 기록
  4. AUROC, 정밀도/재현율, score 분포 출력

Usage:
  python3 fl/evaluate_mn1_hidden_test.py \\
      --model /tmp/fl_models/global/global_round20.pt \\
      --hidden-test /tmp/fl_data/femto/mn1.pkl
  python3 fl/evaluate_mn1_hidden_test.py \\
      --model /tmp/fl_models/global/global_round20.pt \\
      --hidden-test /tmp/fl_data/femto/mn1.pkl \\
      --n-sigma 3.0 --k 3 --batch 32
"""
import sys
import os
import argparse
import pickle

import numpy as np
import torch

sys.path.append('/home/eunjin/federated-learning/fl')
from model import Conv1DAE
import config


# ════════════════════════════════════════════════════════
# 유틸
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

    반환:
      detection_idx : 탐지된 샘플 인덱스 (없으면 None)
      onset_idx     : 첫 이상 샘플 인덱스
      delay         : 탐지 지연 (탐지 인덱스 - 첫 이상 인덱스)
      false_alarms  : 정상 구간에서 잘못 탐지한 횟수
    """
    onset_idx = int(np.where(labels == 1)[0][0]) if (labels == 1).any() else None

    detection_idx = None
    consecutive = 0
    false_alarms = 0

    for i, (score, label) in enumerate(zip(scores, labels)):
        if score > threshold:
            consecutive += 1
            if consecutive >= k:
                detection_idx = i - k + 1  # 연속 시작 인덱스
                # 정상 구간 탐지면 false alarm
                if onset_idx is not None and detection_idx < onset_idx:
                    false_alarms += 1
                    consecutive = 0          # 리셋하고 계속 탐색
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
# Main
# ════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="mn1 hidden test stream 평가 (Conv1DAE)"
    )
    parser.add_argument("--model",       required=True,
                        help="global model .pt 파일 경로")
    parser.add_argument("--hidden-test", required=True,
                        help="mn1.pkl 경로 (test_stream_signals/labels 포함)")
    parser.add_argument("--n-sigma",     type=float, default=3.0,
                        help="threshold = val 정상 MSE mean + N*sigma (기본 3.0)")
    parser.add_argument("--k",           type=int,   default=3,
                        help="K회 연속 초과 시 탐지 (기본 3)")
    parser.add_argument("--batch",       type=int,   default=32,
                        help="배치 크기 (기본 32)")
    args = parser.parse_args()

    print("=" * 60)
    print("  mn1 Hidden Test Stream Evaluation (Conv1DAE)")
    print("=" * 60)
    print(f"  모델      : {args.model}")
    print(f"  hidden test: {args.hidden_test}")
    print(f"  임계값    : val 정상 MSE mean + {args.n_sigma}sigma")
    print(f"  탐지      : {args.k}회 연속 초과")

    # ── 모델 로드 ──
    if not os.path.exists(args.model):
        print(f"\n[ERROR] 모델 파일 없음: {args.model}")
        sys.exit(1)

    ae_cfg = config.AE_CFG
    model = load_model(args.model, ae_cfg)
    print(f"  model loaded OK  (seq_len={ae_cfg.seq_len}, latent_dim={ae_cfg.latent_dim})")

    # ── 데이터 로드 ──
    if not os.path.exists(args.hidden_test):
        print(f"\n[ERROR] pkl 파일 없음: {args.hidden_test}")
        sys.exit(1)

    with open(args.hidden_test, "rb") as f:
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
    n_val_anom    = int((val_labels == 1).sum())
    n_test_normal = int((test_labels == 0).sum())
    n_test_anom   = int((test_labels == 1).sum())

    print(f"\n  모터     : {motors}")
    print(f"  val      : 정상={n_val_normal}  이상={n_val_anom}")
    print(f"  test     : 정상={n_test_normal}  이상={n_test_anom}  (정상→이상 순서)")

    # ── Threshold 결정 (val 정상만) ──
    print(f"\n{'─'*60}")
    print("  [Threshold 결정]")
    val_scores_all    = compute_scores(model, val_sigs, args.batch)
    val_normal_scores = val_scores_all[val_labels == 0]
    val_anom_scores   = val_scores_all[val_labels == 1]

    threshold = compute_threshold(val_normal_scores, args.n_sigma)

    print(f"    val 정상 MSE : mean={val_normal_scores.mean():.5f}  "
          f"std={val_normal_scores.std():.5f}")
    if len(val_anom_scores) > 0:
        print(f"    val 이상 MSE : mean={val_anom_scores.mean():.5f}  "
              f"std={val_anom_scores.std():.5f}")
    print(f"    threshold    : {threshold:.5f}  (mean + {args.n_sigma}sigma)")

    # ── Test stream 스코어 계산 ──
    print(f"\n{'─'*60}")
    print("  [Test Stream 평가]")
    test_scores = compute_scores(model, test_sigs, args.batch)

    normal_scores = test_scores[test_labels == 0]
    anom_scores   = test_scores[test_labels == 1]
    print(f"    정상 구간 MSE : mean={normal_scores.mean():.5f}  "
          f"max={normal_scores.max():.5f}")
    if len(anom_scores) > 0:
        print(f"    이상 구간 MSE : mean={anom_scores.mean():.5f}  "
              f"max={anom_scores.max():.5f}")

    # ── AUROC ──
    auroc = compute_auroc(test_scores, test_labels)
    print(f"\n  [1] AUROC : {auroc:.4f}")

    # ── Precision / Recall / F1 at threshold ──
    prec, rec, f1 = compute_pr(test_scores, test_labels, threshold)
    print(f"  [2] Precision/Recall/F1 @ threshold={threshold:.5f}")
    print(f"      P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")

    # ── Online detection ──
    det = online_detection(test_scores, test_labels, threshold, args.k)
    print(f"\n  [3] Online Detection (K={args.k} 연속 초과)")
    if det["onset_idx"] is not None:
        print(f"      onset_idx      : {det['onset_idx']}  (첫 이상 샘플)")
    if det["detection_idx"] is not None:
        print(f"      detection_idx  : {det['detection_idx']}")
        print(f"      delay          : {det['delay']} 샘플")
    else:
        print(f"      탐지 못함 (threshold 초과 {args.k}회 연속 없음)")
    print(f"      false_alarms   : {det['false_alarms']}회")

    # ── Score 분포 (앞 10개 정상 / 앞 10개 이상) ──
    print(f"\n  [4] Score 분포")
    onset = det["onset_idx"] if det["onset_idx"] is not None else n_test_normal
    preview = min(10, n_test_normal)

    print(f"    정상 구간 (처음 {preview}개):")
    for i in range(preview):
        mark = "  <-- ALARM" if test_scores[i] > threshold else ""
        print(f"      [{i:4d}] score={test_scores[i]:.5f}{mark}")

    preview_anom = min(10, n_test_anom)
    print(f"    이상 구간 (처음 {preview_anom}개):")
    for j in range(preview_anom):
        i = onset + j
        if i < len(test_scores):
            mark = "  <-- ALARM" if test_scores[i] > threshold else ""
            print(f"      [{i:4d}] score={test_scores[i]:.5f}{mark}")

    print(f"\n{'='*60}")
    print("  완료")
    print(f"{'='*60}")

    return {
        "auroc":         auroc,
        "precision":     prec,
        "recall":        rec,
        "f1":            f1,
        "threshold":     threshold,
        "detection_idx": det["detection_idx"],
        "onset_idx":     det["onset_idx"],
        "delay":         det["delay"],
        "false_alarms":  det["false_alarms"],
    }


if __name__ == "__main__":
    main()
