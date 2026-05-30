"""
fl/evaluate_test_stream.py — FL 완료 후 test_stream 평가

각 노드의 test_stream (정상→이상 순서) 에 대해:
  1. val 정상 MSE로 threshold 결정 (mean + N*sigma)
  2. test_stream 각 샘플의 재구성 오차 계산
  3. K회 연속 초과 시 탐지 → 탐지 시점 기록
  4. AUROC, 정밀도/재현율 출력

Usage:
  python3 evaluate_test_stream.py
  python3 evaluate_test_stream.py --model /tmp/fl_models/global/global_round20.pt
  python3 evaluate_test_stream.py --n-sigma 3.0 --k 3
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
        n_channels = ae_cfg.n_channels,
        latent_dim = ae_cfg.latent_dim,
        seq_len    = ae_cfg.seq_len,
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
            xb   = X[i:i+batch_size].unsqueeze(1)   # (B, 1, L)
            recon = model(xb)
            err  = ((xb - recon) ** 2).mean(dim=(1, 2))
            scores.append(err.numpy())
    return np.concatenate(scores)


def compute_threshold(val_scores_normal: np.ndarray, n_sigma: float) -> float:
    return float(val_scores_normal.mean() + n_sigma * val_scores_normal.std())


def online_detection(scores: np.ndarray, labels: np.ndarray,
                     threshold: float, k: int) -> dict:
    """
    정상→이상 스트림에서 K회 연속 threshold 초과 시 탐지.

    반환:
      detection_idx   : 탐지된 샘플 인덱스 (없으면 None)
      onset_idx       : 첫 이상 샘플 인덱스
      delay           : 탐지 지연 (탐지 인덱스 - 첫 이상 인덱스)
      false_alarms    : 정상 구간에서 잘못 탐지한 횟수
    """
    onset_idx = int(np.where(labels == 1)[0][0]) if (labels == 1).any() else None

    detection_idx = None
    consecutive   = 0
    false_alarms  = 0

    for i, (score, label) in enumerate(zip(scores, labels)):
        if score > threshold:
            consecutive += 1
            if consecutive >= k:
                detection_idx = i - k + 1   # 연속 시작 인덱스
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
# 노드별 평가
# ════════════════════════════════════════════════════════

def evaluate_node(node: str, pkl_path: str, model: Conv1DAE,
                  n_sigma: float, k: int, batch_size: int):
    print(f"\n{'─'*55}")
    print(f"  NODE: {node}")
    print(f"{'─'*55}")

    if not os.path.exists(pkl_path):
        print(f"  ✗ pkl 없음: {pkl_path}")
        return None

    with open(pkl_path, "rb") as f:
        ds = pickle.load(f)

    val_sigs    = ds.get("val_signals")
    val_labels  = ds.get("val_labels")
    test_sigs   = ds.get("test_stream_signals")
    test_labels = ds.get("test_stream_labels")

    if val_sigs is None or test_sigs is None:
        print(f"  ✗ val/test 데이터 없음")
        return None

    n_val_normal = int((val_labels == 0).sum())
    n_val_anom   = int((val_labels == 1).sum())
    n_test_normal = int((test_labels == 0).sum())
    n_test_anom   = int((test_labels == 1).sum())
    motors = ds.get("motors", [])

    print(f"  모터: {motors}")
    print(f"  val    : 정상={n_val_normal}  이상={n_val_anom}")
    print(f"  test   : 정상={n_test_normal}  이상={n_test_anom}  (정상→이상 순서)")

    # ── Threshold 결정 (val 정상만) ──
    val_normal_sigs = val_sigs[val_labels == 0]
    val_scores_all  = compute_scores(model, val_sigs, batch_size)
    val_normal_scores = val_scores_all[val_labels == 0]
    val_anom_scores   = val_scores_all[val_labels == 1]

    threshold = compute_threshold(val_normal_scores, n_sigma)

    print(f"\n  [Threshold]")
    print(f"    val 정상 MSE : mean={val_normal_scores.mean():.5f}  std={val_normal_scores.std():.5f}")
    if len(val_anom_scores) > 0:
        print(f"    val 이상 MSE : mean={val_anom_scores.mean():.5f}  std={val_anom_scores.std():.5f}")
    print(f"    threshold    : {threshold:.5f}  (mean + {n_sigma}σ)")

    # ── Test stream ──
    test_scores = compute_scores(model, test_sigs, batch_size)

    normal_scores = test_scores[test_labels == 0]
    anom_scores   = test_scores[test_labels == 1]
    print(f"\n  [Test Stream MSE]")
    print(f"    정상 구간 : mean={normal_scores.mean():.5f}  max={normal_scores.max():.5f}")
    if len(anom_scores) > 0:
        print(f"    이상 구간 : mean={anom_scores.mean():.5f}  max={anom_scores.max():.5f}")

    # ── AUROC ──
    auroc = compute_auroc(test_scores, test_labels)
    print(f"\n  [AUROC] {auroc:.4f}")

    # ── Precision / Recall / F1 ──
    prec, rec, f1 = compute_pr(test_scores, test_labels, threshold)
    print(f"  [P/R/F1 @ threshold] P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")

    # ── Online 탐지 시점 ──
    det = online_detection(test_scores, test_labels, threshold, k)
    print(f"\n  [Online Detection (K={k}연속 초과)]")
    if det["onset_idx"] is not None:
        print(f"    첫 이상 샘플   : index={det['onset_idx']}")
    if det["detection_idx"] is not None:
        print(f"    탐지 시점      : index={det['detection_idx']}")
        print(f"    탐지 지연      : {det['delay']} 샘플")
    else:
        print(f"    탐지 못함 (threshold 초과 {k}회 연속 없음)")
    print(f"    False Alarm    : {det['false_alarms']}회")

    # ── Score 분포 요약 ──
    print(f"\n  [Score 분포] (앞 10개 / 뒤 10개)")
    onset = det["onset_idx"] or n_test_normal
    preview = min(10, n_test_normal)
    print(f"    정상 구간 (처음 {preview}개):")
    for i in range(preview):
        mark = " ▲ALARM" if test_scores[i] > threshold else ""
        print(f"      [{i:3d}] score={test_scores[i]:.5f}{mark}")
    print(f"    이상 구간 (처음 {preview}개):")
    for j in range(min(preview, n_test_anom)):
        i = onset + j
        if i < len(test_scores):
            mark = " ▲ALARM" if test_scores[i] > threshold else ""
            print(f"      [{i:3d}] score={test_scores[i]:.5f}{mark}")

    return {
        "node":           node,
        "auroc":          auroc,
        "precision":      prec,
        "recall":         rec,
        "f1":             f1,
        "threshold":      threshold,
        "detection_idx":  det["detection_idx"],
        "onset_idx":      det["onset_idx"],
        "delay":          det["delay"],
        "false_alarms":   det["false_alarms"],
    }


# ════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default=f"/tmp/fl_models/global/global_round{config.GLOBAL_ROUNDS}.pt")
    parser.add_argument("--n-sigma", type=float, default=3.0)
    parser.add_argument("--k",       type=int,   default=3)
    parser.add_argument("--batch",   type=int,   default=32)
    args = parser.parse_args()

    print("=" * 55)
    print("  FL Test Stream Evaluation (Conv1DAE)")
    print("=" * 55)
    print(f"  모델  : {args.model}")
    print(f"  임계값: val 정상 MSE mean + {args.n_sigma}σ")
    print(f"  탐지  : {args.k}회 연속 초과")

    if not os.path.exists(args.model):
        print(f"\n✗ 모델 파일 없음: {args.model}")
        sys.exit(1)

    ae_cfg = config.AE_CFG
    model  = load_model(args.model, ae_cfg)
    print(f"  ✓ 모델 로드 완료")

    results = []
    for node_idx, node in enumerate(["mn1", "mn2", "mn3"]):
        pkl_path = f"/tmp/fl_data/femto/{node}.pkl"
        r = evaluate_node(node, pkl_path, model,
                          n_sigma=args.n_sigma, k=args.k, batch_size=args.batch)
        if r:
            results.append(r)

    # ── 전체 요약 ──
    if results:
        print(f"\n{'═'*55}")
        print(f"  전체 요약")
        print(f"{'═'*55}")
        print(f"  {'Node':<6} {'AUROC':>6} {'F1':>6} {'Delay':>6} {'FA':>4}")
        print(f"  {'─'*36}")
        for r in results:
            delay = str(r['delay']) if r['delay'] is not None else "miss"
            print(f"  {r['node']:<6} {r['auroc']:>6.4f} {r['f1']:>6.3f}"
                  f" {delay:>6} {r['false_alarms']:>4}")
        avg_auroc = np.mean([r['auroc'] for r in results])
        avg_f1    = np.mean([r['f1']    for r in results])
        print(f"  {'─'*36}")
        print(f"  {'avg':<6} {avg_auroc:>6.4f} {avg_f1:>6.3f}")
        print()


if __name__ == "__main__":
    main()
