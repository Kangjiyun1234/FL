"""
fl/personalize.py — Personalized FL: 글로벌 모델 → 노드별 로컬 fine-tuning

FL 완료 후 각 노드가 자신의 전체 train 데이터로 글로벌 모델을 fine-tuning.
fine-tuning된 개인화 모델로 test_stream 탐지 성능을 평가하고
글로벌 모델 대비 개선 여부를 비교한다.

Usage:
  python3 personalize.py
  python3 personalize.py --global-model /tmp/fl_models/global/global_round10.pt
  python3 personalize.py --epochs 20 --lr 1e-4
"""
import sys
sys.path.append('/home/eunjin/federated-learning/fl')

import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from model import Conv1DAE

try:
    from sklearn.metrics import roc_auc_score
    _has_sklearn = True
except ImportError:
    _has_sklearn = False


# ════════════════════════════════════════════════════════
# 유틸
# ════════════════════════════════════════════════════════

def load_model(model_path: str) -> Conv1DAE:
    ae = Conv1DAE(
        n_channels=config.AE_CFG.n_channels,
        latent_dim=config.AE_CFG.latent_dim,
        seq_len=config.AE_CFG.seq_len,
    )
    sd = torch.load(model_path, map_location="cpu")
    ae.load_state_dict(sd)
    return ae


def compute_scores(model: Conv1DAE, signals: np.ndarray, batch_size: int = 32) -> np.ndarray:
    model.eval()
    X = torch.tensor(signals, dtype=torch.float32)
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size].unsqueeze(1)
            recon = model(xb)
            err = ((xb - recon) ** 2).mean(dim=(1, 2))
            scores.append(err.numpy())
    return np.concatenate(scores)


def compute_threshold(val_normal_scores: np.ndarray, n_sigma: float) -> float:
    return float(val_normal_scores.mean() + n_sigma * val_normal_scores.std())


def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    if not _has_sklearn or len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")


def online_detection(scores: np.ndarray, labels: np.ndarray,
                     threshold: float, k: int) -> dict:
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


def compute_f1(scores: np.ndarray, labels: np.ndarray, threshold: float) -> tuple:
    preds = (scores > threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


# ════════════════════════════════════════════════════════
# 평가 (글로벌 or 개인화 모델 공통)
# ════════════════════════════════════════════════════════

def evaluate(model: Conv1DAE, ds: dict, n_sigma: float, k: int, label: str) -> dict:
    val_sigs    = ds["val_signals"]
    val_labels  = ds["val_labels"]
    test_sigs   = ds["test_stream_signals"]
    test_labels = ds["test_stream_labels"]

    val_scores     = compute_scores(model, val_sigs)
    val_normal_sc  = val_scores[val_labels == 0]
    threshold      = compute_threshold(val_normal_sc, n_sigma)
    test_scores    = compute_scores(model, test_sigs)

    auroc = compute_auroc(test_scores, test_labels)
    p, r, f1 = compute_f1(test_scores, test_labels, threshold)
    det = online_detection(test_scores, test_labels, threshold, k)

    print(f"    [{label}]")
    print(f"      threshold : {threshold:.5f}  (val 정상 mean={val_normal_sc.mean():.4f} std={val_normal_sc.std():.4f})")
    print(f"      AUROC     : {auroc:.4f}")
    print(f"      P/R/F1    : {p:.3f} / {r:.3f} / {f1:.3f}")
    if det['detection_idx'] is not None:
        print(f"      탐지      : index={det['detection_idx']}  delay={det['delay']}  FA={det['false_alarms']}")
    else:
        print(f"      탐지      : miss  FA={det['false_alarms']}")

    return {
        "auroc": auroc, "precision": p, "recall": r, "f1": f1,
        "threshold": threshold,
        "detection_idx": det["detection_idx"],
        "onset_idx":     det["onset_idx"],
        "delay":         det["delay"],
        "false_alarms":  det["false_alarms"],
    }


# ════════════════════════════════════════════════════════
# 로컬 fine-tuning
# ════════════════════════════════════════════════════════

def fine_tune(model: Conv1DAE, train_signals: np.ndarray,
              epochs: int, lr: float, batch_size: int) -> float:
    """
    정상 train_signals 전체로 fine-tuning (DP 없음, 추론 전용).
    반환: 마지막 epoch 평균 loss
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    X = torch.tensor(train_signals, dtype=torch.float32)
    ds = TensorDataset(X)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    last_loss = 0.0
    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        n = 0
        for (xb,) in loader:
            xb_in = xb.unsqueeze(1)
            optimizer.zero_grad()
            recon = model(xb_in)
            loss  = criterion(recon, xb_in)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(xb)
            n += len(xb)
        last_loss = ep_loss / max(n, 1)
        if ep % 5 == 0 or ep == epochs:
            print(f"      epoch {ep:3d}/{epochs}  loss={last_loss:.6f}")

    return last_loss


# ════════════════════════════════════════════════════════
# 노드별 처리
# ════════════════════════════════════════════════════════

def process_node(node: str, pkl_path: str, global_model_path: str,
                 epochs: int, lr: float, batch_size: int,
                 n_sigma: float, k: int, out_dir: str) -> dict | None:

    print(f"\n{'═'*60}")
    print(f"  NODE: {node}")
    print(f"{'═'*60}")

    if not os.path.exists(pkl_path):
        print(f"  ✗ pkl 없음: {pkl_path}")
        return None

    with open(pkl_path, "rb") as f:
        ds = pickle.load(f)

    train_sigs = ds.get("train_signals")
    if train_sigs is None or len(train_sigs) == 0:
        print(f"  ✗ train_signals 없음")
        return None

    print(f"  모터        : {ds.get('motors', [])}")
    print(f"  train 샘플  : {len(train_sigs)}개  (FL 전 라운드 누적)")
    print(f"  val         : 정상={int((ds['val_labels']==0).sum())}  이상={int((ds['val_labels']==1).sum())}")
    print(f"  test_stream : 정상={int((ds['test_stream_labels']==0).sum())}  이상={int((ds['test_stream_labels']==1).sum())}")

    # ── 글로벌 모델 평가 ──
    print(f"\n  [1] 글로벌 모델 평가")
    global_model = load_model(global_model_path)
    result_global = evaluate(global_model, ds, n_sigma, k, "Global")

    # ── Fine-tuning ──
    print(f"\n  [2] 로컬 fine-tuning  (epochs={epochs}, lr={lr})")
    personal_model = load_model(global_model_path)   # 글로벌 모델에서 시작
    fine_tune(personal_model, train_sigs, epochs, lr, batch_size)

    # 개인화 모델 저장
    os.makedirs(out_dir, exist_ok=True)
    personal_path = os.path.join(out_dir, f"personal_{node}.pt")
    torch.save(personal_model.state_dict(), personal_path)
    print(f"      ✓ 저장: {personal_path}")

    # ── 개인화 모델 평가 ──
    print(f"\n  [3] 개인화 모델 평가")
    result_personal = evaluate(personal_model, ds, n_sigma, k, "Personal")

    # ── 비교 ──
    def _delta(key):
        g = result_global[key]
        p = result_personal[key]
        if g is None or p is None:
            return "N/A"
        return f"{p - g:+.4f}"

    print(f"\n  [비교]  Global → Personal")
    print(f"    AUROC : {result_global['auroc']:.4f} → {result_personal['auroc']:.4f}  ({_delta('auroc')})")
    print(f"    F1    : {result_global['f1']:.3f}  → {result_personal['f1']:.3f}  ({_delta('f1')})")

    g_delay = str(result_global['delay'])  if result_global['delay']  is not None else "miss"
    p_delay = str(result_personal['delay']) if result_personal['delay'] is not None else "miss"
    print(f"    delay : {g_delay} → {p_delay} 샘플")

    return {
        "node":     node,
        "global":   result_global,
        "personal": result_personal,
    }


# ════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Personalized FL: 글로벌 모델 로컬 fine-tuning")
    parser.add_argument("--global-model", default=f"/tmp/fl_models/global/global_round{config.GLOBAL_ROUNDS}.pt")
    parser.add_argument("--data-dir",     default="/tmp/fl_data/femto")
    parser.add_argument("--out-dir",      default="/tmp/fl_models/personal")
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--batch",        type=int,   default=32)
    parser.add_argument("--n-sigma",      type=float, default=3.0)
    parser.add_argument("--k",            type=int,   default=3)
    args = parser.parse_args()

    print("=" * 60)
    print("  Personalized FL — 글로벌 모델 로컬 fine-tuning")
    print("=" * 60)
    print(f"  글로벌 모델 : {args.global_model}")
    print(f"  fine-tuning : epochs={args.epochs}, lr={args.lr}")
    print(f"  임계값      : val 정상 MSE mean + {args.n_sigma}σ")
    print(f"  탐지        : {args.k}회 연속 초과")

    if not os.path.exists(args.global_model):
        print(f"\n✗ 글로벌 모델 없음: {args.global_model}")
        sys.exit(1)

    nodes = ["mn1", "mn2", "mn3"]
    results = []

    for node in nodes:
        pkl_path = os.path.join(args.data_dir, f"{node}.pkl")
        r = process_node(
            node=node, pkl_path=pkl_path,
            global_model_path=args.global_model,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch,
            n_sigma=args.n_sigma, k=args.k, out_dir=args.out_dir,
        )
        if r:
            results.append(r)

    # ════════════════════════════════════════
    # 전체 요약 비교
    # ════════════════════════════════════════
    if not results:
        return

    print(f"\n{'═'*65}")
    print(f"  전체 비교 요약  (Global vs Personalized)")
    print(f"{'═'*65}")
    print(f"  {'Node':<6}  {'AUROC_G':>8} {'AUROC_P':>8} {'Δ':>7}  "
          f"{'F1_G':>6} {'F1_P':>6}  {'delay_G':>8} {'delay_P':>8}")
    print(f"  {'─'*60}")

    for r in results:
        g = r["global"]
        p = r["personal"]
        da = g["auroc"];  pa = p["auroc"]
        df = g["f1"];     pf = p["f1"]
        dd = str(g["delay"]) if g["delay"] is not None else "miss"
        pd_ = str(p["delay"]) if p["delay"] is not None else "miss"
        print(f"  {r['node']:<6}  {da:>8.4f} {pa:>8.4f} {pa-da:>+7.4f}  "
              f"{df:>6.3f} {pf:>6.3f}  {dd:>8} {pd_:>8}")

    print(f"  {'─'*60}")
    avg_g_auroc = np.mean([r["global"]["auroc"]   for r in results])
    avg_p_auroc = np.mean([r["personal"]["auroc"] for r in results])
    avg_g_f1    = np.mean([r["global"]["f1"]      for r in results])
    avg_p_f1    = np.mean([r["personal"]["f1"]    for r in results])
    print(f"  {'avg':<6}  {avg_g_auroc:>8.4f} {avg_p_auroc:>8.4f} {avg_p_auroc-avg_g_auroc:>+7.4f}  "
          f"{avg_g_f1:>6.3f} {avg_p_f1:>6.3f}")
    print(f"{'═'*65}")
    print()


if __name__ == "__main__":
    main()
