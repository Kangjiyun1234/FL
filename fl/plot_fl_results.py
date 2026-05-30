"""
fl/plot_fl_results.py — FL 라운드별 탐지 성능 시각화

세 노드(mn1, mn2, mn3)의 AUROC / F1 / 탐지 지연을 라운드별로 플롯.

Usage:
  python3 fl/plot_fl_results.py
  python3 fl/plot_fl_results.py --data-dir /tmp/fl_data/femto --models-dir /tmp/fl_models/global
"""
import sys, os, re, argparse, pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append('/home/eunjin/federated-learning/fl')
from model import Conv1DAE
import config


# ── 유틸 ────────────────────────────────────────────────

def load_model(path, ae_cfg):
    m = Conv1DAE(n_channels=ae_cfg.n_channels,
                 latent_dim=ae_cfg.latent_dim,
                 seq_len=ae_cfg.seq_len)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m

def compute_scores(model, signals, batch_size=64):
    scores = []
    X = torch.tensor(signals, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size].unsqueeze(1)
            recon = model(xb)
            err = ((xb - recon)**2).mean(dim=(1,2))
            scores.append(err.numpy())
    return np.concatenate(scores)

def compute_auroc(scores, labels):
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")

def compute_f1(scores, labels, threshold):
    preds = (scores > threshold).astype(int)
    tp = int(((preds==1)&(labels==1)).sum())
    fp = int(((preds==1)&(labels==0)).sum())
    fn = int(((preds==0)&(labels==1)).sum())
    p = tp/(tp+fp) if (tp+fp)>0 else 0.0
    r = tp/(tp+fn) if (tp+fn)>0 else 0.0
    return 2*p*r/(p+r) if (p+r)>0 else 0.0

def online_detection(scores, labels, threshold, k=3):
    onset = int(np.where(labels==1)[0][0]) if (labels==1).any() else None
    det = None; consec = 0; fa = 0
    for i, (s, l) in enumerate(zip(scores, labels)):
        if s > threshold:
            consec += 1
            if consec >= k:
                det = i - k + 1
                if onset is not None and det < onset:
                    fa += 1; consec = 0; det = None
                else:
                    break
        else:
            consec = 0
    delay = max(0, det - onset) if (det is not None and onset is not None) else None
    return det, onset, delay, fa

def find_round_models(models_dir):
    pat = re.compile(r"global_round(\d+)\.pt$")
    results = []
    for fname in os.listdir(models_dir):
        m = pat.match(fname)
        if m:
            results.append((int(m.group(1)), os.path.join(models_dir, fname)))
    return sorted(results)

def evaluate_node(pkl_path, round_models, ae_cfg, n_sigma=3.0, k=3):
    with open(pkl_path, "rb") as f:
        ds = pickle.load(f)
    val_sigs   = ds["val_signals"]
    val_labels = ds["val_labels"]
    test_sigs  = ds["test_stream_signals"]
    test_labels= ds["test_stream_labels"]

    rows = []
    for rnd, mpath in round_models:
        model = load_model(mpath, ae_cfg)
        val_scores  = compute_scores(model, val_sigs)
        val_normal  = val_scores[val_labels == 0]
        threshold   = float(val_normal.mean() + n_sigma * val_normal.std())
        test_scores = compute_scores(model, test_sigs)
        auroc  = compute_auroc(test_scores, test_labels)
        f1     = compute_f1(test_scores, test_labels, threshold)
        det, onset, delay, fa = online_detection(test_scores, test_labels, threshold, k)
        rows.append(dict(round=rnd, auroc=auroc, f1=f1,
                         det=det, onset=onset, delay=delay, fa=fa,
                         threshold=threshold))
        det_str = str(det) if det is not None else "miss"
        print(f"    Round {rnd:2d}  AUROC={auroc:.4f}  F1={f1:.3f}  det={det_str}")
    return rows


# ── 플롯 ────────────────────────────────────────────────

def plot_results(all_results: dict, save_path: str):
    nodes  = list(all_results.keys())
    colors = {"mn1": "#2196F3", "mn2": "#FF9800", "mn3": "#4CAF50"}
    n_nodes = len(nodes)

    fig, axes = plt.subplots(3, n_nodes, figsize=(5*n_nodes, 11))
    if n_nodes == 1:
        axes = [[ax] for ax in axes]
    fig.suptitle("FL Round-by-Round Anomaly Detection Performance\n(FEMTO PRONOSTIA Bearing Dataset)",
                 fontsize=14, fontweight="bold", y=0.98)

    for col, node in enumerate(nodes):
        rows   = all_results[node]
        rounds = [r["round"] for r in rows]
        aurocs = [r["auroc"] for r in rows]
        f1s    = [r["f1"]    for r in rows]
        delays = [r["delay"] if r["delay"] is not None else float("nan") for r in rows]
        dets   = [r["det"]   for r in rows]
        color  = colors.get(node, "#9C27B0")

        # ── 행 0: AUROC ──────────────────────────────────
        ax0 = axes[0][col]
        ax0.plot(rounds, aurocs, "o-", color=color, linewidth=2, markersize=7)
        ax0.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax0.axhline(y=0.8, color="red",  linestyle="--", linewidth=0.8, alpha=0.4,
                    label="AUROC=0.8 기준선")
        ax0.fill_between(rounds, aurocs, alpha=0.15, color=color)
        ax0.set_ylim(0, 1.05)
        ax0.set_xlim(0.5, max(rounds)+0.5)
        ax0.set_title(f"{node.upper()}\n(Condition {col+1})", fontsize=12, fontweight="bold")
        ax0.set_ylabel("AUROC", fontsize=10)
        ax0.set_xticks(rounds)
        ax0.grid(True, alpha=0.3)
        ax0.legend(fontsize=7)
        # 값 표시
        for rnd, val in zip(rounds, aurocs):
            if not np.isnan(val):
                ax0.annotate(f"{val:.3f}", (rnd, val),
                             textcoords="offset points", xytext=(0, 7),
                             ha="center", fontsize=7, color=color)

        # ── 행 1: F1 ─────────────────────────────────────
        ax1 = axes[1][col]
        ax1.plot(rounds, f1s, "s-", color=color, linewidth=2, markersize=7)
        ax1.axhline(y=0.8, color="red", linestyle="--", linewidth=0.8, alpha=0.4,
                    label="F1=0.8 기준선")
        ax1.fill_between(rounds, f1s, alpha=0.15, color=color)
        ax1.set_ylim(0, 1.05)
        ax1.set_xlim(0.5, max(rounds)+0.5)
        ax1.set_ylabel("F1 Score", fontsize=10)
        ax1.set_xticks(rounds)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=7)
        for rnd, val in zip(rounds, f1s):
            ax1.annotate(f"{val:.3f}", (rnd, val),
                         textcoords="offset points", xytext=(0, 7),
                         ha="center", fontsize=7, color=color)

        # ── 행 2: 탐지 성공/실패 ─────────────────────────
        ax2 = axes[2][col]
        detected = [r for r in rows if r["det"] is not None]
        missed   = [r for r in rows if r["det"] is None]

        ax2.scatter([r["round"] for r in detected],
                    [r["delay"] if r["delay"] is not None else 0 for r in detected],
                    color=color, s=100, zorder=5, label="탐지 성공")
        ax2.scatter([r["round"] for r in missed],
                    [0]*len(missed),
                    color="red", marker="X", s=120, zorder=5, label="탐지 실패(miss)")

        if detected:
            det_rounds = [r["round"] for r in detected]
            det_delays = [r["delay"] if r["delay"] is not None else 0 for r in detected]
            ax2.bar(det_rounds, det_delays, color=color, alpha=0.3, width=0.4)

        ax2.set_ylabel("탐지 지연 (샘플 수)", fontsize=10)
        ax2.set_xlabel("FL Round", fontsize=10)
        ax2.set_xticks(rounds)
        ax2.set_xlim(0.5, max(rounds)+0.5)
        ymax = max([r["delay"] for r in detected if r["delay"] is not None], default=1)
        ax2.set_ylim(-0.5, max(ymax+2, 3))
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)

        # 첫 탐지 라운드 표시
        if detected:
            first = detected[0]["round"]
            ax0.axvline(x=first, color="green", linestyle=":", linewidth=1.5, alpha=0.7)
            ax1.axvline(x=first, color="green", linestyle=":", linewidth=1.5, alpha=0.7)
            ax2.axvline(x=first, color="green", linestyle=":", linewidth=1.5, alpha=0.7,
                        label=f"첫 탐지: Round {first}")
            ax2.legend(fontsize=8)

        # x축 라벨은 맨 아래만
        if col == 0:
            pass
        ax0.set_xlabel("")
        ax1.set_xlabel("")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  그래프 저장: {save_path}")
    plt.close()


# ── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   default="/tmp/fl_data/femto")
    parser.add_argument("--models-dir", default="/tmp/fl_models/global")
    parser.add_argument("--nodes",      default="mn1,mn2,mn3")
    parser.add_argument("--n-sigma",    type=float, default=3.0)
    parser.add_argument("--k",          type=int,   default=3)
    parser.add_argument("--out",        default="/tmp/fl_results_plot.png")
    args = parser.parse_args()

    nodes = args.nodes.split(",")
    round_models = find_round_models(args.models_dir)
    if not round_models:
        print(f"[ERROR] {args.models_dir}에 global_round*.pt 없음"); sys.exit(1)

    print(f"발견된 라운드: {[r for r,_ in round_models]}")
    print(f"평가 노드: {nodes}\n")

    ae_cfg = config.AE_CFG
    all_results = {}

    for node in nodes:
        pkl = os.path.join(args.data_dir, f"{node}.pkl")
        if not os.path.exists(pkl):
            print(f"  [{node}] pkl 없음, 스킵"); continue
        print(f"[{node}] 평가 중...")
        all_results[node] = evaluate_node(pkl, round_models, ae_cfg, args.n_sigma, args.k)

    if not all_results:
        print("[ERROR] 평가 결과 없음"); sys.exit(1)

    # ── 요약 출력 ──
    print("\n" + "="*55)
    print("  최종 요약 (Round 10 기준)")
    print("="*55)
    print(f"  {'Node':<6}  {'AUROC':>6}  {'F1':>6}  {'첫 탐지':>8}  {'지연':>6}")
    print("  " + "-"*45)
    for node, rows in all_results.items():
        last = rows[-1]
        first_det = next((r["round"] for r in rows if r["det"] is not None), None)
        det_str   = f"Round {first_det}" if first_det else "miss"
        delay_str = str(last["delay"]) if last["delay"] is not None else "-"
        auroc_str = f"{last['auroc']:.4f}" if not np.isnan(last["auroc"]) else "nan"
        print(f"  {node:<6}  {auroc_str:>6}  {last['f1']:>6.3f}  {det_str:>8}  {delay_str:>6}")
    print("="*55)

    plot_results(all_results, args.out)


if __name__ == "__main__":
    main()
