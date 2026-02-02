# fl/trainer.py
from __future__ import annotations
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from fl.aggregator import Aggregator, StateDict
from fl.config import FLJobConfig
from fl.data import make_node_dataset
from fl.edge_node import EdgeNode


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _binary_summary_from_cm(cm: np.ndarray) -> Dict[str, float]:
    """
    confusion_matrix = [[TN, FP],
                        [FN, TP]]
    """
    tn, fp, fn, tp = cm.ravel()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "P": p, "R": r, "F1": f1}


@torch.no_grad()
def evaluate_global_model(
    nodes: List[EdgeNode],
    mode: str,
    threshold: float = 0.8,
    pos_weight: float = 6.0,
    print_report: bool = True,
    print_node_losses: bool = True,
    node_loss_use_train: bool = False,
) -> Dict[str, float]:
    """
    글로벌 모델 평가 (Binary classification)
    - 각 노드의 (val 또는 train) 데이터로 node별 loss 산출
    - 전체 노드 데이터를 합쳐 confusion matrix / P,R,F1 계산

    Args:
        nodes: EdgeNode 리스트
        mode: 출력 라벨 문자열
        threshold: 분류 threshold (sigmoid(logits) 기준)
        pos_weight: BCEWithLogitsLoss pos_weight
        print_report: confusion matrix, classification report 출력 여부
        print_node_losses: 노드별 loss 출력 여부
        node_loss_use_train: True면 X_train/y_train로 loss 계산(보통 False)
    """
    device = nodes[0].device
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))

    print(f"\n[Global Model Evaluation: {mode}] (threshold={threshold})")

    # ---- Node-wise loss (weighted average) ----
    total_loss = 0.0
    total_samples = 0

    # ---- Global confusion ----
    all_preds: List[int] = []
    all_labels: List[int] = []

    for n in nodes:
        if node_loss_use_train:
            X, y = n.dataset.X_train, n.dataset.y_train
        else:
            X, y = n.dataset.X_val, n.dataset.y_val

        logits = n.model(X).squeeze(1)
        loss = float(criterion(logits, y.float()).item())

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).long()

        samples = int(y.shape[0])
        total_loss += loss * samples
        total_samples += samples

        all_preds.extend(preds.detach().cpu().numpy().astype(int).tolist())
        all_labels.extend(y.detach().cpu().numpy().astype(int).tolist())

        if print_node_losses:
            print(f"  - Node {n.node_id}: eval_loss={loss:.4f} (samples={samples})")

    avg_loss = total_loss / max(total_samples, 1)
    if print_node_losses:
        print(f"  => Weighted Avg Eval Loss: {avg_loss:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    summ = _binary_summary_from_cm(cm)

    print(
        f"[Summary] TP={summ['TP']} FP={summ['FP']} FN={summ['FN']} TN={summ['TN']} "
        f"| P={summ['P']:.4f} R={summ['R']:.4f} F1={summ['F1']:.4f}"
    )

    if print_report:
        print("\n[Confusion Matrix]")
        print(cm)

        print("\n[Classification Report]")
        # sklearn이 label을 0/1로 깔끔히 보여주도록 int로 통일
        print(classification_report(all_labels, all_preds, digits=4))

    return {"loss": avg_loss, **summ}


@torch.no_grad()
def threshold_sweep(
    nodes: List[EdgeNode],
    mode: str,
    thresholds: Optional[np.ndarray] = None,
    pos_weight: float = 6.0,
    pick_by: str = "f1",              # "f1" or "precision" or "recall"
    min_precision: float = 0.0,       # 예: 0.6
    min_recall: float = 0.0,          # 예: 0.8
    print_each: bool = True,
) -> Dict[str, float]:
    """
    threshold sweep (sigmoid 확률 기준).
    - 모든 노드 데이터를 합쳐서 y_true/y_prob 생성
    - best threshold 선택(기본: F1 최대)
    - (선택) precision/recall 최소조건 걸 수 있음

    Returns:
        best dict: {"thr":..., "P":..., "R":..., "F1":..., "TP":..., ...}
    """
    device = nodes[0].device
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    # collect all probs/labels
    y_true_all: List[int] = []
    y_prob_all: List[float] = []

    total_loss = 0.0
    total_samples = 0

    for n in nodes:
        X, y = n.dataset.X_val, n.dataset.y_val
        logits = n.model(X).squeeze(1)
        loss = float(criterion(logits, y.float()).item())

        probs = torch.sigmoid(logits)

        samples = int(y.shape[0])
        total_loss += loss * samples
        total_samples += samples

        y_true_all.extend(y.detach().cpu().numpy().astype(int).tolist())
        y_prob_all.extend(probs.detach().cpu().numpy().astype(float).tolist())

    base_loss = total_loss / max(total_samples, 1)

    best: Optional[Dict[str, float]] = None

    if print_each:
        print(f"\n[Threshold Sweep: {mode}] (base_loss={base_loss:.4f})")
        print("thr |   P    R    F1  |  TP   FP   FN   TN")

    y_true_np = np.asarray(y_true_all, dtype=int)
    y_prob_np = np.asarray(y_prob_all, dtype=float)

    for t in thresholds:
        y_pred = (y_prob_np >= t).astype(int)
        cm = confusion_matrix(y_true_np, y_pred)
        tn, fp, fn, tp = cm.ravel()

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        if print_each:
            print(f"{t:>.2f} | {p:>.3f} {r:>.3f} {f1:>.3f} | {tp:>4d} {fp:>4d} {fn:>4d} {tn:>4d}")

        # constraints
        if p < min_precision or r < min_recall:
            continue

        score = {"f1": f1, "precision": p, "recall": r}.get(pick_by, f1)

        if best is None or score > best["score"]:
            best = {
                "score": score,
                "thr": float(t),
                "P": float(p),
                "R": float(r),
                "F1": float(f1),
                "TP": float(tp),
                "FP": float(fp),
                "FN": float(fn),
                "TN": float(tn),
                "loss": float(base_loss),
            }

    if best is None:
        # 조건이 너무 빡세서 best가 없으면, 제약 없이 F1 최대를 반환
        best = {"score": -1.0}
        for t in thresholds:
            y_pred = (y_prob_np >= t).astype(int)
            cm = confusion_matrix(y_true_np, y_pred)
            summ = _binary_summary_from_cm(cm)
            if summ["F1"] > best.get("F1", -1.0):
                best = {
                    "score": summ["F1"],
                    "thr": float(t),
                    "P": float(summ["P"]),
                    "R": float(summ["R"]),
                    "F1": float(summ["F1"]),
                    "TP": float(summ["TP"]),
                    "FP": float(summ["FP"]),
                    "FN": float(summ["FN"]),
                    "TN": float(summ["TN"]),
                    "loss": float(base_loss),
                }

    print(
        f"\n[BEST ({pick_by})] thr={best['thr']:.2f} | P={best['P']:.4f} R={best['R']:.4f} F1={best['F1']:.4f} "
        f"| TP={int(best['TP'])} FP={int(best['FP'])} FN={int(best['FN'])} TN={int(best['TN'])}"
    )
    return best


def run_federated_learning(cfg: FLJobConfig) -> None:
    _set_seed(cfg.training.seed)
    device = cfg.training.device

    print(f"=== FL Job: {cfg.job_name} ===")
    print(
        f"Nodes: {cfg.num_nodes}, "
        f"Rounds: {cfg.training.rounds}, "
        f"Local epochs: {cfg.training.local_epochs}, "
        f"Device: {device}"
    )

    # =================================================
    # PHASE 1: TRAIN (Federated Learning)
    # =================================================
    node_dfs, node_sensor_lists, all_sensors, meta = make_node_dataset(cfg.data, cfg.num_nodes)

    nodes: List[EdgeNode] = []
    for i in range(cfg.num_nodes):
        nodes.append(
            EdgeNode(
                node_id=i + 1,
                data=node_dfs[i],
                sensor_cols=all_sensors,
                train_cfg=cfg.training,
                device=device,
            )
        )

    agg = Aggregator()
    global_state = nodes[0].get_state_dict()

    # sync
    for n in nodes:
        n.set_state_dict(global_state)

    # ---- Training Loop ----
    for r in range(1, cfg.training.rounds + 1):
        print(f"\n[Round {r}/{cfg.training.rounds}]")

        client_states: List[Tuple[StateDict, int]] = []

        # round weighted avg (train/val)
        round_train_sum, round_val_sum, round_samples = 0.0, 0.0, 0

        for n in nodes:
            n.set_state_dict(global_state)

            train_loss, val_loss, _ = n.train_local()
            train_samples = n.num_train_samples

            client_states.append((n.get_state_dict(), train_samples))

            print(
                f"  - Node {n.node_id}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"train_samples={train_samples}"
            )

            round_train_sum += float(train_loss) * train_samples
            round_val_sum += float(val_loss) * train_samples
            round_samples += train_samples

        round_train_avg = round_train_sum / max(round_samples, 1)
        round_val_avg = round_val_sum / max(round_samples, 1)

        print(
            f"  [Round Summary] "
            f"train_loss(w.avg)={round_train_avg:.4f}, "
            f"val_loss(w.avg)={round_val_avg:.4f}, "
            f"total_samples={round_samples}"
        )

        # aggregate + sync
        global_state = agg.aggregate(client_states)
        for n in nodes:
            n.set_state_dict(global_state)

    print("\n=== Federated Learning Completed ===")

    # ---- Global Eval on Train/Validation ----
    eval_thr = 0.8  # 네가 쓰는 기준값 유지
    evaluate_global_model(
        nodes,
        mode="Train/Validation",
        threshold=eval_thr,
        pos_weight=6.0,
        print_report=True,
        print_node_losses=True,
        node_loss_use_train=False,
    )

    # (선택) Train/Val threshold sweep 보고 싶으면 True로
    DO_SWEEP_TRAIN = False
    if DO_SWEEP_TRAIN:
        threshold_sweep(
            nodes,
            mode="Train/Validation",
            thresholds=np.linspace(0.05, 0.95, 19),
            pos_weight=6.0,
            pick_by="f1",
            min_precision=0.0,
            min_recall=0.0,
            print_each=True,
        )

    # =================================================
    # PHASE 2: TEST (NASA only) - FULL
    # =================================================
    if meta.get("dataset", "").startswith("FD"):
        print("\n" + "=" * 60)
        print("PHASE 2: EVALUATION ON TEST DATA (FULL)")
        print("=" * 60)

        from fl.data_nasa import prepare_nasa_for_fl

        test_node_dfs, sensor_cols, _ = prepare_nasa_for_fl(
            data_path=f"{cfg.data.data_dir}/nasa_turbofan",
            dataset=meta["dataset"],
            num_nodes=cfg.num_nodes,
            use_test=True,
        )

        test_nodes: List[EdgeNode] = []
        for i in range(cfg.num_nodes):
            n = EdgeNode(
                node_id=i + 1,
                data=test_node_dfs[i],
                sensor_cols=sensor_cols,
                train_cfg=cfg.training,
                device=device,
            )
            n.set_state_dict(global_state)  # trained global weights
            test_nodes.append(n)

        evaluate_global_model(
            test_nodes,
            mode="Test (FULL)",
            threshold=eval_thr,
            pos_weight=6.0,
            print_report=True,
            print_node_losses=True,
            node_loss_use_train=False,
        )

        # (선택) Test FULL sweep도 보고 싶으면 True로
        DO_SWEEP_TEST = False
        if DO_SWEEP_TEST:
            threshold_sweep(
                test_nodes,
                mode="Test (FULL)",
                thresholds=np.linspace(0.05, 0.95, 19),
                pos_weight=6.0,
                pick_by="f1",          # 혹은 "precision" / "recall"
                min_precision=0.0,     # 예: 0.6
                min_recall=0.0,        # 예: 0.8
                print_each=True,
            )
