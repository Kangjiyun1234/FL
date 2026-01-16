# fl/trainer.py
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from fl.aggregator import Aggregator, StateDict
from fl.config import FLJobConfig
from fl.data import make_node_dataset
from fl.edge_node import EdgeNode


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_global_model(agg: Aggregator, nodes: List[EdgeNode]) -> float:
    """
    전체 노드의 validation 데이터로 글로벌 모델 평가.
    - 노드별 loss와 가중 평균(global avg) 모두 출력.
    """
    agg.global_model.eval()
    criterion = torch.nn.MSELoss()

    per_node = []
    total_loss = 0.0
    total_samples = 0

    for n in nodes:
        Xv, yv = n.dataset.X_val, n.dataset.y_val
        preds = agg.global_model(Xv)
        loss = criterion(preds, yv).item()

        per_node.append((n.node_id, float(loss), int(Xv.shape[0])))

        total_loss += float(loss) * int(Xv.shape[0])
        total_samples += int(Xv.shape[0])

    avg_loss = total_loss / max(total_samples, 1)

    print("\n[Global Model Evaluation: Validation]")
    for node_id, loss, samples in per_node:
        print(f"  - Node {node_id}: val_loss={loss:.4f} (samples={samples})")
    print(f"  => Weighted Avg Val Loss: {avg_loss:.4f}")

    return avg_loss


def run_federated_learning(cfg: FLJobConfig) -> None:
    _set_seed(cfg.training.seed)

    device = cfg.training.device
    print(f"=== FL Job: {cfg.job_name} ===")
    print(
        f"Nodes: {cfg.num_nodes}, Rounds: {cfg.training.rounds}, "
        f"Local epochs: {cfg.training.local_epochs}, Device: {device}"
    )

    # Build nodes
    nodes: List[EdgeNode] = []
    for i in range(cfg.num_nodes):
        ds = make_node_dataset(
            node_id=i,
            num_samples=cfg.data.num_samples_per_node,
            num_features=cfg.data.num_features,
            noise_std=cfg.data.noise_std,
            non_iid=cfg.data.non_iid,
            val_ratio=cfg.data.val_ratio,
            seed=cfg.data.seed,
            device=device,
        )
        nodes.append(EdgeNode(node_id=i, dataset=ds, train_cfg=cfg.training))

    # Aggregator
    agg = Aggregator(in_dim=cfg.data.num_features, device=device)

    # Initialize all nodes with global weights
    global_state = agg.get_global_state()
    for n in nodes:
        n.set_state_dict(global_state)

    # FL rounds
    for r in range(1, cfg.training.rounds + 1):
        print(f"\n[Round {r}/{cfg.training.rounds}]")

        client_states: List[Tuple[StateDict, int]] = []
        local_losses = []

        # Local training
        for n in nodes:
            n.set_state_dict(global_state)  # sync
            train_loss = n.train_local()
            val_loss = n.evaluate_val()

            local_losses.append((n.node_id, train_loss, val_loss, n.num_train_samples))
            client_states.append((n.get_state_dict(), n.num_train_samples))

        # Aggregate
        new_global_state = agg.fedavg(client_states)
        agg.set_global_state(new_global_state)
        global_state = new_global_state

        # Logging
        if (r % cfg.training.log_every) == 0:
            for node_id, tr, va, ns in local_losses:
                print(f"  - Node {node_id}: train_loss={tr:.4f}, val_loss={va:.4f}, train_samples={ns}")

    print("\n=== Federated Learning Completed ===")
    evaluate_global_model(agg, nodes)
