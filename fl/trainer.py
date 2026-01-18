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
def evaluate_global_model(nodes: List[EdgeNode]) -> None:
    """
    전체 노드의 validation 데이터로 글로벌 모델 평가 (분류 loss 기준)
    """
    criterion = torch.nn.CrossEntropyLoss()

    print("\n[Global Model Evaluation: Validation]")

    total_loss = 0.0
    total_samples = 0

    for n in nodes:
        Xv, yv = n.val_data
        preds = n.model(Xv)
        loss = criterion(preds, yv).item()

        samples = len(Xv)
        total_loss += loss * samples
        total_samples += samples

        print(f"  - Node {n.node_id}: val_loss={loss:.4f} (samples={samples})")

    avg_loss = total_loss / max(total_samples, 1)
    print(f"  => Weighted Avg Val Loss: {avg_loss:.4f}")


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

    # -------------------------------------------------
    # 1) 데이터 로드 + 노드 분할 (딱 한 번)
    # -------------------------------------------------
    node_dfs, node_sensor_lists, all_sensors, _ = make_node_dataset(cfg.data, cfg.num_nodes)

    nodes = [
        EdgeNode(
            node_id=i,
            data=node_dfs[i],
            sensor_cols=all_sensors,   # <- 여기!
            train_cfg=cfg.training,
            device=device,
        )
        for i in range(cfg.num_nodes)
    ]

    # -------------------------------------------------
    # 3) Aggregator (모델 비의존)
    # -------------------------------------------------
    agg = Aggregator()

    # 초기 글로벌 가중치 = 첫 노드 기준
    global_state = nodes[0].get_state_dict()
    for n in nodes:
        n.set_state_dict(global_state)

    # -------------------------------------------------
    # 4) Federated Learning Rounds
    # -------------------------------------------------
    for r in range(1, cfg.training.rounds + 1):
        print(f"\n[Round {r}/{cfg.training.rounds}]")

        client_states: List[Tuple[StateDict, int]] = []

        for n in nodes:
            n.set_state_dict(global_state)  # sync
            train_loss, val_loss = n.train_local()

            client_states.append(
                (n.get_state_dict(), n.num_train_samples)
            )

            print(
                f"  - Node {n.node_id}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"train_samples={n.num_train_samples}"
            )

        # FedAvg
        global_state = agg.aggregate(client_states)

        # sync global weights
        for n in nodes:
            n.set_state_dict(global_state)

    print("\n=== Federated Learning Completed ===")
    evaluate_global_model(nodes)
