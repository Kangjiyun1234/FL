# fl/aggregator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from fl.model import SimpleRegressor


StateDict = Dict[str, torch.Tensor]


@dataclass
class Aggregator:
    in_dim: int
    device: str

    def __post_init__(self) -> None:
        self.global_model = SimpleRegressor(in_dim=self.in_dim).to(self.device)

    def get_global_state(self) -> StateDict:
        return {k: v.detach().cpu().clone() for k, v in self.global_model.state_dict().items()}

    def set_global_state(self, state: StateDict) -> None:
        self.global_model.load_state_dict(state, strict=True)

    def fedavg(self, client_states: List[Tuple[StateDict, int]]) -> StateDict:
        """
        FedAvg: 샘플 수로 가중 평균.
        client_states: [(state_dict, num_samples), ...]
        """
        total = sum(n for _, n in client_states)
        if total <= 0:
            raise ValueError("Total number of samples must be > 0")

        # Initialize with zeros on CPU
        keys = client_states[0][0].keys()
        avg_state: StateDict = {k: torch.zeros_like(client_states[0][0][k]) for k in keys}

        for state, n in client_states:
            w = n / total
            for k in keys:
                avg_state[k] += state[k] * w

        return avg_state
