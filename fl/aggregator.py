# fl/aggregator.py
from __future__ import annotations

from typing import Dict, List, Tuple
import torch

StateDict = Dict[str, torch.Tensor]


class Aggregator:
    """
    FedAvg Aggregator (모델 비의존적)
    """
    
    def aggregate(
        self, 
        client_states: List[Tuple[StateDict, int]]  #타입 수정
    ) -> StateDict:
        """
        FedAvg: 가중 평균 집계
        
        Args:
            client_states: [(state_dict, num_samples), ...]
        
        Returns:
            집계된 global state_dict
        """
        if not client_states:
            raise ValueError("No client states to aggregate")
        
        #튜플 언패킹
        states = [state for state, _ in client_states]
        weights = [num_samples for _, num_samples in client_states]
        
        total_samples = sum(weights)
        
        if total_samples == 0:
            raise ValueError("Total samples is zero")
        
        # 가중 평균
        global_state = {}
        
        for key in states[0].keys():
            # 각 클라이언트의 파라미터를 가중치로 곱해서 합산
            weighted_sum = sum(
                state[key] * (w / total_samples) 
                for state, w in zip(states, weights)
            )
            global_state[key] = weighted_sum
        
        return global_state