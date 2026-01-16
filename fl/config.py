# fl/config.py
from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    num_samples_per_node: int = 800
    num_features: int = 10
    noise_std: float = 0.1
    non_iid: bool = True
    val_ratio: float = 0.2
    seed: int = 42


@dataclass(frozen=True)
class TrainConfig:
    rounds: int = 5
    local_epochs: int = 1
    batch_size: int = 128
    lr: float = 0.01
    weight_decay: float = 0.0
    device: str = "cpu"
    seed: int = 42
    log_every: int = 1


@dataclass(frozen=True)
class FLJobConfig:
    job_name: str
    num_nodes: int
    data: DataConfig
    training: TrainConfig
