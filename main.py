# main.py
import argparse
import yaml

from fl.config import FLJobConfig, DataConfig, TrainConfig
from fl.trainer import run_federated_learning


def load_config(yaml_path: str) -> FLJobConfig:
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    # Basic validation (fail fast with clear messages)
    for k in ("job_name", "num_nodes", "data", "training"):
        if k not in cfg_dict:
            raise KeyError(f"Missing required config key: '{k}'")

    return FLJobConfig(
        job_name=str(cfg_dict["job_name"]),
        num_nodes=int(cfg_dict["num_nodes"]),
        data=DataConfig(**cfg_dict["data"]),
        training=TrainConfig(**cfg_dict["training"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated Learning Demo Runner")
    parser.add_argument("--config", type=str, default="fljob.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_federated_learning(cfg)


if __name__ == "__main__":
    main()
