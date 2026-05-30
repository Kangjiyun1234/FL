# main.py
import argparse
import yaml

from fl.config import FLJobConfig, DataConfig, TrainConfig, OneM2MJobConfig
from fl.trainer import run_federated_learning, run_coordinator_onem2m, run_trainer_onem2m


def load_config(yaml_path: str) -> FLJobConfig:
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    for k in ("job_name", "num_nodes", "data", "training"):
        if k not in cfg_dict:
            raise KeyError(f"Missing required config key: '{k}'")

    onem2m_cfg = None
    if "onem2m" in cfg_dict:
        onem2m_cfg = OneM2MJobConfig(**cfg_dict["onem2m"])

    return FLJobConfig(
        job_name=str(cfg_dict["job_name"]),
        num_nodes=int(cfg_dict["num_nodes"]),
        data=DataConfig(**cfg_dict["data"]),
        training=TrainConfig(**cfg_dict["training"]),
        onem2m=onem2m_cfg,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated Learning Demo Runner")
    parser.add_argument("--config", type=str, default="fljob.yaml", help="Path to YAML config")
    parser.add_argument("--role", choices=["standalone", "coordinator", "trainer"], default="standalone")
    parser.add_argument("--node-id", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.role == "standalone":
        run_federated_learning(cfg)
    elif args.role == "coordinator":
        run_coordinator_onem2m(cfg)
    else:
        run_trainer_onem2m(cfg, node_id=args.node_id)


if __name__ == "__main__":
    main()
