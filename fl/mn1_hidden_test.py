import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)


class InferredMLP(nn.Module):
    def __init__(self, state_dict: dict):
        super().__init__()
        self.layer_keys = []
        self.layers = nn.ModuleList()

        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor) and v.ndim == 2:
                self.layer_keys.append(k)

        if not self.layer_keys:
            raise ValueError("state_dict에서 linear weight를 찾지 못했습니다.")

        for w_key in self.layer_keys:
            w = state_dict[w_key]
            out_dim, in_dim = w.shape
            layer = nn.Linear(in_dim, out_dim, bias=True)
            layer.weight.data.copy_(w)

            b_key = w_key.replace(".weight", ".bias")
            if b_key in state_dict:
                layer.bias.data.copy_(state_dict[b_key])
            else:
                layer.bias.data.zero_()

            self.layers.append(layer)

    def forward(self, x):
        if x.ndim > 2:
            x = x.view(x.size(0), -1)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


def load_checkpoint_state_dict(model_path: str):
    ckpt = torch.load(model_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]

    if isinstance(ckpt, dict):
        tensor_like = [isinstance(v, torch.Tensor) for v in ckpt.values()]
        if len(tensor_like) > 0 and any(tensor_like):
            return ckpt

    raise ValueError(f"지원되지 않는 체크포인트 형식: {model_path}")


def evaluate_at_threshold(y_true, anom_prob, threshold):
    pred = (anom_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)

    return {
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "pred": pred,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--hidden-test", required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="지정하면 해당 threshold만 평가. 지정 안 하면 sweep 수행"
    )
    parser.add_argument(
        "--normal-class",
        type=int,
        default=0,
        help="멀티클래스 모델에서 정상으로 간주할 클래스 index (기본값: 0)"
    )
    args = parser.parse_args()

    with open(args.hidden_test, "rb") as f:
        d = pickle.load(f)

    X = d["features"].astype(np.float32)
    y = d["labels"].astype(np.int64)

    # hidden test 라벨은 binary(0=정상, 1=이상)라고 가정
    unique_y = np.unique(y)
    if not set(unique_y).issubset({0, 1}):
        raise ValueError(
            f"hidden test labels는 binary(0/1)여야 합니다. 현재 unique={unique_y}"
        )

    print("model      :", args.model)
    print("hidden_test:", args.hidden_test)
    print("num_samples:", len(y))
    print("label_count:", np.bincount(y, minlength=2))

    state_dict = load_checkpoint_state_dict(args.model)
    model = InferredMLP(state_dict)
    model.eval()

    x_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    num_model_classes = probs.shape[1]
    print("model_output_classes:", num_model_classes)

    # ---------------------------
    # anomaly probability 정의
    # ---------------------------
    if num_model_classes == 2:
        # binary classifier
        anomaly_classes = [1]
        anom_prob = probs[:, 1]
        argmax_class = np.argmax(probs, axis=1)
        argmax_pred_binary = argmax_class.astype(int)
    else:
        # multiclass classifier
        normal_class = args.normal_class
        if normal_class < 0 or normal_class >= num_model_classes:
            raise ValueError(
                f"--normal-class={normal_class} is invalid for num_model_classes={num_model_classes}"
            )

        anomaly_classes = [c for c in range(num_model_classes) if c != normal_class]
        anom_prob = probs[:, anomaly_classes].sum(axis=1)

        argmax_class = np.argmax(probs, axis=1)
        argmax_pred_binary = (argmax_class != normal_class).astype(int)

    print("normal_class    :", args.normal_class)
    print("anomaly_classes :", anomaly_classes)

    print("\n[anomaly probability stats]")
    print(f"min={anom_prob.min():.4f}, max={anom_prob.max():.4f}, mean={anom_prob.mean():.4f}")

    if args.threshold is not None:
        thresholds = [args.threshold]
    else:
        thresholds = np.arange(0.05, 1.00, 0.05)

    results = []
    for th in thresholds:
        r = evaluate_at_threshold(y, anom_prob, th)
        results.append(r)

    print("\n=== Threshold Sweep ===")
    print(f"{'th':>6} {'acc':>8} {'prec':>8} {'rec':>8} {'f1':>8}")
    for r in results:
        print(
            f"{r['threshold']:>6.2f} "
            f"{r['accuracy']:>8.4f} "
            f"{r['precision']:>8.4f} "
            f"{r['recall']:>8.4f} "
            f"{r['f1']:>8.4f}"
        )

    best = max(results, key=lambda x: x["f1"])
    best_th = best["threshold"]
    best_pred = best["pred"]

    print("\n=== Best Threshold (by anomaly F1) ===")
    print(f"threshold : {best_th:.2f}")
    print(f"accuracy  : {best['accuracy']:.4f}")
    print(f"precision : {best['precision']:.4f}")
    print(f"recall    : {best['recall']:.4f}")
    print(f"f1        : {best['f1']:.4f}")

    print("\n[classification_report @ best threshold]")
    print(classification_report(
        y,
        best_pred,
        target_names=["정상", "이상"],
        digits=4,
        zero_division=0
    ))

    print("[confusion_matrix @ best threshold]")
    print(confusion_matrix(y, best_pred))

    # ---------------------------
    # Argmax baseline (binary 변환 후)
    # ---------------------------
    print("\n=== Argmax Baseline (binary) ===")
    print("accuracy :", f"{accuracy_score(y, argmax_pred_binary):.4f}")
    print("precision:", f"{precision_score(y, argmax_pred_binary, zero_division=0):.4f}")
    print("recall   :", f"{recall_score(y, argmax_pred_binary, zero_division=0):.4f}")
    print("f1       :", f"{f1_score(y, argmax_pred_binary, zero_division=0):.4f}")
    print(confusion_matrix(y, argmax_pred_binary))

    # 참고용 multiclass argmax 분포 출력
    print("\n[argmax class distribution]")
    print(np.bincount(argmax_class, minlength=num_model_classes))


if __name__ == "__main__":
    main()