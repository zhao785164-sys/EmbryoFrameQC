import argparse
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen frame-QC model on an external labelled set."
    )
    parser.add_argument("--labels", required=True)
    parser.add_argument("--ground-truth-protocol", required=True)
    parser.add_argument("--ground-truth-bundle", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--expected-embryos", type=int)
    parser.add_argument("--expected-invalid", type=int)
    parser.add_argument("--expected-valid", type=int)
    parser.add_argument("--expected-checkpoint-epoch", type=int)
    return parser.parse_args()


def resolve_device(requested):
    if requested == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device is unavailable: {requested}")
    return device


def check_expected(actual, expected, label):
    if expected is not None and int(actual) != int(expected):
        raise RuntimeError(
            f"Unexpected {label}: got {actual}, expected {expected}"
        )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def safe_divide(numerator, denominator):
    if denominator == 0:
        return None
    return float(numerator / denominator)


def calculate_metrics(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    has_valid = bool((labels == 0).any())
    has_invalid = bool((labels == 1).any())

    ap = None
    roc_auc = None

    if has_invalid:
        ap = float(
            average_precision_score(
                labels,
                probabilities,
            )
        )

    if has_valid and has_invalid:
        roc_auc = float(
            roc_auc_score(
                labels,
                probabilities,
            )
        )

    recall_invalid = safe_divide(tp, tp + fn)
    specificity_valid = safe_divide(tn, tn + fp)
    precision_invalid = safe_divide(tp, tp + fp)
    false_positive_rate = safe_divide(fp, tn + fp)
    accuracy = safe_divide(tp + tn, len(labels))

    if recall_invalid is not None and specificity_valid is not None:
        balanced_accuracy = float(
            (recall_invalid + specificity_valid) / 2
        )
    else:
        balanced_accuracy = None

    f1_invalid = safe_divide(
        2 * tp,
        2 * tp + fp + fn,
    )

    return {
        "frame_count": int(len(labels)),
        "true_valid": int((labels == 0).sum()),
        "true_invalid": int((labels == 1).sum()),
        "predicted_valid": int((predictions == 0).sum()),
        "predicted_invalid": int((predictions == 1).sum()),
        "average_precision": ap,
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision_invalid": precision_invalid,
        "recall_invalid": recall_invalid,
        "specificity_valid": specificity_valid,
        "false_positive_rate": false_positive_rate,
        "f1_invalid": f1_invalid,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


class ExternalFrameDataset(Dataset):
    def __init__(self, table, transform):
        self.table = table.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        path = Path(str(row["model_input_path"]))

        try:
            with Image.open(path) as source:
                source.load()
                image = source.convert("RGB")
                image = self.transform(image)
        except Exception as error:
            raise RuntimeError(
                f"无法读取外部测试图像：{path}\n{error}"
            ) from error

        return image, int(index)


def main():
    args = parse_args()
    label_path = Path(args.labels)
    ground_truth_protocol_path = Path(args.ground_truth_protocol)
    ground_truth_bundle_path = Path(args.ground_truth_bundle)
    model_run_dir = Path(args.run_dir)
    checkpoint_path = model_run_dir / "best_model_by_val_ap.pt"
    checkpoint_sha_path = model_run_dir / "best_model_by_val_ap.sha256"
    train_config_path = model_run_dir / "configuration.json"
    output_dir = Path(args.output_dir)
    threshold = float(args.threshold)

    required_files = [
        label_path,
        ground_truth_protocol_path,
        ground_truth_bundle_path,
        checkpoint_path,
        checkpoint_sha_path,
        train_config_path,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"找不到必要文件：{path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 固定推理环境
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # 检查 checkpoint 完整性
    checkpoint_sha = sha256(checkpoint_path)
    recorded_checkpoint_sha = (
        checkpoint_sha_path
        .read_text(encoding="utf-8")
        .strip()
        .split()[0]
    )

    if checkpoint_sha != recorded_checkpoint_sha:
        raise RuntimeError(
            "checkpoint SHA256 与冻结记录不一致，停止测试"
        )

    # 检查冻结标签完整性
    label_sha = sha256(label_path)
    bundle_text = ground_truth_bundle_path.read_text(
        encoding="utf-8"
    )

    if label_sha not in bundle_text:
        raise RuntimeError(
            "冻结标签 SHA256 未出现在 ground-truth bundle 中"
        )

    table = pd.read_csv(label_path)

    required_columns = {
        "embryo_id",
        "run",
        "target_invalid",
        "model_input_path",
    }

    missing_columns = required_columns - set(table.columns)
    if missing_columns:
        raise RuntimeError(
            f"冻结标签缺少字段：{sorted(missing_columns)}"
        )

    table["embryo_id"] = (
        table["embryo_id"]
        .astype(str)
        .str.strip()
    )
    table["run"] = pd.to_numeric(
        table["run"],
        errors="raise",
    ).astype(int)
    table["target_invalid"] = pd.to_numeric(
        table["target_invalid"],
        errors="raise",
    ).astype(int)

    check_expected(len(table), args.expected_frames, "frame count")
    check_expected(
        table["embryo_id"].nunique(),
        args.expected_embryos,
        "embryo count",
    )
    check_expected(
        int(table["target_invalid"].sum()),
        args.expected_invalid,
        "INVALID label count",
    )
    check_expected(
        int((table["target_invalid"] == 0).sum()),
        args.expected_valid,
        "VALID label count",
    )

    if set(table["target_invalid"].unique()) != {0, 1}:
        raise RuntimeError("标签取值必须恰好为0和1")

    if table.duplicated(["embryo_id", "run"]).any():
        raise RuntimeError("发现重复 embryo_id + run")

    missing_paths = [
        path for path in table["model_input_path"]
        if not Path(str(path)).exists()
    ]

    if missing_paths:
        raise RuntimeError(
            f"存在{len(missing_paths)}个缺失图像路径"
        )

    transform = transforms.Compose([
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = ExternalFrameDataset(
        table=table,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device != "cpu"),
        persistent_workers=(args.num_workers > 0),
    )

    device = resolve_device(args.device)

    # 架构与原冻结评估脚本完全一致
    model = resnet18(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features,
        2,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    checkpoint_epoch = int(checkpoint["epoch"])
    if (
        args.expected_checkpoint_epoch is not None
        and checkpoint_epoch != args.expected_checkpoint_epoch
    ):
        raise RuntimeError(
            f"checkpoint epoch={checkpoint_epoch}，"
            f"预期={args.expected_checkpoint_epoch}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model = model.to(device)
    model.eval()

    probabilities = np.full(
        len(table),
        np.nan,
        dtype=np.float64,
    )

    print("=" * 80)
    print("开始冻结外部测试推理")
    print("=" * 80)
    print(f"外部测试帧数：{len(table)}")
    print(f"外部测试胚胎数：{table['embryo_id'].nunique()}")
    print(f"冻结阈值：{threshold}")
    print(f"checkpoint epoch：{checkpoint_epoch}")
    print(f"checkpoint SHA256：{checkpoint_sha}")
    print(f"设备：{device}")

    with torch.inference_mode():
        for batch_number, (images, indices) in enumerate(
            loader,
            start=1,
        ):
            images = images.to(
                device,
                non_blocking=True,
            )

            logits = model(images)

            batch_probabilities = (
                torch.softmax(logits, dim=1)[:, 1]
                .detach()
                .cpu()
                .numpy()
            )

            indices = indices.numpy()
            probabilities[indices] = batch_probabilities

            if (
                batch_number == 1
                or batch_number % 10 == 0
                or batch_number == len(loader)
            ):
                print(
                    f"批次 {batch_number}/{len(loader)} 完成"
                )

    if np.isnan(probabilities).any():
        raise RuntimeError("部分帧没有得到预测概率")

    predictions = (
        probabilities >= threshold
    ).astype(int)

    result = table.copy()
    result["probability_invalid"] = probabilities
    result["prediction_invalid"] = predictions
    result["threshold"] = threshold

    result["error_type"] = np.select(
        [
            (
                (result["target_invalid"] == 0)
                & (result["prediction_invalid"] == 1)
            ),
            (
                (result["target_invalid"] == 1)
                & (result["prediction_invalid"] == 0)
            ),
        ],
        [
            "false_positive",
            "false_negative",
        ],
        default="correct",
    )

    overall_metrics = calculate_metrics(
        labels=result["target_invalid"].to_numpy(),
        probabilities=result["probability_invalid"].to_numpy(),
        threshold=threshold,
    )

    per_embryo_rows = []

    for embryo_id, group in result.groupby(
        "embryo_id",
        sort=True,
    ):
        row = {
            "embryo_id": embryo_id,
        }
        row.update(
            calculate_metrics(
                labels=group["target_invalid"].to_numpy(),
                probabilities=group[
                    "probability_invalid"
                ].to_numpy(),
                threshold=threshold,
            )
        )
        per_embryo_rows.append(row)

    per_embryo = pd.DataFrame(per_embryo_rows)

    predictions_path = (
        output_dir /
        "external_test_predictions_frozen_v1.csv"
    )
    metrics_path = (
        output_dir /
        "external_test_metrics_frozen_v1.json"
    )
    per_embryo_path = (
        output_dir /
        "external_test_metrics_by_embryo_frozen_v1.csv"
    )
    false_positive_path = (
        output_dir /
        "external_test_false_positives_frozen_v1.csv"
    )
    false_negative_path = (
        output_dir /
        "external_test_false_negatives_frozen_v1.csv"
    )
    protocol_path = (
        output_dir /
        "external_test_evaluation_protocol_v1.json"
    )
    bundle_path = (
        output_dir /
        "external_test_evaluation_bundle_v1.sha256"
    )

    result.to_csv(
        predictions_path,
        index=False,
    )

    per_embryo.to_csv(
        per_embryo_path,
        index=False,
    )

    result[
        result["error_type"].eq("false_positive")
    ].to_csv(
        false_positive_path,
        index=False,
    )

    result[
        result["error_type"].eq("false_negative")
    ].to_csv(
        false_negative_path,
        index=False,
    )

    metrics_document = {
        "evaluation_type": "independent_external_frozen_test",
        "positive_class": "invalid",
        "threshold": threshold,
        "metrics": overall_metrics,
    }

    metrics_path.write_text(
        json.dumps(
            metrics_document,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    protocol = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "evaluation_type": "independent_external_frozen_test",
        "test_used_once": True,
        "threshold_selected_from_external_test": False,
        "model_modified": False,
        "ground_truth_modified_during_inference": False,
        "positive_class": {
            "0": "VALID",
            "1": "INVALID",
        },
        "threshold": threshold,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "frozen_label_path": str(label_path),
        "frozen_label_sha256": label_sha,
        "ground_truth_protocol_sha256": sha256(
            ground_truth_protocol_path
        ),
        "training_configuration_sha256": sha256(
            train_config_path
        ),
        "preprocessing": {
            "color_mode": "RGB",
            "resize": [224, 224],
            "interpolation": "bilinear",
            "tensor_conversion": "ToTensor",
            "normalization_mean": [
                0.485,
                0.456,
                0.406,
            ],
            "normalization_std": [
                0.229,
                0.224,
                0.225,
            ],
        },
        "architecture": {
            "name": "ResNet18",
            "output_classes": 2,
            "final_layer": "Linear(in_features, 2)",
        },
        "frame_count": int(len(result)),
        "embryo_count": int(
            result["embryo_id"].nunique()
        ),
        "device": str(device),
    }

    protocol_path.write_text(
        json.dumps(
            protocol,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    bundle_files = [
        predictions_path,
        metrics_path,
        per_embryo_path,
        false_positive_path,
        false_negative_path,
        protocol_path,
    ]

    bundle_lines = [
        f"{sha256(path)}  {path.name}"
        for path in bundle_files
    ]

    bundle_path.write_text(
        "\n".join(bundle_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("冻结外部测试完成")
    print("=" * 80)
    print(json.dumps(
        overall_metrics,
        ensure_ascii=False,
        indent=2,
    ))

    print()
    print("各胚胎结果：")
    print(
        per_embryo[
            [
                "embryo_id",
                "frame_count",
                "true_invalid",
                "predicted_invalid",
                "fp",
                "fn",
                "tp",
                "recall_invalid",
                "specificity_valid",
            ]
        ].to_string(index=False)
    )

    print()
    print("结果文件：")
    for path in bundle_files:
        print(path.resolve())
    print(bundle_path.resolve())

    print()
    print(
        "注意：外部测试已正式使用，"
        "不得根据本次结果重新选择阈值后再报告同一测试集。"
    )


if __name__ == "__main__":
    main()
