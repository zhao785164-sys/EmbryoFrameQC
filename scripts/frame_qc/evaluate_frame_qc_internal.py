import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18


ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen frame-QC model on its internal test split."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--index")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
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


def sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)

    return digest.hexdigest()


def to_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


class FrameDataset(Dataset):
    def __init__(self, table, transform):
        self.table = (
            table.reset_index(
                drop=True
            ).copy()
        )
        self.transform = transform

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        path = Path(
            str(row["image_path"])
        )

        try:
            with Image.open(path) as source:
                source.load()
                image = source.convert("RGB")
                image = self.transform(image)
        except Exception as error:
            raise RuntimeError(
                f"无法读取：{path}\n{error}"
            ) from error

        return (
            image,
            int(row["target"]),
            str(row["embryo_id"]),
            int(row["run"]),
            str(path),
        )


def calculate_metrics(
    labels,
    probabilities,
    threshold,
):
    labels = np.asarray(
        labels,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    has_valid = bool(
        (labels == 0).any()
    )

    has_invalid = bool(
        (labels == 1).any()
    )

    # AP和ROC-AUC只有在真实标签同时包含
    # VALID和INVALID时才有定义。
    if has_valid and has_invalid:
        average_precision = float(
            average_precision_score(
                labels,
                probabilities,
            )
        )

        roc_auc = float(
            roc_auc_score(
                labels,
                probabilities,
            )
        )
    else:
        average_precision = float("nan")
        roc_auc = float("nan")

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else float("nan")
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else float("nan")
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else float("nan")
    )

    false_positive_rate = (
        fp / (tn + fp)
        if (tn + fp) > 0
        else float("nan")
    )

    false_negative_rate = (
        fn / (tp + fn)
        if (tp + fn) > 0
        else float("nan")
    )

    f1_denominator = (
        2 * tp + fp + fn
    )

    f1 = (
        2 * tp / f1_denominator
        if f1_denominator > 0
        else float("nan")
    )

    if (
        np.isfinite(recall)
        and np.isfinite(specificity)
    ):
        balanced_accuracy = (
            recall + specificity
        ) / 2
    else:
        balanced_accuracy = float("nan")

    return {
        "threshold": float(threshold),
        "frame_count": int(
            len(labels)
        ),
        "true_valid": int(
            (labels == 0).sum()
        ),
        "true_invalid": int(
            (labels == 1).sum()
        ),
        "average_precision": (
            average_precision
        ),
        "roc_auc": roc_auc,
        "precision_invalid": float(
            precision
        ),
        "recall_invalid": float(
            recall
        ),
        "specificity_valid": float(
            specificity
        ),
        "false_positive_rate": float(
            false_positive_rate
        ),
        "false_negative_rate": float(
            false_negative_rate
        ),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "f1_invalid": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)

    protocol_path = (
        run_dir /
        "frozen_test_protocol_v1.json"
    )

    predictions_path = (
        run_dir /
        "test_predictions_frozen_v1.csv"
    )

    metrics_path = (
        run_dir /
        "test_metrics_frozen_v1.json"
    )

    per_embryo_path = (
        run_dir /
        "test_metrics_by_embryo_frozen_v1.csv"
    )

    output_paths = [
        predictions_path,
        metrics_path,
        per_embryo_path,
    ]

    existing = [
        str(path)
        for path in output_paths
        if path.exists()
    ]

    if existing:
        raise RuntimeError(
            "测试结果已经存在，禁止重复运行：\n"
            + "\n".join(existing)
        )

    if not protocol_path.exists():
        raise FileNotFoundError(
            f"缺少冻结方案：{protocol_path}"
        )

    protocol = json.loads(
        protocol_path.read_text(
            encoding="utf-8"
        )
    )

    if protocol["test_set_evaluated"]:
        raise RuntimeError(
            "冻结方案显示测试集已经评估"
        )

    checkpoint_path = Path(
        args.checkpoint or protocol["checkpoint_path"]
    )

    index_path = Path(
        args.index or protocol["index_path"]
    )

    if (
        sha256(checkpoint_path)
        != protocol[
            "checkpoint_sha256"
        ]
    ):
        raise RuntimeError(
            "模型哈希与冻结记录不一致"
        )

    if (
        sha256(index_path)
        != protocol["index_sha256"]
    ):
        raise RuntimeError(
            "索引哈希与冻结记录不一致"
        )

    threshold = float(
        protocol[
            "frozen_invalid_probability_threshold"
        ]
    )

    table = pd.read_csv(
        index_path
    )

    table["valid_bool"] = to_bool(
        table["is_frame_valid"]
    )

    table["target"] = (
        ~table["valid_bool"]
    ).astype(int)

    embryo_split_count = (
        table.groupby(
            "embryo_id"
        )["split"].nunique()
    )

    if (
        embryo_split_count > 1
    ).any():
        raise RuntimeError(
            "发现胚胎跨集合泄漏"
        )

    test_table = table[
        table["split"] == "test"
    ].copy()

    if (
        test_table[
            "embryo_id"
        ].nunique()
        != protocol["test_embryos"]
    ):
        raise RuntimeError(
            "测试集胚胎数与冻结记录不同"
        )

    if (
        len(test_table)
        != protocol["test_frames"]
    ):
        raise RuntimeError(
            "测试集帧数与冻结记录不同"
        )

    transform = transforms.Compose([
        transforms.Resize(
            (224, 224),
            interpolation=(
                transforms
                .InterpolationMode
                .BILINEAR
            ),
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ])

    dataset = FrameDataset(
        test_table,
        transform,
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

    model = resnet18(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        2,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if (
        args.expected_checkpoint_epoch is not None
        and int(checkpoint["epoch"]) != args.expected_checkpoint_epoch
    ):
        raise RuntimeError(
            "checkpoint epoch differs from --expected-checkpoint-epoch"
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(device)
    model.eval()

    labels_all = []
    probabilities_all = []
    embryo_ids_all = []
    runs_all = []
    paths_all = []

    with torch.inference_mode():
        for (
            images,
            labels,
            embryo_ids,
            runs,
            paths,
        ) in loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda"),
            ):
                logits = model(images)

            probabilities = (
                torch.softmax(
                    logits,
                    dim=1,
                )[:, 1]
                .cpu()
                .numpy()
            )

            labels_all.extend(
                labels.numpy().tolist()
            )
            probabilities_all.extend(
                probabilities.tolist()
            )
            embryo_ids_all.extend(
                list(embryo_ids)
            )
            runs_all.extend(
                runs.numpy().tolist()
            )
            paths_all.extend(
                list(paths)
            )

    predictions = pd.DataFrame({
        "embryo_id": embryo_ids_all,
        "run": runs_all,
        "image_path": paths_all,
        "target_invalid": labels_all,
        "probability_invalid": (
            probabilities_all
        ),
    })

    predictions[
        "prediction_invalid"
    ] = (
        predictions[
            "probability_invalid"
        ] >= threshold
    ).astype(int)

    predictions["error_type"] = (
        "correct"
    )

    predictions.loc[
        (
            predictions[
                "target_invalid"
            ] == 1
        )
        & (
            predictions[
                "prediction_invalid"
            ] == 0
        ),
        "error_type",
    ] = "false_negative"

    predictions.loc[
        (
            predictions[
                "target_invalid"
            ] == 0
        )
        & (
            predictions[
                "prediction_invalid"
            ] == 1
        ),
        "error_type",
    ] = "false_positive"

    metrics = calculate_metrics(
        labels=labels_all,
        probabilities=(
            probabilities_all
        ),
        threshold=threshold,
    )

    metrics.update({
        "evaluation_type": (
            "single final frozen test"
        ),
        "evaluated_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
        "protocol_path": str(
            protocol_path.resolve()
        ),
        "checkpoint_sha256": (
            protocol[
                "checkpoint_sha256"
            ]
        ),
        "index_sha256": (
            protocol[
                "index_sha256"
            ]
        ),
    })

    per_embryo_rows = []

    for embryo_id, group in (
        predictions.groupby(
            "embryo_id"
        )
    ):
        group_metrics = (
            calculate_metrics(
                labels=group[
                    "target_invalid"
                ].to_numpy(),
                probabilities=group[
                    "probability_invalid"
                ].to_numpy(),
                threshold=threshold,
            )
        )

        per_embryo_rows.append({
            "embryo_id": embryo_id,
            "frame_count": len(group),
            "true_invalid": int(
                group[
                    "target_invalid"
                ].sum()
            ),
            "predicted_invalid": int(
                group[
                    "prediction_invalid"
                ].sum()
            ),
            "false_negative": int(
                (
                    group["error_type"]
                    == "false_negative"
                ).sum()
            ),
            "false_positive": int(
                (
                    group["error_type"]
                    == "false_positive"
                ).sum()
            ),
            "recall_invalid": (
                group_metrics[
                    "recall_invalid"
                ]
            ),
            "specificity_valid": (
                group_metrics[
                    "specificity_valid"
                ]
            ),
        })

    per_embryo = pd.DataFrame(
        per_embryo_rows
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    per_embryo.to_csv(
        per_embryo_path,
        index=False,
    )

    with metrics_path.open(
        "x",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=2,
        )

    protocol[
        "test_set_evaluated"
    ] = True

    protocol[
        "test_evaluated_at"
    ] = metrics["evaluated_at"]

    protocol[
        "test_result_files"
    ] = {
        "predictions": str(
            predictions_path.resolve()
        ),
        "metrics": str(
            metrics_path.resolve()
        ),
        "per_embryo": str(
            per_embryo_path.resolve()
        ),
    }

    protocol[
        "test_result_sha256"
    ] = {
        "predictions": sha256(
            predictions_path
        ),
        "metrics": sha256(
            metrics_path
        ),
        "per_embryo": sha256(
            per_embryo_path
        ),
    }

    protocol_path.write_text(
        json.dumps(
            protocol,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 82)
    print("冻结测试集最终评估完成")
    print("=" * 82)
    print(
        "冻结阈值：",
        f"{threshold:.12f}",
    )
    print(
        "测试集帧数：",
        metrics["frame_count"],
    )
    print(
        "测试集AP：",
        f"{metrics['average_precision']:.6f}",
    )
    print(
        "测试集ROC-AUC：",
        f"{metrics['roc_auc']:.6f}",
    )
    print(
        "异常召回率：",
        f"{metrics['recall_invalid']:.6f}",
    )
    print(
        "正常帧特异度：",
        f"{metrics['specificity_valid']:.6f}",
    )
    print(
        "异常精确率：",
        f"{metrics['precision_invalid']:.6f}",
    )
    print(
        "正常帧误删率：",
        f"{metrics['false_positive_rate']:.6f}",
    )
    print(
        "混淆矩阵：",
        f"TN={metrics['tn']}，"
        f"FP={metrics['fp']}，"
        f"FN={metrics['fn']}，"
        f"TP={metrics['tp']}",
    )

    print("\n各测试集胚胎：")
    print(
        per_embryo.to_string(
            index=False
        )
    )

    print("\n错误数量：")
    print(
        predictions[
            "error_type"
        ].value_counts().to_string()
    )

    print("\n结果文件：")
    print(predictions_path)
    print(metrics_path)
    print(per_embryo_path)
    print(
        "\n该测试集已正式使用，"
        "不要再根据结果调整阈值后重测。"
    )


if __name__ == "__main__":
    main()
