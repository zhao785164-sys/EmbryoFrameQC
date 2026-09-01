import argparse
import csv
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


# 允许读取已确认显示完整的截断JPEG
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--index",
        required=True,
        help="CSV containing embryo_id, run, split, is_frame_valid and image_path",
    )
    parser.add_argument(
        "--output-root",
        default=(
            "result/movement_screening_test30/"
            "resnet18_frame_qc_runs_v1"
        ),
    )
    parser.add_argument("--output-tag", default="full")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device (auto, cpu, cuda or cuda:0)",
    )
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--expected-embryos", type=int)
    parser.add_argument("--expected-valid", type=int)
    parser.add_argument("--expected-invalid", type=int)
    parser.add_argument("--expected-train-embryos", type=int)
    parser.add_argument("--expected-val-embryos", type=int)
    parser.add_argument("--expected-test-embryos", type=int)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        raise ValueError(
            f"Unexpected {label}: got {actual}, expected {expected}"
        )


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


class FrameQCDataset(Dataset):
    def __init__(self, table, transform):
        self.table = table.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        path = Path(str(row["image_path"]))

        try:
            with Image.open(path) as source:
                source.load()
                image = source.convert("RGB")
                image = self.transform(image)
        except Exception as error:
            raise RuntimeError(
                f"无法读取图像：{path}\n{error}"
            ) from error

        target = torch.tensor(
            int(row["target"]),
            dtype=torch.long,
        )

        return (
            image,
            target,
            str(row["embryo_id"]),
            int(row["run"]),
        )


def build_transforms():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # 翻转不会改变胚胎是否可用；
    # 不使用裁剪、平移和亮度扰动，以免改变QC标签含义。
    train_transform = transforms.Compose([
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, eval_transform


def calculate_metrics(labels, probabilities, threshold=0.5):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            pos_label=1,
            zero_division=0,
        )
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else float("nan")
    )

    balanced_accuracy = (
        (recall + specificity) / 2
    )

    return {
        "ap": float(
            average_precision_score(
                labels,
                probabilities,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                labels,
                probabilities,
            )
        ),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    max_batches,
):
    model.train()

    total_loss = 0.0
    total_count = 0
    labels_all = []
    probabilities_all = []

    use_amp = device.type == "cuda"

    for batch_number, (
        images,
        labels,
        _,
        _,
    ) in enumerate(loader, start=1):

        if (
            max_batches is not None
            and batch_number > max_batches
        ):
            break

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        probabilities = (
            torch.softmax(logits.detach(), dim=1)[:, 1]
            .cpu()
            .numpy()
        )

        batch_size = labels.size(0)

        total_loss += (
            float(loss.item()) * batch_size
        )
        total_count += batch_size

        labels_all.extend(
            labels.detach().cpu().numpy().tolist()
        )

        probabilities_all.extend(
            probabilities.tolist()
        )

    metrics = calculate_metrics(
        labels_all,
        probabilities_all,
        threshold=0.5,
    )

    metrics["loss"] = (
        total_loss / total_count
    )

    metrics["sample_count"] = int(
        total_count
    )

    return metrics


@torch.inference_mode()
def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total_count = 0

    labels_all = []
    probabilities_all = []
    embryo_ids_all = []
    runs_all = []

    use_amp = device.type == "cuda"

    for (
        images,
        labels,
        embryo_ids,
        runs,
    ) in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels_device = labels.to(
            device,
            non_blocking=True,
        )

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = criterion(
                logits,
                labels_device,
            )

        probabilities = (
            torch.softmax(logits, dim=1)[:, 1]
            .cpu()
            .numpy()
        )

        batch_size = labels.size(0)

        total_loss += (
            float(loss.item()) * batch_size
        )

        total_count += batch_size

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

    metrics = calculate_metrics(
        labels_all,
        probabilities_all,
        threshold=0.5,
    )

    metrics["loss"] = (
        total_loss / total_count
    )

    metrics["sample_count"] = int(
        total_count
    )

    prediction_table = pd.DataFrame({
        "embryo_id": embryo_ids_all,
        "run": runs_all,
        "target_invalid": labels_all,
        "probability_invalid": probabilities_all,
    })

    return metrics, prediction_table


def choose_recall_threshold(
    labels,
    probabilities,
    target_recall,
):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    precision, recall, thresholds = (
        precision_recall_curve(
            labels,
            probabilities,
        )
    )

    eligible = np.where(
        recall[:-1] >= target_recall
    )[0]

    if len(eligible) == 0:
        raise RuntimeError(
            "验证集上无法达到目标异常召回率"
        )

    # 在召回率达到目标的情况下，
    # 选择最高阈值以减少正常帧误隔离。
    chosen_index = eligible[
        np.argmax(thresholds[eligible])
    ]

    return float(
        thresholds[chosen_index]
    )


def sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def main():
    args = parse_args()
    set_seed(args.seed)

    index_path = Path(args.index)

    if not index_path.exists():
        raise FileNotFoundError(
            f"找不到训练索引：{index_path}"
        )

    device = resolve_device(args.device)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_root = Path(args.output_root)

    output_dir = (
        output_root /
        f"{args.output_tag}_{timestamp}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    df = pd.read_csv(index_path)

    required = {
        "embryo_id",
        "run",
        "split",
        "is_frame_valid",
        "image_path",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"索引缺少字段：{sorted(missing)}"
        )

    df["embryo_id"] = (
        df["embryo_id"]
        .astype(str)
        .str.strip()
    )

    df["run"] = pd.to_numeric(
        df["run"],
        errors="raise",
    ).astype(int)

    df["valid_bool"] = to_bool(
        df["is_frame_valid"]
    )

    # 标签定义：
    # VALID = 0
    # INVALID = 1
    df["target"] = (
        ~df["valid_bool"]
    ).astype(int)

    check_expected(len(df), args.expected_frames, "frame count")
    check_expected(
        df["embryo_id"].nunique(),
        args.expected_embryos,
        "embryo count",
    )
    check_expected(
        int((df["target"] == 0).sum()),
        args.expected_valid,
        "VALID frame count",
    )
    check_expected(
        int((df["target"] == 1).sum()),
        args.expected_invalid,
        "INVALID frame count",
    )

    embryo_split_count = (
        df.groupby("embryo_id")["split"]
        .nunique()
    )

    if (embryo_split_count > 1).any():
        raise ValueError(
            "发现embryo_id跨集合泄漏"
        )

    train_df = df[
        df["split"] == "train"
    ].copy()

    val_df = df[
        df["split"] == "val"
    ].copy()

    test_df = df[
        df["split"] == "test"
    ].copy()

    check_expected(
        train_df["embryo_id"].nunique(),
        args.expected_train_embryos,
        "training embryo count",
    )
    check_expected(
        val_df["embryo_id"].nunique(),
        args.expected_val_embryos,
        "validation embryo count",
    )
    check_expected(
        test_df["embryo_id"].nunique(),
        args.expected_test_embryos,
        "test embryo count",
    )

    train_transform, eval_transform = (
        build_transforms()
    )

    train_dataset = FrameQCDataset(
        train_df,
        train_transform,
    )

    val_dataset = FrameQCDataset(
        val_df,
        eval_transform,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(
            args.num_workers > 0
        ),
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    train_labels = train_df[
        "target"
    ].to_numpy()

    valid_count = int(
        (train_labels == 0).sum()
    )

    invalid_count = int(
        (train_labels == 1).sum()
    )

    total_count = (
        valid_count + invalid_count
    )

    class_weights = torch.tensor(
        [
            total_count / (2 * valid_count),
            total_count / (2 * invalid_count),
        ],
        dtype=torch.float32,
        device=device,
    )

    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        2,
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        )
    )

    scaler = torch.amp.GradScaler(
        device.type,
        enabled=(device.type == "cuda"),
    )

    configuration = {
        "created_at": datetime.now().isoformat(),
        "index_path": str(index_path),
        "index_sha256": sha256(index_path),
        "output_dir": str(output_dir),
        "label_definition": {
            "0": "VALID",
            "1": "INVALID",
        },
        "model": "ImageNet-pretrained ResNet18",
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "seed": args.seed,
        "target_invalid_recall": (
            args.target_recall
        ),
        "max_train_batches": (
            args.max_train_batches
        ),
        "train_embryos": int(
            train_df["embryo_id"].nunique()
        ),
        "val_embryos": int(
            val_df["embryo_id"].nunique()
        ),
        "test_embryos_untouched": int(
            test_df["embryo_id"].nunique()
        ),
        "train_frames": len(train_df),
        "val_frames": len(val_df),
        "test_frames_untouched": len(test_df),
        "class_weights": {
            "valid_0": float(
                class_weights[0].item()
            ),
            "invalid_1": float(
                class_weights[1].item()
            ),
        },
    }

    with (
        output_dir /
        "configuration.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            configuration,
            file,
            ensure_ascii=False,
            indent=2,
        )

    history_path = (
        output_dir /
        "training_history.csv"
    )

    best_checkpoint_path = (
        output_dir /
        "best_model_by_val_ap.pt"
    )

    history_fields = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_ap",
        "train_roc_auc",
        "train_recall_at_0_5",
        "train_specificity_at_0_5",
        "val_loss",
        "val_ap",
        "val_roc_auc",
        "val_precision_at_0_5",
        "val_recall_at_0_5",
        "val_specificity_at_0_5",
        "val_f1_at_0_5",
    ]

    best_val_ap = -float("inf")
    best_epoch = None
    epochs_without_improvement = 0

    print("=" * 84)
    print("ResNet18单帧QC训练开始")
    print("=" * 84)
    print("设备：", device)
    print("输出目录：", output_dir)
    print("训练帧数：", len(train_df))
    print("验证帧数：", len(val_df))
    print("测试帧数：", len(test_df), "（不使用）")
    print("VALID权重：", class_weights[0].item())
    print("INVALID权重：", class_weights[1].item())

    with history_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as history_file:

        writer = csv.DictWriter(
            history_file,
            fieldnames=history_fields,
        )

        writer.writeheader()

        for epoch in range(
            1,
            args.epochs + 1,
        ):
            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                max_batches=(
                    args.max_train_batches
                ),
            )

            val_metrics, _ = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )

            learning_rate = (
                optimizer.param_groups[0]["lr"]
            )

            record = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": train_metrics["loss"],
                "train_ap": train_metrics["ap"],
                "train_roc_auc": (
                    train_metrics["roc_auc"]
                ),
                "train_recall_at_0_5": (
                    train_metrics["recall"]
                ),
                "train_specificity_at_0_5": (
                    train_metrics["specificity"]
                ),
                "val_loss": val_metrics["loss"],
                "val_ap": val_metrics["ap"],
                "val_roc_auc": (
                    val_metrics["roc_auc"]
                ),
                "val_precision_at_0_5": (
                    val_metrics["precision"]
                ),
                "val_recall_at_0_5": (
                    val_metrics["recall"]
                ),
                "val_specificity_at_0_5": (
                    val_metrics["specificity"]
                ),
                "val_f1_at_0_5": (
                    val_metrics["f1"]
                ),
            }

            writer.writerow(record)
            history_file.flush()

            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"lr={learning_rate:.2e} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"train_AP={train_metrics['ap']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_AP={val_metrics['ap']:.4f} | "
                f"val_recall@0.5={val_metrics['recall']:.4f} | "
                f"val_specificity@0.5="
                f"{val_metrics['specificity']:.4f}"
            )

            scheduler.step(
                val_metrics["ap"]
            )

            if (
                val_metrics["ap"]
                > best_val_ap + 1e-6
            ):
                best_val_ap = (
                    val_metrics["ap"]
                )

                best_epoch = epoch
                epochs_without_improvement = 0

                checkpoint = {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "configuration": configuration,
                }

                torch.save(
                    checkpoint,
                    best_checkpoint_path,
                )

                print(
                    "  已保存新的最佳模型："
                    f"val_AP={best_val_ap:.4f}"
                )
            else:
                epochs_without_improvement += 1

                if (
                    epochs_without_improvement
                    >= args.patience
                ):
                    print(
                        "触发早停：连续"
                        f"{args.patience}个epoch"
                        "验证集AP未改善。"
                    )
                    break

    # ========================================================
    # 使用最佳模型在验证集选择高召回率阈值
    # ========================================================

    checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    val_metrics_05, val_predictions = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
    )

    chosen_threshold = (
        choose_recall_threshold(
            labels=val_predictions[
                "target_invalid"
            ].to_numpy(),
            probabilities=val_predictions[
                "probability_invalid"
            ].to_numpy(),
            target_recall=args.target_recall,
        )
    )

    val_metrics_chosen = calculate_metrics(
        labels=val_predictions[
            "target_invalid"
        ].to_numpy(),
        probabilities=val_predictions[
            "probability_invalid"
        ].to_numpy(),
        threshold=chosen_threshold,
    )

    val_predictions[
        "prediction_at_threshold"
    ] = (
        val_predictions[
            "probability_invalid"
        ] >= chosen_threshold
    ).astype(int)

    val_predictions.to_csv(
        output_dir /
        "validation_predictions_best_model.csv",
        index=False,
    )

    threshold_summary = {
        "best_epoch": int(best_epoch),
        "best_validation_ap": float(
            best_val_ap
        ),
        "target_invalid_recall": float(
            args.target_recall
        ),
        "selected_invalid_probability_threshold": (
            float(chosen_threshold)
        ),
        "validation_metrics_at_0_5": (
            val_metrics_05
        ),
        "validation_metrics_at_selected_threshold": (
            val_metrics_chosen
        ),
        "test_set_evaluated": False,
    }

    with (
        output_dir /
        "validation_threshold_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            threshold_summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    checkpoint_sha256 = sha256(
        best_checkpoint_path
    )

    (
        output_dir /
        "best_model_by_val_ap.sha256"
    ).write_text(
        (
            f"{checkpoint_sha256}  "
            f"{best_checkpoint_path.name}\n"
        ),
        encoding="utf-8",
    )

    (
        output_root /
        "latest_run_path.txt"
    ).write_text(
        str(output_dir.resolve()) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 84)
    print("训练和验证集阈值选择完成")
    print("=" * 84)
    print("最佳epoch：", best_epoch)
    print("最佳验证集AP：", f"{best_val_ap:.6f}")
    print(
        "验证集选择的异常概率阈值：",
        f"{chosen_threshold:.6f}",
    )
    print(
        "该阈值下异常召回率：",
        f"{val_metrics_chosen['recall']:.6f}",
    )
    print(
        "该阈值下正常帧特异度：",
        f"{val_metrics_chosen['specificity']:.6f}",
    )
    print(
        "该阈值下异常精确率：",
        f"{val_metrics_chosen['precision']:.6f}",
    )
    print(
        "混淆矩阵：",
        f"TN={val_metrics_chosen['tn']}，"
        f"FP={val_metrics_chosen['fp']}，"
        f"FN={val_metrics_chosen['fn']}，"
        f"TP={val_metrics_chosen['tp']}",
    )
    print("最佳模型：", best_checkpoint_path)
    print(
        "阈值报告：",
        output_dir /
        "validation_threshold_summary.json",
    )
    print("测试集尚未使用。")


if __name__ == "__main__":
    main()
