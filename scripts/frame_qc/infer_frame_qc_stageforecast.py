#!/usr/bin/env python3

import argparse
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18


ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply a frozen frame-QC model to a StageForecast index."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk-size", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-embryos", type=int)
    parser.add_argument("--expected-train-frames", type=int)
    parser.add_argument("--expected-val-frames", type=int)
    parser.add_argument("--expected-test-frames", type=int)
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
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(table, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(data, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def as_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
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


class FullDataset(Dataset):
    def __init__(self, table):
        self.table = table.reset_index(drop=True)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        path = Path(
            str(self.table.iloc[index]["image_path"])
        )

        try:
            with Image.open(path) as source:
                source.load()
                image = source.convert("RGB")
                image = transform(image)

            return image, index, True, ""

        except Exception as error:
            # 先保留位置，最终禁止带读取错误生成Full-B。
            image = torch.zeros(
                (3, 224, 224),
                dtype=torch.float32,
            )
            message = (
                f"{type(error).__name__}: {error}"
            )
            return image, index, False, message


def main():
    args = parse_args()
    SOURCE = Path(args.source)
    RUN = Path(args.run_dir)
    CHECKPOINT = RUN / "best_model_by_val_ap.pt"
    PROTOCOL = RUN / "frozen_test_protocol_v1.json"
    OUT = Path(args.output_dir)
    PARTS = OUT / "prediction_parts"
    OUT.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)

    SPEC_PATH = OUT / "full_qc_inference_spec_v1.json"
    SUMMARY_PATH = OUT / "full_qc_inference_summary_v1.json"
    ERROR_PATH = OUT / "full_qc_read_errors_v1.csv"
    FULL_A_PATH = OUT / "index_final_FullA_v1.csv"
    FULL_B_PATH = OUT / "index_final_FullB_frozen_qc_v1.csv"
    PREDICTIONS_PATH = OUT / "full_qc_predictions_v1.csv"
    REMOVED_PATH = OUT / "full_qc_removed_frames_v1.csv"
    PER_EMBRYO_PATH = OUT / "full_qc_removal_by_embryo_v1.csv"

    EXPECTED_ROWS = args.expected_rows
    EXPECTED_EMBRYOS = args.expected_embryos
    CHUNK_SIZE = args.chunk_size
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers

    if SUMMARY_PATH.exists():
        raise RuntimeError(
            f"正式汇总已存在，禁止覆盖：{SUMMARY_PATH}"
        )

    protocol = json.loads(
        PROTOCOL.read_text(encoding="utf-8")
    )

    checkpoint_hash = sha256(CHECKPOINT)
    expected_checkpoint_hash = protocol["checkpoint_sha256"]

    if checkpoint_hash != expected_checkpoint_hash:
        raise RuntimeError("checkpoint SHA256不一致")

    threshold = float(
        protocol["frozen_invalid_probability_threshold"]
    )

    source_hash = sha256(SOURCE)
    source = pd.read_csv(SOURCE)

    required = {
        "embryo_id",
        "frame_index",
        "image_path",
        "split",
    }

    missing = required - set(source.columns)
    if missing:
        raise RuntimeError(
            f"StageForecast索引缺少字段：{sorted(missing)}"
        )

    source["embryo_id"] = (
        source["embryo_id"].astype(str).str.strip()
    )
    source["frame_index"] = pd.to_numeric(
        source["frame_index"],
        errors="raise",
    ).astype(int)
    source["split"] = (
        source["split"].astype(str).str.strip().str.lower()
    )
    source["_source_row"] = np.arange(len(source))

    if EXPECTED_ROWS is not None and len(source) != EXPECTED_ROWS:
        raise RuntimeError(
            f"全库帧数为{len(source)}，预期{EXPECTED_ROWS}"
        )

    if (
        EXPECTED_EMBRYOS is not None
        and source["embryo_id"].nunique() != EXPECTED_EMBRYOS
    ):
        raise RuntimeError(
            f"全库胚胎数不是预期的{EXPECTED_EMBRYOS}"
        )

    if source.duplicated(
        ["embryo_id", "frame_index"]
    ).any():
        raise RuntimeError(
            "存在重复的embryo_id + frame_index"
        )

    cross_split = (
        source.groupby("embryo_id")["split"].nunique()
    )

    if cross_split.gt(1).any():
        raise RuntimeError("发现胚胎跨split泄漏")

    split_counts = {
        str(key): int(value)
        for key, value in
        source.groupby("split").size().items()
    }

    expected_split_counts = {
        key: value
        for key, value in {
            "train": args.expected_train_frames,
            "val": args.expected_val_frames,
            "test": args.expected_test_frames,
        }.items()
        if value is not None
    }

    if (
        expected_split_counts
        and any(
            split_counts.get(key) != value
            for key, value in expected_split_counts.items()
        )
    ):
        raise RuntimeError(
            f"split数量异常：{split_counts}"
        )

    specification = {
        "version": "stageforecast_full_qc_inference_v1",
        "source_path": str(SOURCE.resolve()),
        "source_sha256": source_hash,
        "source_rows": int(len(source)),
        "source_embryos": int(
            source["embryo_id"].nunique()
        ),
        "checkpoint_path": str(CHECKPOINT.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "expected_checkpoint_epoch": args.expected_checkpoint_epoch,
        "threshold": threshold,
        "chunk_size": CHUNK_SIZE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "autocast": args.device != "cpu",
        "original_images_modified": False,
    }

    if SPEC_PATH.exists():
        old_specification = json.loads(
            SPEC_PATH.read_text(encoding="utf-8")
        )
        if old_specification != specification:
            raise RuntimeError(
                "已有分块推理规格与本次不一致，禁止混用"
            )
    else:
        atomic_json(specification, SPEC_PATH)

    device = resolve_device(args.device)

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    if (
        args.expected_checkpoint_epoch is not None
        and int(checkpoint["epoch"]) != args.expected_checkpoint_epoch
    ):
        raise RuntimeError(
            f"checkpoint epoch={checkpoint['epoch']}，"
            f"预期={args.expected_checkpoint_epoch}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model = model.to(device)
    model.eval()

    part_paths = []

    for start in range(0, len(source), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(source))

        part_path = (
            PARTS /
            f"predictions_{start:06d}_{end - 1:06d}.csv"
        )
        part_paths.append(part_path)

        expected_rows = np.arange(start, end)

        if part_path.exists():
            existing = pd.read_csv(part_path)

            valid_existing = (
                len(existing) == len(expected_rows)
                and "_source_row" in existing.columns
                and np.array_equal(
                    pd.to_numeric(
                        existing["_source_row"],
                        errors="coerce",
                    ).to_numpy(),
                    expected_rows,
                )
                and "read_ok" in existing.columns
                and as_bool(existing["read_ok"]).all()
            )

            if valid_existing:
                print(
                    f"[跳过] {start}-{end - 1} 已完成",
                    flush=True,
                )
                continue

            print(
                f"[重算] {part_path.name}不完整或含读取错误",
                flush=True,
            )

        chunk = source.iloc[start:end].copy()
        dataset = FullDataset(chunk)

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(NUM_WORKERS > 0),
        )

        probabilities = np.full(
            len(chunk),
            np.nan,
            dtype=np.float64,
        )
        read_ok = np.zeros(
            len(chunk),
            dtype=bool,
        )
        errors = [""] * len(chunk)

        print(
            f"[推理] {start}-{end - 1}，"
            f"共{len(chunk)}帧",
            flush=True,
        )

        with torch.inference_mode():
            for batch_number, (
                images,
                local_indices,
                batch_ok,
                batch_errors,
            ) in enumerate(loader, start=1):

                images = images.to(
                    device,
                    non_blocking=True,
                )

                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=(device.type == "cuda"),
                ):
                    logits = model(images)

                batch_probabilities = (
                    torch.softmax(logits, dim=1)[:, 1]
                    .cpu()
                    .numpy()
                )

                indices = local_indices.numpy()
                ok_values = batch_ok.numpy().astype(bool)

                probabilities[indices] = batch_probabilities
                read_ok[indices] = ok_values

                for local_index, ok, error in zip(
                    indices,
                    ok_values,
                    batch_errors,
                ):
                    if not ok:
                        errors[int(local_index)] = str(error)

                if batch_number % 50 == 0:
                    processed = min(
                        batch_number * BATCH_SIZE,
                        len(chunk),
                    )
                    print(
                        f"  已处理 {processed}/{len(chunk)}",
                        flush=True,
                    )

        probabilities[~read_ok] = np.nan

        predictions = np.full(
            len(chunk),
            -1,
            dtype=np.int8,
        )
        predictions[read_ok] = (
            probabilities[read_ok] >= threshold
        ).astype(np.int8)

        part = pd.DataFrame({
            "_source_row": expected_rows,
            "qc_probability_invalid": probabilities,
            "qc_prediction_invalid": predictions,
            "read_ok": read_ok,
            "read_error": errors,
        })

        atomic_csv(part, part_path)

        print(
            f"[保存] {part_path.name}",
            flush=True,
        )

    parts = []

    for part_path in part_paths:
        if not part_path.exists():
            raise RuntimeError(
                f"缺少推理分块：{part_path}"
            )
        parts.append(pd.read_csv(part_path))

    qc = (
        pd.concat(parts, ignore_index=True)
        .sort_values("_source_row")
        .reset_index(drop=True)
    )

    if len(qc) != len(source):
        raise RuntimeError(
            f"合并后帧数为{len(qc)}，预期{len(source)}"
        )

    if not np.array_equal(
        pd.to_numeric(
            qc["_source_row"],
            errors="raise",
        ).to_numpy(),
        np.arange(len(source)),
    ):
        raise RuntimeError("合并后的行号不连续")

    qc["read_ok"] = as_bool(qc["read_ok"])

    if not qc["read_ok"].all():
        failed = source.loc[
            ~qc["read_ok"].to_numpy()
        ].copy()

        failed["read_error"] = qc.loc[
            ~qc["read_ok"],
            "read_error",
        ].to_numpy()

        atomic_csv(failed, ERROR_PATH)

        raise RuntimeError(
            f"有{len(failed)}帧读取失败；"
            f"已写入：{ERROR_PATH}。"
            "修复路径或图像后重新运行。"
        )

    probabilities = pd.to_numeric(
        qc["qc_probability_invalid"],
        errors="raise",
    ).to_numpy()

    predictions = pd.to_numeric(
        qc["qc_prediction_invalid"],
        errors="raise",
    ).astype(int).to_numpy()

    if not np.isfinite(probabilities).all():
        raise RuntimeError("推理概率中存在NaN或无穷值")

    if not np.isin(predictions, [0, 1]).all():
        raise RuntimeError("预测标签中存在0/1以外的值")

    prediction_table = source.drop(
        columns=["_source_row"]
    ).copy()

    prediction_table["qc_probability_invalid"] = (
        probabilities
    )
    prediction_table["qc_prediction_invalid"] = (
        predictions
    )

    full_b = prediction_table[
        prediction_table["qc_prediction_invalid"].eq(0)
    ].copy()

    removed = prediction_table[
        prediction_table["qc_prediction_invalid"].eq(1)
    ].copy()

    # 下游输入只保留StageForecast原生字段。
    original_columns = [
        column
        for column in source.columns
        if column != "_source_row"
    ]

    full_b_native = full_b[original_columns].copy()

    per_embryo = (
        prediction_table
        .groupby(["split", "embryo_id"], as_index=False)
        .agg(
            frames_FullA=(
                "qc_prediction_invalid",
                "size",
            ),
            removed_frames=(
                "qc_prediction_invalid",
                "sum",
            ),
        )
    )

    per_embryo["frames_FullB"] = (
        per_embryo["frames_FullA"]
        - per_embryo["removed_frames"]
    )

    per_embryo["removal_rate"] = (
        per_embryo["removed_frames"]
        / per_embryo["frames_FullA"]
    )

    lost = per_embryo[
        per_embryo["frames_FullB"].eq(0)
    ].copy()

    by_split = []

    for split_name, group in prediction_table.groupby(
        "split",
        sort=True,
    ):
        group_b = full_b[
            full_b["split"].eq(split_name)
        ]
        group_removed = removed[
            removed["split"].eq(split_name)
        ]

        by_split.append({
            "split": str(split_name),
            "frames_FullA": int(len(group)),
            "frames_FullB": int(len(group_b)),
            "removed_frames": int(len(group_removed)),
            "removal_rate": float(
                len(group_removed) / len(group)
            ),
            "embryos_FullA": int(
                group["embryo_id"].nunique()
            ),
            "embryos_FullB": int(
                group_b["embryo_id"].nunique()
            ),
        })

    # Full-A保存为源索引的逐字节快照。
    temporary_a = FULL_A_PATH.with_suffix(
        FULL_A_PATH.suffix + ".tmp"
    )
    shutil.copyfile(SOURCE, temporary_a)
    temporary_a.replace(FULL_A_PATH)

    atomic_csv(full_b_native, FULL_B_PATH)
    atomic_csv(prediction_table, PREDICTIONS_PATH)
    atomic_csv(removed, REMOVED_PATH)
    atomic_csv(per_embryo, PER_EMBRYO_PATH)

    summary = {
        "definition": {
            "FullA": (
                "StageForecast原始全库索引"
            ),
            "FullB": (
                "FullA经过冻结ResNet18 QC过滤"
            ),
            "authoritative_split": (
                "StageForecast index_final.csv"
            ),
            "paired_design": True,
            "threshold_modified": False,
            "original_images_modified": False,
        },
        "checkpoint_sha256": checkpoint_hash,
        "source_index_sha256": source_hash,
        "threshold": threshold,
        "frames_FullA": int(len(prediction_table)),
        "frames_FullB": int(len(full_b)),
        "removed_frames": int(len(removed)),
        "removal_rate": float(
            len(removed) / len(prediction_table)
        ),
        "embryos_FullA": int(
            prediction_table["embryo_id"].nunique()
        ),
        "embryos_FullB": int(
            full_b["embryo_id"].nunique()
        ),
        "whole_embryos_lost": int(len(lost)),
        "whole_embryos_lost_ids": (
            lost["embryo_id"].astype(str).tolist()
        ),
        "by_split": by_split,
        "probability_summary": {
            "min": float(np.min(probabilities)),
            "median": float(np.median(probabilities)),
            "mean": float(np.mean(probabilities)),
            "max": float(np.max(probabilities)),
        },
        "output_sha256": {
            "FullA": sha256(FULL_A_PATH),
            "FullB": sha256(FULL_B_PATH),
            "predictions": sha256(PREDICTIONS_PATH),
            "removed": sha256(REMOVED_PATH),
            "per_embryo": sha256(PER_EMBRYO_PATH),
        },
    }

    atomic_json(summary, SUMMARY_PATH)

    print("\n" + "=" * 80)
    print("StageForecast全库冻结QC推理完成")
    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n正式输出：")
    for path in [
        FULL_A_PATH,
        FULL_B_PATH,
        PREDICTIONS_PATH,
        REMOVED_PATH,
        PER_EMBRYO_PATH,
        SUMMARY_PATH,
    ]:
        print(path.resolve())


if __name__ == "__main__":
    main()
