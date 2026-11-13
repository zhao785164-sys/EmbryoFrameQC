"""Portable inference wrapper; archived preprocessing and checkpoint stay fixed."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys


MODEL_SHA256 = "54b65cbb9d626051b2b9c6f1827eef215ba4f5b2da48183128ec7f3aafad1e30"
MODULE_HASHES = {
    "qc_image_io_v1.py": "d1e82bc8fa02afd8a09d76ca8fe53835e0eefe114d322a16adb2d2b27a9d643b",
    "qc_preprocess_v1.py": "7b30626856059d918f43aee2bdfed653dd7c02c34ab36a708c2d3d23793e609f",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def open_csv(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    return opener(path, "rt", encoding="utf-8-sig", newline="")


def image_path(value, image_root, old_prefix):
    p = Path(value)
    if old_prefix:
        if image_root is None:
            raise ValueError("--old-image-prefix requires --image-root")
        p = p.relative_to(Path(old_prefix))
    elif p.is_absolute():
        return p
    if image_root is None:
        raise ValueError("Relative image paths require --image-root")
    return Path(image_root) / p


def probability_threshold(value):
    x = float(value)
    if not math.isfinite(x) or not 0 <= x <= 1:
        raise argparse.ArgumentTypeError("Threshold must be finite and in [0, 1]")
    return x


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--threshold", type=probability_threshold, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-root", type=Path)
    p.add_argument("--old-image-prefix")
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("--batch-size must be positive")
    if args.output_dir.exists():
        p.error("Use a new --output-dir; existing results are never overwritten")
    with open_csv(args.index) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        if not fields or not {"embryo_id", "run", "image_path"}.issubset(fields):
            p.error("Index requires embryo_id, run, image_path; split is optional")
        if len(fields) != len(set(fields)) or any(k.startswith("qc_") for k in fields):
            p.error("Duplicate headers or existing qc_* columns are not supported")
        rows = list(reader)
    if not rows:
        p.error("Empty index")
    seen = set()
    resolved = []
    for i, row in enumerate(rows, 2):
        if None in row or any(v is None for v in row.values()):
            p.error("Malformed CSV at row %d" % i)
        try:
            run = int(row["run"])
            key = (row["embryo_id"], run)
            if not row["embryo_id"] or run < 0 or key in seen:
                raise ValueError("empty embryo_id, negative RUN or duplicate frame")
            seen.add(key)
            if not row["image_path"]:
                raise ValueError("Empty image_path")
            resolved.append(image_path(row["image_path"], args.image_root, args.old_image_prefix))
        except (ValueError, TypeError) as e:
            p.error("Index row %d: %s" % (i, e))
    if sha256(args.checkpoint) != MODEL_SHA256:
        p.error("Checkpoint SHA256 differs from the archived epoch-6 model")
    here = Path(__file__).resolve().parent
    for name, expected in MODULE_HASHES.items():
        if sha256(here / name) != expected:
            p.error("Archived preprocessing changed: " + name)

    # Keep imports lazy so --help and index/path checks need no ML environment.
    import torch
    import torchvision
    import PIL
    import numpy as np
    from qc_image_io_v1 import load_qc_rgb
    from qc_preprocess_v1 import prepare_qc

    torch.set_num_threads(2)
    # Hash-checked, self-trained checkpoint. Matches the original loader policy.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("epoch") != 6:
        raise RuntimeError("Expected archived epoch-6 checkpoint dictionary")
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(args.device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "predictions.csv"
    extra = ["qc_resolved_image_path", "qc_status", "qc_decode_mode",
             "qc_score_delete", "qc_candidate_action", "qc_threshold", "qc_error"]
    counts = {"KEEP": 0, "DELETE": 0, "REVIEW": 0}
    with output.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields + extra)
        writer.writeheader()
        for start in range(0, len(rows), args.batch_size):
            results, tensors, positions = [], [], []
            for j in range(start, min(start + args.batch_size, len(rows))):
                r = dict(rows[j], qc_resolved_image_path=str(resolved[j]),
                         qc_status="", qc_decode_mode="", qc_score_delete="",
                         qc_candidate_action="REVIEW", qc_threshold=args.threshold, qc_error="")
                results.append(r)
                try:
                    rgb, meta = load_qc_rgb(resolved[j])
                except Exception as exc:
                    r.update(qc_status="DECODE_ERROR", qc_error=repr(exc))
                    continue
                try:
                    with rgb:
                        r["qc_decode_mode"] = meta["decode_mode"]
                        tensor, display, _ = prepare_qc(rgb)
                        display.close()
                    if not torch.isfinite(tensor).all().item():
                        raise ValueError("Nonfinite preprocessing tensor")
                except Exception as exc:
                    r.update(qc_status="PREPROCESS_ERROR", qc_error=repr(exc))
                    continue
                tensors.append(tensor)
                positions.append(len(results) - 1)
            if tensors:
                with torch.inference_mode():
                    scores = model(torch.stack(tensors).to(args.device)).softmax(dim=1)[:, 1].cpu().tolist()
                if not all(math.isfinite(s) and 0 <= s <= 1 for s in scores):
                    raise RuntimeError("Nonfinite or out-of-range model score")
                for pos, score in zip(positions, scores):
                    results[pos].update(qc_status="OK", qc_score_delete=score,
                                        qc_candidate_action="DELETE" if score >= args.threshold else "KEEP")
            for r in results:
                counts[r["qc_candidate_action"]] += 1
            writer.writerows(results)
    record = {
        "status": "INFERENCE_COMPLETE", "scope": "F0 candidate scoring only; no images deleted",
        "deployment_approved": False, "manual_overrides_applied": False,
        "rows": len(rows), "embryos": len({r["embryo_id"] for r in rows}), "counts": counts,
        "threshold": args.threshold, "decision_rule": "score_delete >= threshold => DELETE",
        "checkpoint_epoch": 6, "checkpoint_sha256": MODEL_SHA256,
        "index_sha256": sha256(args.index), "predictions_sha256": sha256(output),
        "image_root": str(args.image_root) if args.image_root else None,
        "old_image_prefix": args.old_image_prefix, "device": args.device,
        "versions": {"python": sys.version, "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "pillow": PIL.__version__, "numpy": np.__version__},
    }
    (args.output_dir / "run_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
