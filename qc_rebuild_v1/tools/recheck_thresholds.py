"""Recompute observed errors on supplied reviewed frames; no extrapolation."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path


def compute(rows, threshold, score_column, label_column):
    counts = dict(true_keep_retained=0, false_delete=0,
                  false_keep=0, true_delete_detected=0, unreviewed=0, unscored=0)
    for r in rows:
        label = r[label_column].strip().upper()
        if label not in {"KEEP", "DELETE"}:
            counts["unreviewed"] += 1
            continue
        try:
            score = float(r[score_column])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("nonfinite/out-of-range score")
        except (ValueError, TypeError):
            counts["unscored"] += 1
            continue
        predicted = "DELETE" if score >= threshold else "KEEP"
        k = {("KEEP", "KEEP"): "true_keep_retained", ("KEEP", "DELETE"): "false_delete",
             ("DELETE", "KEEP"): "false_keep", ("DELETE", "DELETE"): "true_delete_detected"}
        counts[k[label, predicted]] += 1
    tn, fp, fn, tp = [counts[k] for k in ["true_keep_retained", "false_delete", "false_keep", "true_delete_detected"]]
    def ratio(n, d):
        return n / d if d else ""
    return dict(threshold=threshold, **counts, actual_KEEP=tn+fp, actual_DELETE=tp+fn,
                false_delete_rate=ratio(fp, tn+fp), anomaly_miss_rate=ratio(fn, tp+fn),
                kept_set_error_rate=ratio(fn, tn+fn), deleted_set_error_rate=ratio(fp, tp+fp))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--score-column", default="score_delete")
    p.add_argument("--label-column", default="review_action")
    p.add_argument("--thresholds", type=float, nargs="+", default=[i/1000 for i in range(10, 21)])
    args = p.parse_args()
    if any(not math.isfinite(t) or not 0 <= t <= 1 for t in args.thresholds):
        p.error("Thresholds must be finite and in [0, 1]")
    opener = gzip.open if args.labels.suffix == ".gz" else open
    with opener(args.labels, "rt", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"embryo_id", "run", args.score_column, args.label_column}
        if not required.issubset(reader.fieldnames or []):
            p.error("Missing columns: " + str(required - set(reader.fieldnames or [])))
        rows = list(reader)
    if not rows:
        p.error("Empty label table")
    keys = [(r["embryo_id"], int(r["run"])) for r in rows]
    if len(keys) != len(set(keys)):
        p.error("Duplicate embryo_id/RUN pairs; resolve review precedence explicitly first")
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        p.error("Malformed CSV row")
    result = [compute(rows, t, args.score_column, args.label_column) for t in args.thresholds]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result[0]))
        w.writeheader()
        w.writerows(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("Observed reviewed-frame metrics only; not whole-library or independent-test estimates.")


if __name__ == "__main__":
    main()
