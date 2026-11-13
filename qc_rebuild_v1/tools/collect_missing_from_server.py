"""Export missing weights and CSVs from the original server, without running QC."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import zipfile


BATCH = Path("runs/qc_f0_batch_VuRe5ghM")
TEST = Path("runs/qc_independent_test_seed42_20261105_170047_313701")
MODEL_SHA = "54b65cbb9d626051b2b9c6f1827eef215ba4f5b2da48183128ec7f3aafad1e30"
F0_SHA = "92548fbaf23a0b54e4081bdbe487a2d29b7fed4535180a43886c7494f8d5b85a"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=Path("/home/zhl/projects/QC_Rebuild_v1"))
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    root = args.project_root.resolve()
    output = args.output or root.parent / ("QC_rebuild_supplement_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".zip")
    if output.exists():
        p.error("Output already exists; choose a new filename")
    if not (root / BATCH).is_dir():
        p.error("F0 batch directory missing")
    files, missing = {}, []

    def add(rel, expected=None):
        path = root / rel
        if not path.is_file() or path.is_symlink():
            missing.append(str(rel))
            return
        digest = sha256(path)
        if expected and digest != expected:
            raise SystemExit("SHA256 mismatch; export stopped: " + str(rel))
        files[path] = dict(relative_path=str(rel), bytes=path.stat().st_size, sha256=digest)

    add(BATCH / "model.pt", MODEL_SHA)
    add(BATCH / "f0_manifest.csv", F0_SHA)
    for name in ["model_protocol.json", "batch_summary.json", "per_embryo_summary.csv"]:
        add(BATCH / name)

    predictions = sorted((root / BATCH / "per_embryo").glob("*.csv"))
    n_rows = 0
    for path in predictions:
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not {"embryo_id", "run", "score_delete", "predicted_action", "status"}.issubset(reader.fieldnames or []):
                raise SystemExit("Unexpected prediction schema: " + str(path))
            n_rows += sum(1 for _ in reader)
        add(path.relative_to(root))
    if len(predictions) != 704 or n_rows != 342363:
        missing.append("Complete per_embryo scores: expected 704 CSVs / 342363 rows; found %d / %d" % (len(predictions), n_rows))

    checksums = root / TEST / "result_checksums_20261106_090830_214352.json"
    expectations = json.loads(checksums.read_text(encoding="utf-8"))["files_sha256"] if checksums.is_file() else {}
    for name in ["test_manifest.csv", "test_predictions.csv", "per_embryo_metrics.csv",
                 "manual_boundaries.csv", "manual_segments.csv", "test_selection.csv",
                 "supplement_review_index.csv", "test_protocol.json", "test_metrics.json",
                 "evaluation_record.json"]:
        add(TEST / name, expectations.get(name))

    # Small training/label inputs excluded by the original packaging script.
    for folder in ["data/qc30_decode_audit_v1_20261023_162958_452370", "data/qc30_labels_v1_20261023_155615_176150"]:
        base = root / folder
        if not base.is_dir():
            missing.append(folder)
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and not path.is_symlink() and (path.suffix in {".csv", ".json", ".py", ".txt", ".sha256"} or path.name == "SHA256SUMS"):
                add(path.relative_to(root))

    report = dict(status="COMPLETE_REQUESTED_FILES" if not missing else "PARTIAL_EXPORT",
                  source_project=str(root), expected_model_sha256=MODEL_SHA,
                  per_embryo_files=len(predictions), per_embryo_rows=n_rows,
                  files=list(files.values()), missing=missing,
                  raw_images_included=False, final_deployment_threshold_selected=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, item in files.items():
            z.write(src, "qc_rebuild_supplement/" + item["relative_path"])
        z.writestr("qc_rebuild_supplement/SUPPLEMENT_REPORT.json", json.dumps(report, ensure_ascii=False, indent=2))
    print("Archive:", output)
    print("Size: %.2f MiB" % (output.stat().st_size / 1024 / 1024))
    print("Files:", len(files), "Status:", report["status"])
    print("Missing:", json.dumps(missing, ensure_ascii=False, indent=2))
    print("Original files unchanged; no inference or training executed.")


if __name__ == "__main__":
    main()
