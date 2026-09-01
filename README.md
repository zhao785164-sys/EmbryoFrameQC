# Frame-level QC preprocessing

This directory contains the reproducible model-facing part of the frame-QC
workflow used before embryo-stage prediction:

1. train a binary ResNet18 classifier (`VALID=0`, `INVALID=1`);
2. freeze a validation-derived decision threshold and evaluate once on the
   internal test split;
3. evaluate the same checkpoint and threshold on a separately frozen external
   labelled set;
4. apply the frozen model to a StageForecast index and write a paired,
   row-preserving Full-A/Full-B comparison.

The scripts never modify source images. They write new CSV/JSON/model artifacts
under an explicit output directory and refuse to overwrite a completed formal
summary.

## Scientific scope and current limitation

This is an experimental *central-focal-plane* QC component, not a finalized
multi-focus QC system. Its intended target is a frame with insufficient visible
embryo information (for example an empty well, a severe acquisition failure, or
major corruption). A frame should not be rejected merely because the embryo is
off-centre, dim, late-stage, or morphologically unusual when it remains usable
for staging.

The current classifier was developed from a relatively small manually labelled
set. Full-dataset audit revealed domain-shift false positives, so its output must
be treated as a candidate exclusion mask and reviewed before biological or
clinical interpretation. Do not silently delete images or report the filtered
cohort without the threshold, checkpoint hash, removal counts by split/embryo,
and a false-positive audit.

This directory covers training, frozen evaluation and inference. It does **not**
include the upstream manual-label construction or montage-review utilities.

## Environment

Python dependencies are listed in `requirements.txt`. For an exact reproduction,
record the Python/CUDA/PyTorch versions and preserve the generated configuration,
protocol, SHA256 and summary files with each run.

## Required CSV columns

Training/internal evaluation index:

- `embryo_id`
- `run`
- `split` (`train`, `val`, `test`; split at embryo level)
- `is_frame_valid`
- `image_path`

Full-dataset inference index:

- `embryo_id`
- `frame_index`
- `split`
- `image_path`

External labels additionally use `target_invalid` and `model_input_path`.

## Example commands

Train using the counts from the audited 30-embryo development set:

```bash
python scripts/frame_qc/train_frame_qc_resnet18.py \
  --index /path/to/frame_qc_supervised_index.csv \
  --output-root outputs/frame_qc/runs \
  --device cuda:0 \
  --expected-frames 14380 \
  --expected-embryos 30 \
  --expected-valid 11965 \
  --expected-invalid 2415 \
  --expected-train-embryos 21 \
  --expected-val-embryos 4 \
  --expected-test-embryos 5
```

Run the frozen internal test exactly once:

```bash
python scripts/frame_qc/evaluate_frame_qc_internal.py \
  --run-dir outputs/frame_qc/runs/<run_name> \
  --device cuda:0 \
  --expected-checkpoint-epoch 3
```

Run frozen external evaluation:

```bash
python scripts/frame_qc/evaluate_frame_qc_external.py \
  --labels /path/to/external_test_frame_labels.csv \
  --ground-truth-protocol /path/to/ground_truth_protocol.json \
  --ground-truth-bundle /path/to/ground_truth_bundle.sha256 \
  --run-dir outputs/frame_qc/runs/<run_name> \
  --output-dir outputs/frame_qc/external_evaluation \
  --threshold 0.391845703125 \
  --device cuda:0 \
  --expected-frames 7501 \
  --expected-embryos 15 \
  --expected-valid 6670 \
  --expected-invalid 831 \
  --expected-checkpoint-epoch 3
```

Apply the frozen QC model to the 704-embryo StageForecast index:

```bash
python scripts/frame_qc/infer_frame_qc_stageforecast.py \
  --source /path/to/index_final.csv \
  --run-dir outputs/frame_qc/runs/<run_name> \
  --output-dir outputs/frame_qc/full_inference \
  --device cuda:0 \
  --expected-rows 297420 \
  --expected-embryos 704 \
  --expected-train-frames 207130 \
  --expected-val-frames 45581 \
  --expected-test-frames 44709 \
  --expected-checkpoint-epoch 3
```

Omit the optional `--expected-*` checks for a different dataset. For a formal
reproduction, provide them so a wrong index fails early. Part files make the
full inference resumable, but an existing final summary prevents an accidental
overwrite.

## Reporting checklist

- state that embryo-level splitting was used;
- report the frozen checkpoint SHA256 and threshold;
- report invalid-frame recall and valid-frame false-positive rate;
- report removal counts by split and embryo, including whole embryos lost;
- manually review high-removal and long-contiguous-removal cases;
- compare downstream results using paired Full-A/Full-B inputs;
- retain unfiltered Full-A as the authoritative source index.
