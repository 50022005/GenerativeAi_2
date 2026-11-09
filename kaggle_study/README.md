# Private Kaggle GPU extension

Completed run and measured results: see `KAGGLE_RESULTS.md`. Version 1 completed successfully; the downloaded outputs were locally verified and retained in `experiments/kaggle_373d2a0208f8/`.

This is a separate exploratory study requested after the original CPU experiment was completed. The original `results/`, adapter and fixed split are preserved. The student subsequently authorized GitHub publication of the project; the Kaggle notebook remains private.

The upload-ready file is `output/kaggle/Criterion_GPU.ipynb`. It embeds only the source modules, 603 training rows, 206 validation rows and split manifest. It excludes the report/student ID, test annotations, prior test predictions, credentials and existing weights. Keep the Kaggle notebook private.

Use a free GPU accelerator and enable Internet for public model/package downloads. The notebook retains Kaggle's CUDA-enabled PyTorch. It deliberately fails if a GPU is unavailable instead of silently starting another long CPU run.

The pinned target is SmolLM2-360M-Instruct. The smaller draft is the actual pretrained SmolLM2-135M-Instruct, with full vocabulary/special-token compatibility checked. Rank-16 query/value LoRA trains in every target layer. Two predetermined objectives are compared: ordinary response CE and a task-balanced objective with training-only label-frequency and evidence-positive weights. Each uses three epochs, seed 42, learning rate 0.0005 and gradient accumulation eight.

Checkpoint/configuration selection maximizes the mean of validation classification macro-F1 and evidence micro-F1. Selection is based on actual generated predictions, not teacher-forced loss alone. Baseline validation predictions and every epoch's validation results are retained.

The original fixed test has already been evaluated, so this extension is explicitly disclosed as post-test exploratory work. This notebook does not include or evaluate test annotations and does not claim a new blind test. Do not replace the original held-out table with validation metrics.

After selection, the notebook runs synchronized CUDA decoding and tiled-attention checks and exports `criterion_gpu_results.zip`. GPU memory numbers are reset peak tensor allocations/reservations in a shared process; they are not CPU RSS or whole-board consumption. Existing frozen decoding source is embedded with synchronization instrumentation; local original source stays unchanged.

Rebuild after code changes: `python scripts/build_kaggle_notebook.py`. Local checks: `python -m pytest tests/test_kaggle_study.py -q`. Those CPU checks verify the new objective, LoRA identity/frozen base, upload whitelist and generated decoding logic. They are not proof that a GPU run has occurred.

Do not copy this 360M adapter over the original 135M adapter. They have different tensor shapes. Import a successful result into a separate experiment directory and use a matching 360M loader.
