# Completed private Kaggle GPU study

The free Kaggle run completed successfully on November 9, 2026. Version 1 ran for 1,676.1 seconds (27 minutes 56 seconds) with GPU T4 x2 allocated. The implementation uses one Tesla T4, not distributed training. The student subsequently authorized source publication at https://github.com/50022005/GenerativeAi_2; the Kaggle notebook remains private.

Notebook: https://www.kaggle.com/code/hashimrahim/criterion-gpu-validation-study?scriptVersionId=367198548

The notebook remains private. Its owner can view or download it while signed in.

## What to open

- `output/Criterion_With_Kaggle.zip`: combined delivery, including the original CPU project and verified GPU extension.
- `output/pdf/Criterion_Report.pdf`: original IEEE LaTeX research paper and held-out CPU experiment.
- `output/pdf/Criterion_Kaggle_Study.pdf`: three-page companion explaining the measured GPU extension, limitations and benchmarks.
- `experiments/kaggle_373d2a0208f8/`: imported GPU adapters, all epoch predictions and metrics, training logs, protocol, selection and raw benchmark repeats.
- `output/kaggle/Criterion_GPU.ipynb`: self-contained notebook that reproduces the GPU experiment.
- `output/kaggle/criterion_gpu_results.zip`: exact downloaded experiment archive.

## Results and their limits

The pinned SmolLM2-360M-Instruct target was compared before and after all-layer query/value LoRA (1,638,400 trainable parameters). Two predetermined objectives each trained for three epochs. Selection used generated validation classification macro-F1 and evidence micro-F1, with no test annotations uploaded or evaluated.

| Same 360M target, validation only | Accuracy | Classification macro-F1 | Evidence micro-F1 |
| --- | ---: | ---: | ---: |
| Before adaptation | 0.6602 | 0.2621 | 0.2617 |
| Selected: uniform loss, epoch 3 | 0.6845 | 0.3181 | 0.3529 |

All six checkpoint scores are retained. The balanced objective did not outperform the selected uniform objective. Rare-label reasoning remains weak: the selected model has zero recall for not included, excluded, and not applicable on this validation cohort. These results do not establish clinical reliability.

This extension was requested after the original CPU test had already been evaluated. It is **post-test exploratory work**, not a new blind-test result. The original fixed split, test predictions, frozen CPU code and selected 135M adapter remain unchanged. Do not substitute these validation scores for the main report's test scores.

GPU decoding equality passed for uncached, cached and speculative generation at 32/8, 128/16 and 256/32 input/output tokens, including all three measured repeats after warmup. Cache speedups were 1.03x, 1.02x and 1.43x. Speculative generation used the full pretrained 135M draft; even with 100% proposal acceptance in these prompts, it was slower than cached target-only generation (0.74-0.81x speed). Tiled attention passed six shapes/masks with maximum absolute error 5.36e-7, but was slower than the dense reference. The report explains these negative performance results.

CUDA timing is synchronized. Memory is reset peak tensor allocation/reservation in a shared process, including active models and warmup, not whole-board memory or CPU RSS.

## Using the application without a GPU

The original standalone application remains the CPU-friendly 135M model. Double-click `Start_App.bat` (run `Setup_CPU.bat` first on a new machine). Its three inference modes, trial screening and export remain available without Kaggle. **The app does not use the exploratory 360M GPU adapter.** The larger adapter has different tensor shapes and must not overwrite `models/adapter/adapter.pt`.

The main IEEE report remains the primary assignment paper. The GPU PDF is supplementary research evidence, not a replacement for its original held-out evaluation or application model.

## Reproduction and verification

Keep the notebook private, select the free T4 accelerator, enable Internet and use Save Version > Save & Run All. It retains Kaggle's CUDA PyTorch and installs the other pinned dependencies itself. Do not install the local CPU `requirements.txt` into Kaggle.

From the project environment, local checks are:

```text
python -m pytest tests/test_kaggle_study.py -q
python scripts/build_kaggle_notebook.py
python scripts/import_kaggle_results.py output/kaggle/criterion_gpu_results.zip
```

The importer intentionally refuses to overwrite an already imported experiment. This delivered experiment has already passed it. All seven metric files were independently recomputed against the 206 original validation annotations, selection was checked, and saved decoding/attention correctness was rechecked. The 13 executed source/data files matched the upload manifest exactly. Local verification is recorded in `local_verification.json`.

The companion PDF can be regenerated with `python scripts/report_kaggle.py experiments/kaggle_373d2a0208f8` in an environment with ReportLab 4.4.9. The original paper remains reproducible from `report/main.tex`. The GPU trainer/model loader is in `kaggle_study/` and pins both Hugging Face model revisions.

Downloaded study archive SHA-256: `373d2a0208f81fc385c166730922afaba1573b6f7daa24666042317f9779df34`.

## Remaining student submission steps

Follow `DEMO_GUIDE.md` to record the required five-to-seven-minute application demonstration and prepare for the live defense. Use the GitHub source URL above after verifying the uploaded files are accessible. No video or Classroom submission is claimed. Review `RUBRIC_REVIEW.md`, understand the implementation and AI-use disclosure, and add a real application screenshot before submission.
