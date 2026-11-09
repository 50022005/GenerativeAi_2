# Criterion — CPU clinical trial eligibility lab

**Muhammad Hashim Rahim · i230028 · FAST NUCES · Generative AI**

A locally executable SmolLM2-135M system for the TrialGPT criterion benchmark, with supervised low-rank adaptation, evidence sentence IDs, explicit cached/uncached generation, greedy speculative decoding, educational tiled attention, and a standalone browser application.

**Source repository:** https://github.com/50022005/GenerativeAi_2

The original CPU application and held-out study are accompanied by a completed private Kaggle GPU extension. See `KAGGLE_RESULTS.md` for its validation-only results and `RUBRIC_REVIEW.md` for the submission audit. The app continues to use the original CPU-friendly 135M adapter.

**Educational screening prototype. Not a clinical enrollment decision.** Predictions and citations can be wrong. The aggregate message is an explicit rule over supplied criteria, not a ground-truth-labeled trial outcome.

## Recorded study

| Held-out method | Accuracy | Macro-F1 | Evidence micro-F1 |
|---|---:|---:|---:|
| Unadapted target | 23.3% | 0.155 | 0.176 |
| Selected adapted target | 61.2% | 0.259 | 0.145 |
| Training-majority label reference | 62.6% | 0.226 | Not attempted |

The adapted model predicts `not excluded` for all 131 exclusion test records and cites no sentence in 181/206 test records. These are major measured failures, not evidence of clinical reliability. The report discusses them explicitly. The implementation and inference checks pass, but model quality remains limited.

All 21 current correctness checks pass (16 original checks plus five GPU-extension checks). Cached and speculative generation agree token-for-token with ordinary greedy decoding at all three measured length pairs. In the original CPU study, caching is 2.02–8.28 times as fast as the uncached reference; speculation is slower. Maximum tiled-attention error is below 0.000001.

See `FINAL_STATUS.md` for remaining submission steps. Final browser capture was blocked by the computer-use tool's URL-confidence check; an actual screenshot and the student's recorded demonstration still need to be added. The final trained HTTP backend, export and live comparison have independent verification records in `results/application_check/`.

## Open the application

On the prepared assignment computer, double-click **Start_App.bat**. It opens [the local application](http://127.0.0.1:7860). Keep the launcher window open while using the application. Close that window to stop the server.

On a fresh Windows computer:

1. Install Python 3.11 or newer (the recorded experiment used 3.13.5).
2. Download/clone this repository into a writable folder.
3. Double-click **Setup_CPU.bat** once. It creates an environment, installs CPU dependencies, downloads the public base model and required dataset, and prepares the split.
4. Double-click **Start_App.bat**. No IDE or notebook is needed to use the application.

The first setup needs internet access and approximately 1–3 GB of disk space depending on already installed dependencies. Model execution is local after setup. The base weight file is approximately 269 MB on disk and is loaded as float32 for the recorded CPU experiments. The trained adapter is small and included in `models/adapter`; the larger base weights remain hosted on Hugging Face. Four logical CPU threads and approximately 8 GB of RAM were available; PyTorch uses two threads by default. Training is much slower than inference and is not needed to use the supplied adapter.

On macOS/Linux, create a Python environment, install CPU-compatible PyTorch and the pinned requirements, run `python scripts/download.py`, `python scripts/prepare.py`, then `python app.py`. Browser operation is the same.

## Screening workflow

- Click **Load example**, or enter an unseen de-identified note. Existing `0.`, `1.`, … sentence IDs are preserved.
- Enter one criterion per line: `inclusion: Age 18 to 65 years` or `exclusion: Severe renal impairment`. A training-set trial metadata selector is also provided. It is not a live trial registry.
- Choose cached, uncached or speculative greedy inference, then screen the criteria.
- Inspect labels, exact cited source sentences, missing-information/review flags and truncation information.
- Use **Export readable report** to download an escaped HTML report. Open it in a browser and print/save as PDF if desired.
- **Inference lab** runs a live raw-generation comparison on the current record and first criterion. It uses one warmup and two timed repetitions per mode, a 16-token output budget, and reports exact token agreement, timings and draft acceptance.
- **Study results** exposes saved evaluation metrics and repeated benchmark records.

Patient input stays in the local server's memory. Up to 20 recent jobs are retained until the process stops or old completed jobs expire. The application has no clinical-data authentication or deployment controls. Do not expose its loopback server to the internet.

## Reproduce the experiments

Use a fresh study directory or run `python scripts/new_study.py` to archive the existing results and adapter before a new experiment. This preserves the original recorded study. The common split and base-model revision remain fixed.

```text
python scripts/download.py
python scripts/prepare.py
python -m pytest -q
python scripts/train.py --epochs 3
python scripts/evaluate.py --split validation --checkpoint base
python scripts/evaluate.py --split validation --checkpoint adapted
python scripts/freeze.py
python scripts/evaluate.py --split test --checkpoint base --final-test
python scripts/evaluate.py --split test --checkpoint adapted --final-test
python scripts/benchmark.py
python scripts/stress_cases.py
python scripts/analyze.py
```

Run these experiments sequentially on low-memory CPUs. The final benchmark should run while training, evaluation and app inference are idle. Training selects the minimum token-weighted validation loss across three predetermined epochs. Do not adjust hyperparameters, prompts or thresholds using test results. Evaluation refuses to overwrite existing predictions. `freeze.py` pins the selected adapter and inference/evaluation source hashes before testing. Test evaluation refuses changed source or weights. An interrupted partial test run must be retained and documented; it must not be treated as a complete evaluation.

For a quick development smoke check **in a separate directory**, use `python scripts/train.py --epochs 1 --limit 4`. Such a checkpoint is not the submitted final adapter. The supplied final configuration records 603 training examples and all 206 validation examples. Development and interrupted runs are excluded from final tables.

`TRIALMATCH_THREADS` sets the number of CPU threads. Use the recorded default of two to reproduce timings. Training uses float32, eager attention, batch size one, accumulation eight, rank eight, alpha 16, learning rate 0.001, AdamW default weight decay 0.01/betas 0.9/0.999/epsilon 1e-8, gradient clipping 1, and seed 42. Only query/value adapters in the last six blocks train. The first 24 frozen blocks still execute. This is a compute/memory compromise rather than a claim of optimal adaptation.

## Data and output contract

The sole supervised source is [ncbi/TrialGPT-Criterion-Annotations](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations). The raw parquet is SHA-256 verified. The split sorts 53 unique patient IDs, shuffles them with NumPy `default_rng(42)` and takes 37/8/8 patients, yielding **603/206/206** annotations. `data/split_manifest.json` contains all patient IDs and counts. All annotations for each patient belong to one split.

Preprocessing whitelists the note, criterion type/text, expert label and evidence, plus audit identifiers. GPT predictions/explanations/evidence, correctness annotations, and the dataset's old `training` flag are excluded from model features. Patient/trial IDs, trial title, labels and expert evidence are not inserted in prompts. Expert annotations are only supervision/evaluation targets.

Label codes are A=included, B=not included, C=excluded, D=not excluded, E=not enough information, F=not applicable. Inclusion permits A/B/E/F; exclusion permits C/D/E/F. The model emits one code token followed by one `0`/`1` token per retained sentence, e.g. ` A1001`. The parser returns a structured label and original IDs. This bitmap permits every subset, including no evidence, without limiting the number of cited sentences. A deterministic token mask enforces syntax identically before and after tuning; it does not decide clinical labels.

The input+response limit is 1,024 tokens. Complete leading sentences are kept; whole trailing sentences are omitted as necessary without consulting gold evidence. Omitted IDs are shown, and omitted gold evidence remains a false negative in evaluation. The configured model context is 8,192 tokens; the smaller experiment limit is a CPU budget choice. Oversized criteria or a record with no complete sentence fitting the budget cause a visible validation error.

Explanations use templates and verbatim model-selected text. They do not independently verify entailment or produce a clinician-quality rationale. Empty evidence never automatically changes a prediction into eligible/ineligible. It triggers human review. Overall aggregation checks conflicts first, then insufficient information/applicability, then no predicted conflict; it covers supplied criteria only.

## Implementations and verification

| Component | Implementation | Evidence |
|---|---|---|
| Data integrity | `trialmatch/data.py` | Split manifest, source checksum, patient-overlap and whitelist checks |
| Fine-tuning | `scripts/train.py`, `trialmatch/model.py` | Response-only CE, custom LoRA, configs, epoch logs, selected adapter |
| Eligibility/evidence metrics | `trialmatch/metrics.py`, `scripts/evaluate.py` | Overall and per-type accuracy/macro-F1, per-class P/R/F1, confusion matrices, evidence micro/macro P/R/F1 |
| Cache decoding | `trialmatch/decoding.py` | Full-prefix reference vs explicit dynamic KV reuse; absolute positions, full attention masks, cache-length traces |
| Tiled attention | `trialmatch/attention.py` | Stable dense reference; blocked running maxima/normalizers/value accumulator; causal and offset tests |
| Speculation | `trialmatch/decoding.py` | Drafting, block verification, matching prefix, rejection correction, cache cropping/synchronization, bonus token |
| Browser application | `app.py`, `web/` | Real model screening, evidence inspection, export, live inference benchmark |
| Report | `report/main.tex`, `scripts/analyze.py` | IEEE source, measured tables/graphs and error cases |

The target is [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), pinned at revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`. The alternative was [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct). Both are Apache-2.0. The target has 30 blocks, hidden size 576, 9 query/3 KV heads of width 64, 49,152 vocabulary entries and configured 8,192-token context. The alternative has 32 blocks, width 960 and 15 query/5 KV heads. The 135M model was selected to fit this machine and keep full CPU evaluation feasible.

The draft reuses the base embeddings, first six decoder blocks and final normalization, with the exact same tokenizer. It is an **uncalibrated layer-pruned model**, not a separately pretrained off-the-shelf small model. It may accept very few tokens and run more slowly. Its weakness is disclosed and measured. Speculative tests compare outputs to the target, not to the draft. Both caches are cropped after rejection and advanced through correction/bonus tokens. There is no call to an assisted-generation API.

The tiled attention implementation is an independent educational experiment; the target uses Transformers eager attention. It does not invoke FlashAttention or claim a fused GPU kernel. Final benchmark memory is fresh-subprocess peak process RSS including interpreter, weights, warmup and tensor buffers. Dense/tile score-buffer estimates are separate mathematical values. `tracemalloc` is not used to claim tensor memory. Timings include warmup exclusion and three repeated runs. CPU work is synchronous. Speculative prefill includes both target and draft; its comparison baseline is ordinary cached greedy decoding.

Evidence scoring uses original-ID sets. Both-empty gold/prediction sets score P/R/F1=1 per record and contribute no pooled counts; exactly one empty set scores zero. Micro scores pool counts. Classification macro-F1 averages over the complete applicable label list, using zero for undefined classes. Tables can therefore include zero-support categories.

## Report, artifacts and submission

- `report/main.tex`: IEEE LaTeX source, with author details and AI-use appendix.
- `output/pdf/Criterion_Report.pdf`: final compiled report.
- `models/adapter/adapter.pt` and `config.json`: selected adapter and base revision/configuration. Base weights download through `scripts/download.py`; no paid service is needed.
- `data/split_manifest.json`: fixed patient IDs and counts.
- `results/`: raw predictions, metrics, training logs, frozen protocol, benchmarks, measured figures and error analysis.
- `DEMO_GUIDE.md`: a timed six-minute student-recording plan and defense preparation.
- `Build_Report.bat`: rebuild the paper after saving an actual app screenshot as `report/figures/application.png` or `.jpg`.
- `FINAL_STATUS.md`: completed work and the remaining browser screenshot/video/submission steps.
- `SUBMISSION_CHECKLIST.md`: final items to verify before uploading to Classroom.

Compile with a standard LaTeX installation: run `pdflatex main.tex` twice inside `report/` (all bibliography entries are embedded). Alternatively, use `tectonic report/main.tex`; first use downloads its free standard TeX packages. The resulting PDF can be copied to `output/pdf/Criterion_Report.pdf`. Re-run `scripts/analyze.py` before compilation only after all final experiments finish. The analysis script refuses incomplete required inputs rather than inventing results.

The recorded application video must show **your own explanation and actual execution**, last five to seven minutes, and have a working viewable link. No video upload or Classroom submission is implied by the source code. Attend the live demo and be prepared to modify code or evaluate an unseen record.

## Research sources and acknowledgments

1. Jin et al., [Matching patients to clinical trials with large language models](https://www.nature.com/articles/s41467-024-53081-z), Nature Communications, 2024. Criterion-level labels and evidence localization inform the task; its performance is not reused as ours.
2. Hugging Face, [SmolLM2 model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct). A compact decoder supports local experimentation, with limited clinical reasoning capacity.
3. Hu et al., [LoRA](https://arxiv.org/abs/2106.09685). Low-rank updates reduce trainable/optimizer memory; forward computation and activations still cost resources.
4. Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135). Tiling and online normalization inspire our educational mathematical implementation, not its accelerator kernel.
5. Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192). Our deterministic specialization verifies greedy equality and reports failures to accelerate.
6. Hugging Face, [Cache explanation](https://huggingface.co/docs/transformers/cache_explanation). Cache length, masks and positions inform our explicit loops.

OpenAI Codex assisted with research, code, debugging, tests, interface design and report drafting. Transformers supplies the pretrained model/cache representation; PyTorch supplies tensor operations and differentiation. The student must understand, review and defend all submitted work. The TrialGPT repository was not run as the submitted implementation. Third-party model/data licenses remain applicable.
