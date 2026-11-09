# Assignment rubric review

Reviewed November 9, 2026 against the supplied Assignment 2 PDF. This is an evidence audit, not a predicted grade or an instructor sign-off.

**Status: substantial technical implementation, but not yet a fully complete submission.** Source publication is authorized. A real application screenshot in the IEEE paper, the student's recorded video/link, and live defense remain outstanding. Model performance is a material weakness, even though its failures are documented honestly.

The final pre-publication test run passed all 21 checks in 27.39 seconds. This verifies implementation checks, not clinical quality or the student's defense.

| Rubric area | Existing evidence | Remaining limitation or action |
| --- | --- | --- |
| A: Research and architecture | IEEE related work, at least five credible sources, 135M/360M model comparison, licensing/configurations and CPU feasibility | Student must understand and defend these choices. |
| B: Dataset and adaptation | Prescribed TrialGPT source; fixed 37/8/8 patient split; whitelist/leakage checks; executable LoRA trainer, adapter, logs; same-model before/after validation and held-out CPU results | GPU extension is explicitly exploratory after the CPU test; its validation scores cannot replace the original test table. |
| C: Eligibility and evidence | Full label categories; overall/per-type classification metrics, per-class scores, confusion matrices, evidence micro/macro metrics, empty-set policy, eight error cases and missing-information probes | Held-out adapted evidence F1 is 0.145; all 131 exclusion records predict not excluded; 181/206 predictions cite no sentence. This is weak semantic/evidence quality and can reduce marks. |
| D: KV cache | Explicit forward/cache/position/mask handling, uncached reference, token-equality checks at three lengths; warmup/repeated latency and memory measurements | Correctness evidence exists; student must explain cache growth and numerical differences. |
| E: Tiled attention | Dense reference, tiled running-max/normalizer/value accumulation, square/rectangular/offset masks and error/runtime/memory checks | Educational PyTorch tiling is slower; it is tested separately and does not replace model attention. No fused-kernel speed claim is made. |
| F: Speculative decoding | Implemented drafting/verification/acceptance/rejection/cropping/bonus-token flow, cache invariants, equality tests, acceptance/latency/throughput/memory results | CPU draft is an explicitly disclosed uncalibrated pruned model. GPU study uses a full pretrained 135M draft with the 360M target. Both measured speculation slower than caching. |
| G: Browser application | Standalone Flask app, double-click launch, real model screening, criteria entry/selection, exact source citations, review flags, export, live inference comparison and input validation; actual HTTP backend verification | Final visible browser workflow still needs verification and capture. The app uses the original CPU 135M model, not the exploratory GPU adapter. Explanation is a template plus selected quotes, not an independent entailment check. |
| H: Paper and defense | IEEE LaTeX source/PDF, figures/tables/metrics/limitations/error analysis, AI-use appendix; reproducible source and adapters; supplementary GPU report | Add a real app screenshot to the IEEE report. Record and share a 5-7 minute video. Test source/model/video accessibility, submit through Classroom and attend the live demo. |

## Required remaining steps

1. Launch `Start_App.bat`, verify the actual browser screening, evidence, live comparison and export workflow.
2. Save a genuine screenshot as `report/figures/application.png` or `.jpg`, then run `Build_Report.bat`. The report has a conditional screenshot slot; no fabricated capture is supplied.
3. Record your own 5-7 minute demonstration using `DEMO_GUIDE.md`, upload it, test evaluator access and add its link to the README/Classroom submission.
4. Verify the pushed repository contains the report, source, adapters, fixed manifest and logs. Base weights are reproducibly downloaded by `scripts/download.py`; they do not need to be committed.
5. Attach the main IEEE PDF, project ZIP, supplementary GPU PDF, source URL and video link to Classroom. Check the exact Classroom time for the November 10 deadline. Attend the live demo and prepare to explain/modify code or evaluate an unseen case.

The PDF's rubric heading says 100 marks, while its listed categories total 135. Only the instructor can resolve the weighting. Implemented components and passing tests do not guarantee full-credit performance or defense marks.
