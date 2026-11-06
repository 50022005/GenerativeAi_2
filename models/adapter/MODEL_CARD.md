# Criterion CPU LoRA adapter

Base: HuggingFaceTB/SmolLM2-135M-Instruct, revision 12fd25f77366fa6b3b4b768ec3050bf629380bac (Apache-2.0).

Author/submission: Muhammad Hashim Rahim, i230028, FAST NUCES, Generative AI. OpenAI Codex assisted implementation and report preparation.

Adaptation: 92,160 trainable rank-eight query/value adapter parameters in the last six blocks; alpha 16; three fixed epochs; float32 CPU; AdamW learning rate 0.001; batch one, accumulation eight; seed 42. Selected epoch: 3, using validation response loss 0.3446.

Data: ncbi/TrialGPT-Criterion-Annotations, public-domain dataset. Fixed patient-grouped 37/8/8 split, 603/206/206 criterion rows. GPT-generated fields never enter model features. The complete patient-ID split is in data/split_manifest.json.

Held-out accuracy: 0.6117; six-class macro-F1: 0.2589; evidence micro-F1: 0.1452. Consult the report and full JSON metrics for minority-class errors, evidence conventions and limitations. No clinical reliability claim is made.

Observed prediction distribution: {"inclusion": {"included": 21, "not enough information": 54}, "exclusion": {"not excluded": 131}}. No sentence was cited in 181/206 test records. In particular, the model's concentration on frequent labels and weak exclusion evidence retrieval are major measured failures, even though adaptation improves accuracy over the unadapted decoder.

Output: one constrained label code followed by one evidence bit per patient sentence, parsed into label and original sentence IDs. Use trialmatch.pipeline.predict rather than loading this small adapter file as a complete language model.

The adapter file contains only LoRA matrices. Download public base weights with scripts/download.py and load the adapter through trialmatch.model.load. The uncalibrated six-layer speculative draft is constructed from base weights and does not use this adapter.

Educational screening prototype. Not a clinical enrollment decision. The general-purpose small model can produce incorrect labels and irrelevant/missing citations. Template explanations do not prove clinical entailment. All outputs require human review.
