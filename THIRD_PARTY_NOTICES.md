# Third-party components

- **SmolLM2-135M-Instruct**: HuggingFaceTB, Apache-2.0. Model page and license: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct . The base weights are not checked into this repository; the download script pins their revision. Local low-rank adapters modify inference behavior, and the layer-pruned draft is explicitly identified as a project modification.
- **SmolLM2-360M-Instruct**: HuggingFaceTB, Apache-2.0. Considered as an alternative, not downloaded as the submitted target: https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct .
- **TrialGPT Criterion Annotations**: NCBI, listed as public-domain by the dataset card: https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations . Patient/trial IDs and original expert annotation semantics are preserved. GPT-produced fields are removed from prepared features.
- **Transformers** and **PyTorch**: model execution, cache representation, tensor operations and differentiation; see their installed distributions for their Apache-2.0/BSD-style licenses and notices.
- **Flask, NumPy, pandas, PyArrow, scikit-learn, Matplotlib and pytest**: runtime, analysis and verification components. Their own licenses continue to apply.
- **IEEEtran**: report-formatting class obtained through the TeX bundle. The paper is authored for this project; the class is not presented as original project code.
- **Tectonic**: optional local report compiler, downloaded from its official GitHub release. Its binary/cache are excluded from the source repository.

The low-rank wrapper, explicit decoding loops, tiled attention, preprocessing, evaluation, browser application and report source are project implementations assisted by OpenAI Codex. The original TrialGPT application was not used as the submitted system.
