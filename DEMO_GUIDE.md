# Six-minute application demonstration

Record your own voice and the actual application window. This is a recording plan, not evidence that a video already exists. Keep the resulting video between five and seven minutes. Upload it to a location your evaluator can open, test the link while signed out, and put the link in the Classroom submission and repository README.

| Time | Show and explain |
|---|---|
| 0:00–0:35 | Introduce yourself, i230028, and the CPU-only educational screening task. State that this is not a clinical enrollment decision. |
| 0:35–1:25 | Open the app through Start_App.bat. Load the synthetic example, explain sentence IDs, and enter an extra unseen criterion. Show inclusion versus exclusion types. |
| 1:25–2:15 | Run screening with KV caching. Inspect each model label and exact citation. Explain missing information and why a grammatical output or an irrelevant citation can still be wrong. Show a visible failure honestly. |
| 2:15–2:45 | Export the readable report and open it. Explain provenance, truncation notices and rule-based aggregation. |
| 2:45–3:55 | Open Inference lab and run a live comparison. Explain uncached recomputation, cached reuse, drafting/verification, matching-prefix acceptance, correction and cache rollback. Show token equality and actual timings, even if speculation is slower. You can shorten idle waiting in the recording but disclose the edit. |
| 3:55–4:45 | Open Study results. Show before/after held-out metrics, separate inclusion/exclusion results, evidence scores and a confusion matrix from the report. Explain at least one rare-class error. |
| 4:45–5:25 | Show the split manifest and training curve. Explain 37/8/8 patient grouping and that GPT annotations are excluded. State the selected epoch and actual training time from the logs. |
| 5:25–6:00 | Explain tiled online softmax, numerical agreement, CPU limitations and AI assistance. End with the repository/report/weights locations and a short limitation statement. |

Prepare the page and example before recording so loading does not consume the whole video. Do not present precomputed logs as a live benchmark. The app's benchmark really runs when clicked. The final recorded application must use the supplied selected adapter, not a smoke checkpoint.

## Defense questions to practice

1. **Why patient grouping?** Criteria from the same patient share the same note. Row-random splits can leak patient facts. The assignment's exact sorted-then-shuffled split is fixed.
2. **What features are forbidden?** GPT label/explanation/citations, annotation-quality judgments, old training flags, and any expert target inserted into inputs. The prompt uses only note, criterion type/text and fixed instructions.
3. **What is trained?** Rank-eight query/value residuals in the last six blocks: 92,160 parameters. Base weights stay frozen. Frozen early blocks still perform computation; gradients start at trainable late blocks.
4. **What does response-only loss mean?** The model predicts the code and evidence bitmap autoregressively. Prompt tokens have no loss. Projecting only the necessary response hidden states avoids a large vocabulary tensor without changing the intended objective.
5. **Does the grammar decide eligibility?** No. It permits the applicable label tokens and binary evidence tokens. Their relative logits still come from the Transformer.
6. **How does the cache handle positions?** A new token's absolute position starts at the stored sequence length. The attention mask includes past plus current tokens. Prior keys/values are reused; the cache extends across steps.
7. **What happens after a speculative rejection?** Both caches are cropped to the accepted prefix. The target's corrective token is emitted and forwarded through both models. The rejected draft state cannot remain.
8. **Why can speculation be slower?** The draft performs extra work. Low acceptance means repeated drafts and corrections, while block verification and model memory have costs. A smaller model does not guarantee useful proposals.
9. **What does online softmax retain?** A row maximum, scaled exponential sum, and scaled weighted-value accumulator. When the maximum changes, previous sums are rescaled so the result remains mathematically equivalent to dense attention.
10. **What does memory mean in the table?** Fresh-process peak resident RAM including interpreter, weights and warmup. It is not Python-only allocation or an isolated cache measurement. Logical score-buffer sizes are listed separately.
11. **What does no evidence mean?** It can mean the record is missing information or that retrieval failed. It does not establish either eligibility or ineligibility. The app retains the model label and requests review.
12. **Why not claim clinical reliability?** Small general-purpose model, imbalanced small cohort, uncalibrated confidence, template explanations, limited logical/temporal reasoning and no prospective clinical validation.

Practice changing a tile size, the speculative block size, or a synthetic criterion and rerunning a correctness test. Do not tune on test cases during defense.
