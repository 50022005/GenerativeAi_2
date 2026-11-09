# Final browser verification

Completed November 9, 2026 using the actual local Flask application in the in-app browser. All records here use synthetic patient descriptions; no personal medical record was entered.

- Rejected an empty note and a malformed criterion, then recovered to successful screening.
- Ran cached, uncached and speculative screening through browser controls. The three saved screening results have identical emitted tokens and the original selected epoch-3 adapter fingerprint.
- Downloaded the cached report with the Export control. `export.html` is that unchanged download; it was opened in a browser to check record, criterion, citation, review and model provenance rendering.
- Triggered the live comparison from the inference tab. Equality passed with one warmup and two measured repeats per mode. Its observed values are in `summary.json`; this smoke comparison is separate from the original controlled benchmarks.
- Loaded the original held-out/validation metrics in Study results.
- Found NaN/missing criterion text invalidating the trial-list JSON. Filtered only invalid application convenience metadata, added the strict-JSON regression test, restarted and reloaded. The browser then listed 72 trials, and selecting NCT00015626 filled seven criteria.
- Repeated cached screening after the fix and checked a newly entered synthetic record and two custom criteria. Their outputs are retained separately.

Captures in `report/figures/application.jpg`, `inference.jpg` and `export.jpg` came directly from CUA browser screenshots. The original JPEG bytes were saved without retouching, rendering the page from another program, generating an image or altering model outputs. Hashes are in `summary.json`. A temporary loopback form transferred those capture bytes to disk; it was stopped after use and is not part of the product.

The nine-page IEEE PDF includes `application.jpg` on page 8. Its embedded raw JPEG stream matches the source capture bytes exactly. All nine rendered pages were inspected. The screenshot intentionally preserves an incorrect abstention on an age criterion despite a source sentence stating the age.

The final full suite passed 22 checks in 32.43 seconds. Frozen benchmark split/annotations, model/inference source and the original selected adapter stayed unchanged. These checks verify software behavior, not clinical reliability. Student recording, submission and defense remain separate requirements.
