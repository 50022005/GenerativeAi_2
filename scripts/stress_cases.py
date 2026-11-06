"""Supplementary synthetic probes, never mixed into common benchmark metrics."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from trialmatch.data import ROOT
from trialmatch.model import load
from trialmatch.pipeline import predict
if not (ROOT/'results/frozen_protocol.json').exists():
    raise SystemExit('Run only after the protocol is frozen; do not tune on these outputs.')
model,tokenizer=load()
cases=json.loads((ROOT/'data/demo_cases.json').read_text())
results=[]
for case in cases:
    prediction=predict(model,tokenizer,case)
    results.append({'case':case,'prediction':prediction})
    print(case['case_id'],prediction['label'],prediction['evidence'],flush=True)
(ROOT/'results/stress_cases.json').write_text(json.dumps({'note':'Synthetic qualitative probes, not supervised benchmark rows or held-out accuracy estimates.',
                                                         'records':results},indent=2))
