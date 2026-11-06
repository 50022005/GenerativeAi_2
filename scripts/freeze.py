import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
path = ROOT/'results/frozen_protocol.json'
if path.exists():
    raise SystemExit('Protocol already frozen; preserve this audit record.')
files = ['trialmatch/data.py', 'trialmatch/model.py', 'trialmatch/decoding.py',
         'trialmatch/pipeline.py', 'trialmatch/metrics.py', 'scripts/evaluate.py',
         'data/split_manifest.json', 'data/processed/train.jsonl',
         'data/processed/validation.jsonl', 'data/processed/test.jsonl',
         'results/training_config.json']
record = {'frozen_utc': datetime.now(timezone.utc).isoformat(),
          'adapter_sha256': hashlib.sha256((ROOT/'models/adapter/adapter.pt').read_bytes()).hexdigest(),
          'source_sha256': {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
          'selection': 'Minimum token-weighted validation loss across three fixed epochs.',
          'decoding': 'Greedy, finite label/bitmap grammar, max input+output 1024, float32 CPU, eager attention',
          'test_policy': 'Evaluate base and adapted once, identical examples. No tuning after test.'}
path.write_text(json.dumps(record, indent=2))
print('Final evaluation protocol frozen.')
