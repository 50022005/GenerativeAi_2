import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trialmatch.data import ROOT, read_split
from trialmatch.model import load
from trialmatch.pipeline import predict
from trialmatch.metrics import compute

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['validation', 'test'], default='validation')
    p.add_argument('--checkpoint', choices=['base', 'adapted'], required=True)
    p.add_argument('--final-test', action='store_true')
    args = p.parse_args()
    if args.split == 'test' and not args.final_test:
        p.error('Test is locked. Select/freeze model using validation, then pass --final-test once.')
    if args.split == 'test':
        freeze = ROOT/'results/frozen_protocol.json'
        if not freeze.exists():
            p.error('Run scripts/freeze.py before final test.')
        protocol = json.loads(freeze.read_text())
        digest = hashlib.sha256((ROOT/'models/adapter/adapter.pt').read_bytes()).hexdigest()
        if digest != protocol['adapter_sha256']:
            p.error('Adapter changed since protocol freeze.')
        for file, expected in protocol.get('source_sha256', {}).items():
            if hashlib.sha256((ROOT/file).read_bytes()).hexdigest() != expected:
                p.error('Inference/evaluation source changed after freeze: '+file)
    name = f'{args.split}_{args.checkpoint}'
    output = ROOT/'results'/f'{name}_predictions.jsonl'
    if output.exists():
        p.error('Output already exists; preserve the original run. Use a separate results directory for a new study.')
    model, tokenizer = load(adapter=args.checkpoint == 'adapted')
    rows = []
    with output.open('w', encoding='utf-8') as f:
        for row in read_split(args.split):
            pred = predict(model, tokenizer, row)
            result = {'annotation_id': row['annotation_id'], 'patient_id': row['patient_id'],
                      'trial_id': row['trial_id'], 'criterion_type': row['criterion_type'],
                      'gold_label': row['label'], 'gold_evidence': row['evidence'], 'prediction': pred}
            rows.append(result)
            f.write(json.dumps(result, ensure_ascii=False)+'\n')
            f.flush()
            if len(rows) % 10 == 0:
                print(name, len(rows), flush=True)
    metrics = compute(rows)
    (ROOT/'results'/f'{name}_metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics['overall'], indent=2), flush=True)

if __name__ == '__main__':
    main()
