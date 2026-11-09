"""Validate and retain a downloaded GPU study without changing the CPU study."""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trialmatch.data import ROOT, read_split
from trialmatch.metrics import compute
from kaggle_study.core import TARGET, validation_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    digest = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    destination = ROOT / 'experiments' / ('kaggle_' + digest[:12])
    if destination.exists():
        raise SystemExit(f'Already imported: {destination}')
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        with zipfile.ZipFile(args.archive) as archive:
            for entry in archive.infolist():
                path = Path(entry.filename)
                if path.is_absolute() or '..' in path.parts or ':' in entry.filename:
                    raise ValueError('Unsafe archive member')
            archive.extractall(stage)
        success = json.loads((stage / 'success.json').read_text())
        protocol = json.loads((stage / 'protocol.json').read_text())
        selection = json.loads((stage / 'selection.json').read_text())
        history = json.loads((stage / 'validation_history.json').read_text())
        assert success['completed'] and success['test_evaluated'] is False
        assert protocol['prior_test_use'] and protocol['smoke_limit'] == 0
        assert tuple(protocol['target']) == TARGET
        assert protocol['split_manifest_sha256'] == hashlib.sha256((ROOT / 'data/split_manifest.json').read_bytes()).hexdigest()
        expected = {r['annotation_id']: r for r in read_split('validation')}
        verified = []
        for path in sorted(stage.glob('*_validation_predictions.jsonl')):
            predictions = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(predictions) == len(expected) == 206
            assert len({r['annotation_id'] for r in predictions}) == 206
            for prediction in predictions:
                gold = expected[prediction['annotation_id']]
                assert prediction['patient_id'] == gold['patient_id']
                assert prediction['gold_label'] == gold['label']
                assert prediction['gold_evidence'] == gold['evidence']
            metrics = compute(predictions)
            saved = json.loads(path.with_name(path.name.replace('predictions.jsonl', 'metrics.json')).read_text())
            assert metrics == saved, f'Metrics mismatch: {path.name}'
            verified.append(path.name)
        assert len(verified) == 7 and len(history) == 6
        chosen = max(history, key=lambda entry: entry['selection_score'])
        assert chosen == selection['validation']
        for entry in history:
            metrics = json.loads((stage / f"{entry['objective']}_epoch{entry['epoch']}_validation_metrics.json").read_text())
            assert entry['selection_score'] == validation_score(metrics)
        benchmarks = json.loads((stage / 'gpu_benchmarks.json').read_text())
        decoding = [r for r in benchmarks['records'] if r['kind'] == 'decoding']
        attention = [r for r in benchmarks['records'] if r['kind'] == 'attention']
        assert len(decoding) == 9 and len(attention) == 12
        assert all(r['token_equality'] for r in decoding)
        for length in (32, 128, 256):
            rows = [r for r in decoding if r['input_tokens'] == length]
            assert all(run['tokens'] == rows[0]['tokens'] for r in rows for run in r['runs'])
        assert all(r['max_abs_error'] < 2e-6 for r in attention)
        adapter = stage / 'selected_adapter' / 'adapter.pt'
        assert adapter.is_file()
        validation = {'archive_sha256': digest, 'validated_prediction_files': verified,
                      'validation_rows': 206, 'test_evaluated': False,
                      'selected_adapter_sha256': hashlib.sha256(adapter.read_bytes()).hexdigest()}
        (stage / 'local_verification.json').write_text(json.dumps(validation, indent=2))
        destination.parent.mkdir(exist_ok=True)
        shutil.copytree(stage, destination)
    print(destination)


if __name__ == '__main__':
    main()
