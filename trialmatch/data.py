import json
import re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'A': 'included', 'B': 'not included', 'C': 'excluded',
          'D': 'not excluded', 'E': 'not enough information', 'F': 'not applicable'}
ALLOWED = {'inclusion': 'ABEF', 'exclusion': 'CDEF'}

def sentences(note):
    """Preserve explicit original IDs; number free text deterministically otherwise."""
    lines = re.findall(r'(?m)^\s*(\d+)\.\s+(.*?)(?=^\s*\d+\.\s+|\Z)', note, re.S)
    if lines:
        result = [(int(i), text.strip()) for i, text in lines]
        if len({i for i, _ in result}) != len(result):
            raise ValueError('Duplicate sentence IDs.')
        return result
    return list(enumerate(s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', note) if s.strip()))

def prompt(row, sents):
    mapping = ', '.join(f'{c}={LABELS[c]}' for c in ALLOWED[row['criterion_type']])
    return ('Task: judge a clinical trial criterion using only the patient record. '
            'Missing evidence does not prove eligibility or ineligibility. '
            f'Label codes: {mapping}. Return one label code, then one binary evidence flag '
            'per listed sentence in order (1=relevant, 0=not cited).\nPatient:\n' +
            '\n'.join(f'{i}. {s}' for i, s in sents) +
            f"\nType: {row['criterion_type']}\nCriterion: {row['criterion_text']}\nAnswer:")

def encode(row, tokenizer, max_length=1024, supervised=False):
    # Keep complete sentences in original order. Never inspect gold to choose a window.
    original = sentences(row['note'])
    kept = original.copy()
    while kept:
        text = prompt(row, kept)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) + 1 + len(kept) <= max_length:
            break
        kept.pop()
    if not kept:
        raise ValueError('Criterion/prompt too long or no complete patient sentence fits.')
    info = {'sentence_ids': [i for i, _ in kept],
            'omitted_sentence_ids': [i for i, _ in original if i not in {j for j, _ in kept}],
            'input_tokens': len(ids)}
    if not supervised:
        return ids, kept, info
    code = next(k for k, v in LABELS.items() if v == row['label'])
    output = tokenizer.encode(' ' + code, add_special_tokens=False)
    for i, _ in kept:
        output += tokenizer.encode('1' if i in row['evidence'] else '0', add_special_tokens=False)
    assert len(output) == len(kept) + 1, 'Tokenizer does not support single-token grammar.'
    info['omitted_gold_evidence'] = sorted(set(row['evidence']) - set(info['sentence_ids']))
    return ids + output, [-100] * len(ids) + output, info

def grammar(tokenizer, criterion_type, count):
    ids = {s: tokenizer.encode((' ' + s) if s in LABELS else s, add_special_tokens=False) for s in 'ABCDEF01'}
    assert all(len(v) == 1 for v in ids.values())
    def restrict(logits, step):
        allowed = ALLOWED[criterion_type] if step == 0 else '01'
        valid = [ids[s][0] for s in allowed]
        result = logits.new_full(logits.shape, -float('inf'))
        result[..., valid] = logits[..., valid]
        return result
    return restrict

def read_split(split):
    return [json.loads(s) for s in (ROOT / 'data/processed' / f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]

def prepare():
    import hashlib
    import pandas as pd
    raw = ROOT / 'data/raw/annotations.parquet'
    df = pd.read_parquet(raw)
    patients = sorted(df.patient_id.unique().tolist())
    if len(patients) != 53:
        raise ValueError(f'Expected 53 patients, found {len(patients)}.')
    np.random.default_rng(42).shuffle(patients)
    groups = {'train': patients[:37], 'validation': patients[37:45], 'test': patients[45:]}
    manifest = {'seed': 42, 'algorithm': 'sort unique IDs; default_rng(42).shuffle',
                'source': 'ncbi/TrialGPT-Criterion-Annotations',
                'raw_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(), 'splits': {}}
    for name, pids in groups.items():
        rows = []
        for r in df[df.patient_id.isin(pids)].to_dict('records'):
            # Whitelist: never copy GPT outputs, explanations, old training flags or titles.
            rows.append({'annotation_id': int(r['annotation_id']), 'patient_id': r['patient_id'],
                         'trial_id': r['trial_id'], 'note': r['note'],
                         'criterion_type': r['criterion_type'], 'criterion_text': r['criterion_text'],
                         'label': r['expert_eligibility'], 'evidence': json.loads(r['expert_sentences'])})
        dest = ROOT / 'data/processed' / f'{name}.jsonl'
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n', encoding='utf-8')
        manifest['splits'][name] = {'patient_ids': pids, 'patient_count': len(pids),
                                    'annotation_count': len(rows),
                                    'per_patient_counts': {p: sum(r['patient_id'] == p for r in rows) for p in pids}}
    assert not (set(groups['train']) & set(groups['test']) or set(groups['validation']) & set(groups['test']) or set(groups['train']) & set(groups['validation']))
    (ROOT / 'data/split_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    prepare()
