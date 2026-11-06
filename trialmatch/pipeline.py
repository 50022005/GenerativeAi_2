import hashlib
import json
from datetime import datetime, timezone
from .data import encode, grammar, LABELS
from .decoding import greedy, speculative

NOTICE = 'Educational screening prototype. Not a clinical enrollment decision.'

def predict(model, tokenizer, row, mode='cached', draft=None, max_length=1024):
    ids, sents, info = encode(row, tokenizer, max_length)
    restrict = grammar(tokenizer, row['criterion_type'], len(sents))
    if mode == 'speculative':
        if draft is None:
            raise ValueError('Draft model required.')
        timing = speculative(model, draft, ids, len(sents)+1, restrict=restrict)
    elif mode in ('cached', 'uncached'):
        timing = greedy(model, ids, len(sents)+1, cached=mode == 'cached', restrict=restrict)
    else:
        raise ValueError('Unknown decoding mode.')
    raw = [tokenizer.decode([t]) for t in timing['tokens']]
    code = raw[0].strip()
    evidence = [i for (i, _), flag in zip(sents, raw[1:]) if flag == '1']
    label = LABELS[code]
    extracted = [{'sentence_id': i, 'text': s} for i, s in sents if i in evidence]
    # Verbatim, templated evidence rationale; no unverified generated clinical claims.
    explanation = f'The Transformer predicted "{label}" for this {row["criterion_type"]} criterion. '
    if extracted:
        explanation += 'Model-cited source text: ' + ' '.join(f'[{e["sentence_id"]}] {e["text"]}' for e in extracted)
    else:
        explanation += 'The model cited no patient sentence. This does not establish absence of the condition.'
    missing = label == 'not enough information' or not extracted or bool(info['omitted_sentence_ids'])
    return {'criterion_type': row['criterion_type'], 'criterion_text': row['criterion_text'],
            'label': label, 'evidence': evidence, 'evidence_text': extracted,
            'explanation': explanation, 'needs_review': missing,
            'review_reason': 'Check missing information, unsupported conclusion, or truncated record.' if missing else 'Verify the model interpretation against the criterion.',
            'raw_output': ''.join(raw), 'preprocessing': info, 'timing': timing,
            'explanation_method': 'Template plus verbatim model-selected source sentences; not a clinical rationale.'}

def aggregate(results):
    # A rule-based summary, never a dataset-labeled trial outcome.
    if any(r['label'] in ('not included', 'excluded') for r in results):
        return 'Potential criterion conflict - human review required'
    if any(r['needs_review'] or r['label'] == 'not applicable' for r in results):
        return 'Insufficient information or applicability review required'
    return 'No conflict predicted for supplied criteria - human review required'

def report(patient, results, mode):
    return {'notice': NOTICE, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'model': 'SmolLM2-135M-Instruct + local LoRA', 'decoding_mode': mode,
            'patient_sha256': hashlib.sha256(patient.encode()).hexdigest(),
            'patient_record': patient, 'aggregation_rule': 'Any conflict > any missing/applicability review > no predicted conflict. Supplied criteria only.',
            'overall_preliminary_outcome': aggregate(results), 'criteria': results}
