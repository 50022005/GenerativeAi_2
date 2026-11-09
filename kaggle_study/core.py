import json
import math
from collections import Counter
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from trialmatch.data import ALLOWED, LABELS

TARGET = ('HuggingFaceTB/SmolLM2-360M-Instruct', 'a10cc1512eabd3dde888204e902eca88bddb4951')
DRAFT = ('HuggingFaceTB/SmolLM2-135M-Instruct', '12fd25f77366fa6b3b4b768ec3050bf629380bac')

class LoRA(nn.Module):
    def __init__(self, base, rank=16, alpha=32):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
    def forward(self, x):
        return self.base(x) + (x @ self.a.T @ self.b.T) * self.scale

def attach(model, rank=16, alpha=32):
    model.requires_grad_(False)
    for layer in model.model.layers:
        for name in ('q_proj', 'v_proj'):
            setattr(layer.self_attn, name, LoRA(getattr(layer.self_attn, name), rank, alpha))
    return model

def load_base(device='cuda', draft=False):
    repo, revision = DRAFT if draft else TARGET
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(repo, revision=revision, dtype=torch.float32,
                                               attn_implementation='eager').to(device)
    return model.eval(), tokenizer

def balance_statistics(rows, encoded):
    """Only training labels/evidence are consulted; no validation/test-derived weights."""
    class_weights = {}
    for kind in ALLOWED:
        counts = Counter(r['label'] for r in rows if r['criterion_type'] == kind)
        total = sum(counts.values())
        initial = {label: 1/math.sqrt(n) for label, n in counts.items()}
        scale = total/sum(counts[label]*v for label, v in initial.items())
        clipped = {label: min(4., max(.5, v*scale)) for label, v in initial.items()}
        norm = sum(counts[label]*v for label, v in clipped.items())/total
        class_weights[kind] = {label: v/norm for label, v in clipped.items()}
    positive = sum(sum(i in r['evidence'] for i in item[2]['sentence_ids']) for r, item in zip(rows, encoded))
    total = sum(len(item[2]['sentence_ids']) for item in encoded)
    return {'class_weights': class_weights, 'positive_evidence_tokens': positive,
            'negative_evidence_tokens': total-positive,
            'positive_evidence_weight': min(4., max(1., math.sqrt((total-positive)/max(1, positive))))}

def task_loss(logits, targets, objective, label_weight=1., positive_weight=1., one_id=None):
    """Logits predict one code and then all bitmap flags. Uses full-vocabulary CE."""
    ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction='none')
    if objective == 'uniform':
        return ce.mean()
    if objective != 'balanced' or one_id is None:
        raise ValueError('Balanced objective requires the evidence-one token ID.')
    flags = targets.reshape(-1)[1:]
    weights = torch.where(flags == one_id, ce.new_tensor(positive_weight), ce.new_tensor(1.))
    evidence = (ce[1:]*weights).sum()/weights.sum().clamp_min(1)
    return .5 * label_weight * ce[0] + .5 * evidence

def response_loss(model, tokens, labels, objective='uniform', **weights):
    n = int((labels != -100).sum().item())
    if tokens.shape[0] != 1:
        raise ValueError('This trainer uses microbatch one with gradient accumulation.')
    output = model(input_ids=tokens, use_cache=False, logits_to_keep=n+1)
    return task_loss(output.logits[:, :-1], labels[:, -n:], objective, **weights)

def save(model, config, destination):
    path = Path(destination)
    path.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if k.endswith(('.a','.b'))}
    torch.save(tensors, path/'adapter.pt')
    (path/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

def restore(model, path):
    state = torch.load(Path(path)/'adapter.pt', map_location=next(model.parameters()).device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(k.endswith(('.a','.b')) for k in missing):
        raise ValueError('Incomplete GPU adapter.')

def validation_score(metrics):
    overall = metrics['overall']
    return .5 * (overall['macro_f1'] + overall['evidence']['micro_f1'])
