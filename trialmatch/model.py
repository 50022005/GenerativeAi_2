import json
import math
import os
from pathlib import Path
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from .data import ROOT

def setup(seed=42):
    torch.manual_seed(seed)
    torch.set_num_threads(int(os.environ.get('TRIALMATCH_THREADS', '2')))

class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
    def forward(self, x):
        return self.base(x) + (x @ self.a.T @ self.b.T) * self.scale

def attach_lora(model, last_layers=6, rank=8, alpha=16):
    model.requires_grad_(False)
    for layer in model.model.layers[-last_layers:]:
        for name in ('q_proj', 'v_proj'):
            attn = layer.self_attn
            setattr(attn, name, LoRALinear(getattr(attn, name), rank, alpha))
    return model

def load(adapter=True, draft=False):
    setup()
    base = ROOT / 'models/base'
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True,
                                               dtype=torch.float32, attn_implementation='eager')
    if draft:
        # An explicitly identified, uncalibrated, layer-pruned draft. No tokenizer conversion.
        model.model.layers = nn.ModuleList(list(model.model.layers[:6]))
        model.config.num_hidden_layers = 6
        for i, layer in enumerate(model.model.layers):
            layer.self_attn.layer_idx = i
    elif adapter:
        config_path = ROOT / 'models/adapter/config.json'
        if not config_path.exists():
            raise FileNotFoundError('Trained adapter missing. Run Train_CPU.bat first.')
        config = json.loads(config_path.read_text())
        attach_lora(model, **{k: config[k] for k in ('last_layers', 'rank', 'alpha')})
        state = torch.load(ROOT / 'models/adapter/adapter.pt', map_location='cpu', weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(k.endswith(('.a', '.b')) for k in missing):
            raise ValueError('Incomplete or incompatible adapter.')
    return model.eval(), tokenizer

def save_adapter(model, config, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()
                if k.endswith(('.a', '.b'))}, dest / 'adapter.pt')
    (dest / 'config.json').write_text(json.dumps(config, indent=2))
