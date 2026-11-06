import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from trialmatch.data import ROOT, read_split, encode
from trialmatch.model import load, attach_lora
from scripts.train import response_loss
model, tokenizer=load(adapter=False)
attach_lora(model)
model.eval()
tokens,labels,_=encode(read_split('train')[0],tokenizer,1024,True)
x,y=torch.tensor([tokens]),torch.tensor([labels])
with torch.no_grad():
    t=time.perf_counter(); full=model(input_ids=x,labels=y,use_cache=False).loss; full_s=time.perf_counter()-t
    t=time.perf_counter(); tail=response_loss(model,x,y); tail_s=time.perf_counter()-t
torch.testing.assert_close(full,tail,rtol=1e-6,atol=1e-6)
model.train()
t=time.perf_counter(); response_loss(model,x,y).backward(); backward_s=time.perf_counter()-t
record={'full_loss':float(full),'response_loss':float(tail),'full_forward_s':full_s,
        'response_only_forward_s':tail_s,'response_forward_backward_s':backward_s,
        'input_tokens':len(tokens),'label_tokens':sum(t!=-100 for t in labels),
        'note':'Single training-example engineering check during development, not final benchmark.'}
(ROOT/'results/cpu_optimization.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2),flush=True)
