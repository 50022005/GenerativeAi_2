"""Pinned public downloads, atomic replacement, no token or paid service needed."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
REPO='HuggingFaceTB/SmolLM2-135M-Instruct'
REVISION='12fd25f77366fa6b3b4b768ec3050bf629380bac'
DATA_SHA='14c547184fac9b013cd17441fd028845b40150c76b022e58334f5494077e5234'

def fetch(url,path,sha=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and (sha is None or hashlib.sha256(path.read_bytes()).hexdigest()==sha):
        print('Present:',path.name,flush=True)
        return
    print('Downloading:',path.name,flush=True)
    temp=path.with_suffix(path.suffix+'.partial')
    with urllib.request.urlopen(url,timeout=60) as source,temp.open('wb') as out:
        while chunk:=source.read(1024*1024): out.write(chunk)
    if sha and hashlib.sha256(temp.read_bytes()).hexdigest()!=sha:
        raise ValueError('Download checksum failed: '+path.name)
    temp.replace(path)

def main():
    fetch('https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations/resolve/main/data/train-00000-of-00001.parquet',
          ROOT/'data/raw/annotations.parquet',DATA_SHA)
    for f in ['config.json','generation_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','model.safetensors']:
        fetch(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{f}',ROOT/'models/base'/f)
    (ROOT/'models/base/revision.json').write_text(json.dumps({'repo':REPO,'sha':REVISION},indent=2))
    print('Public weights and dataset ready. Base weights are Apache-2.0; dataset is public-domain.',flush=True)

if __name__=='__main__': main()
