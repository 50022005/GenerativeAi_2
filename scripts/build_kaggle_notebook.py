"""Build one self-contained private-Kaggle upload; omit personal report/test artifacts."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def synchronized_decoder(source):
    source=source.replace('\r\n','\n')
    marker='@torch.inference_mode()\ndef greedy'
    prefix,tail=source.split(marker,1)
    first,second=tail.split('@torch.inference_mode()\ndef speculative',1)
    helper='def _clock(model):\n    device = next(model.parameters()).device\n    if device.type == "cuda":\n        torch.cuda.synchronize(device)\n    return time.perf_counter()\n\n'
    return prefix+helper+marker+first.replace('time.perf_counter()', '_clock(model)')+'@torch.inference_mode()\ndef speculative'+second.replace('time.perf_counter()', '_clock(target)')

def make_notebook():
    paths=['trialmatch/__init__.py','trialmatch/data.py','trialmatch/decoding.py','trialmatch/pipeline.py',
           'trialmatch/metrics.py','trialmatch/attention.py','kaggle_study/__init__.py','kaggle_study/core.py',
           'kaggle_study/run.py','kaggle_study/benchmark.py','data/processed/train.jsonl',
           'data/processed/validation.jsonl','data/split_manifest.json']
    files={p:(ROOT/p).read_bytes() for p in paths}
    files['trialmatch/decoding.py']=synchronized_decoder(files['trialmatch/decoding.py'].decode()).encode()
    manifest={'files':{p:hashlib.sha256(data).hexdigest() for p,data in files.items()},
              'purpose':'Private GPU validation study; no original report, student ID, test annotations, test predictions, credentials or old model weights.',
              'prior_test_use':True,'decoder_change':'Explicit CUDA synchronization added to CPU decoding timestamps only.'}
    files['source_bundle_manifest.json']=json.dumps(manifest,indent=2).encode()
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(files.items()):z.writestr(name,data)
    encoded=base64.b64encode(buffer.getvalue()).decode()
    bootstrap=f'''import base64, hashlib, io, json, os, zipfile
from pathlib import Path
WORK=Path("/kaggle/working/criterion_gpu")
WORK.mkdir(parents=True,exist_ok=True)
payload=base64.b64decode({encoded!r})
with zipfile.ZipFile(io.BytesIO(payload)) as z:
    for name in z.namelist():
        target=(WORK/name).resolve()
        if WORK.resolve() not in target.parents: raise ValueError("Unsafe bundle path")
    z.extractall(WORK)
manifest=json.loads((WORK/"source_bundle_manifest.json").read_text())
for name,expected in manifest["files"].items():
    assert hashlib.sha256((WORK/name).read_bytes()).hexdigest()==expected,name
assert not (WORK/"data/processed/test.jsonl").exists()
print("Verified isolated training/validation bundle. No held-out test annotations included.")
'''
    setup='''import os, subprocess, sys
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch
if not torch.cuda.is_available():
    raise RuntimeError("Select a Kaggle GPU in Settings > Accelerator before running. CPU fallback is disabled.")
print("GPU:",torch.cuda.get_device_name(0),"PyTorch:",torch.__version__)
# Retain Kaggle's CUDA-enabled PyTorch; never install the CPU-only project torch pin.
subprocess.check_call([sys.executable,"-m","pip","install","--quiet",
    "transformers==5.8.0","huggingface-hub==1.14.0","scikit-learn>=1.4,<2","matplotlib>=3.8,<4"])
'''
    run='''import os, subprocess, sys
os.chdir(WORK)
subprocess.run([sys.executable,"-u","-m","kaggle_study.run"],check=True)
'''
    export='''from IPython.display import FileLink, display
display(FileLink(str(WORK/"criterion_gpu_results.zip")))
print("Keep this notebook private. Download the ZIP after Save Version > Save & Run All finishes.")
'''
    def markdown(text):return {'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)}
    def code(text):return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':text.splitlines(keepends=True)}
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
             'language_info':{'name':'python','version':'3.11'},'kaggle':{'isGpuEnabled':True,'isInternetEnabled':True,'language':'python','sourceType':'notebook'}},
             'cells':[markdown('# Criterion: private GPU validation study\n\nUse GPU T4 and Internet. Keep this notebook **private**. This is an exploratory extension after the original CPU test was already evaluated. It does not claim a new blind test. Only the prescribed training and validation patients are embedded. The standalone local browser app remains part of the project.\n\nThe 360M target is compared before/after all-layer LoRA, with uniform and task-balanced training objectives. Selection uses validation classification/evidence F1, not test results. GPU benchmark timers synchronize CUDA. Results are measured, never assumed.'),
                      code(setup),code(bootstrap),code(run),code(export)]}
    for i,cell in enumerate(notebook['cells']):cell['id']=f'criterion-{i:02d}'
    output=ROOT/'output/kaggle';output.mkdir(parents=True,exist_ok=True)
    dest=output/'Criterion_GPU.ipynb';dest.write_text(json.dumps(notebook,indent=1),encoding='utf-8')
    (output/'upload_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Notebook built:',dest,'bytes:',dest.stat().st_size)
    return notebook,files

if __name__=='__main__':make_notebook()
