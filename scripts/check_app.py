"""Check the running local HTTP application with synthetic input, without UI automation."""
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/application_check'
OUT.mkdir(exist_ok=True)
BASE='http://127.0.0.1:7860'

def get(path):
    with urllib.request.urlopen(BASE+path,timeout=30) as r:return json.load(r)

def post(path,payload):
    req=urllib.request.Request(BASE+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=30) as r:return json.load(r)

def wait(job):
    deadline=time.monotonic()+300
    while time.monotonic()<deadline:
        state=get('/api/job/'+job)
        if state['status']=='done':return state['result']
        if state['status']=='error':raise RuntimeError(state['error'])
        time.sleep(.4)
    raise TimeoutError('Local job timed out.')

payload={'patient':'0. A 52-year-old woman has type 2 diabetes.\n1. She takes metformin daily.\n2. Her most recent HbA1c is 7.5%.\n3. Kidney function is not documented.',
         'criteria':[{'type':'inclusion','text':'Age 18 to 65 years'},
                     {'type':'exclusion','text':'Severe renal impairment'}]}
status=get('/api/status')
assert status['checkpoint']['selected_epoch']==3
trials=get('/api/criteria')
assert trials and all(c['type'] in ('inclusion','exclusion') for rows in trials.values() for c in rows)
results={}
for mode in ('cached','uncached','speculative'):
    job=post('/api/screen',dict(payload,mode=mode))['job_id']
    result=wait(job)
    assert len(result['criteria'])==2 and result['adapter_sha256']
    results[mode]=result
    (OUT/f'screening_{mode}.json').write_text(json.dumps(result,indent=2))
    if mode=='cached':
        with urllib.request.urlopen(BASE+'/api/export/'+job) as response:html=response.read()
        assert b'Clinical eligibility screening' in html and b'Adapter SHA-256' in html
        (OUT/'report.html').write_bytes(html)
    print('Checked screening:',mode,flush=True)
tokens=lambda r:[c['timing']['tokens'] for c in r['criteria']]
assert tokens(results['cached'])==tokens(results['uncached'])==tokens(results['speculative'])
benchmark=wait(post('/api/benchmark',dict(payload,mode='cached'))['job_id'])
assert benchmark['token_equality'] is True
(OUT/'live_benchmark.json').write_text(json.dumps(benchmark,indent=2))
summary={'synthetic_input_only':True,'screening_token_equality':True,'live_benchmark_token_equality':True,
         'selected_epoch':3,'trial_metadata_count':len(trials),'export_verified':True,
         'scope':'Actual local HTTP backend and generated HTML; final browser UI capture was blocked by the computer-use tool.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2),flush=True)
