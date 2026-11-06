"""Standalone local browser application. Patient data stays in this process."""
import json
import hashlib
import os
import threading
import time
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from flask import Flask, jsonify, render_template, request, Response
from trialmatch.data import ROOT, read_split, encode
from trialmatch.model import load
from trialmatch.pipeline import predict, report, NOTICE
from trialmatch.decoding import greedy, speculative

app = Flask(__name__, template_folder='web', static_folder='web', static_url_path='/assets')
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024
executor = ThreadPoolExecutor(max_workers=1)
jobs = {}
models = {}
state = {'loading': False, 'ready': False, 'error': None}

def get_models(draft=False):
    if 'target' not in models:
        state['loading'] = True
        try:
            models['target'], models['tokenizer'] = load()
            models['checkpoint'] = json.loads((ROOT/'models/adapter/config.json').read_text())
            models['adapter_sha256'] = hashlib.sha256((ROOT/'models/adapter/adapter.pt').read_bytes()).hexdigest()
            state.update(ready=True, error=None)
        except Exception as exc:
            state['error'] = str(exc)
            raise
        finally:
            state['loading'] = False
    if draft and 'draft' not in models:
        models['draft'] = load(draft=True)[0]
    return models

def submit(fn):
    if sum(j['status'] in ('queued','running') for j in jobs.values()) >= 3:
        return jsonify(error='Three requests are already queued. Please wait.'), 429
    for key in list(jobs):
        if len(jobs) < 20:
            break
        if jobs[key]['status'] in ('done','error'):
            del jobs[key]
    key = uuid.uuid4().hex
    jobs[key] = {'status':'queued'}
    def run():
        jobs[key] = {'status':'running'}
        try:
            jobs[key] = {'status':'done','result':fn()}
        except Exception as exc:
            jobs[key] = {'status':'error','error':str(exc)}
    executor.submit(run)
    return jsonify(job_id=key), 202

@app.before_request
def local_requests():
    if request.method == 'POST':
        origin = request.headers.get('Origin')
        if origin and origin != request.host_url.rstrip('/'):
            return jsonify(error='Use this app from its local browser page.'), 403
        if not request.is_json:
            return jsonify(error='Expected a JSON request.'), 415

@app.errorhandler(413)
def too_large(_):
    return jsonify(error='Input is too large. Use a shorter record.'), 413

@app.get('/')
def index():
    return render_template('index.html')

@app.get('/api/status')
def status():
    checkpoint = ROOT/'models/adapter/config.json'
    return jsonify(**state, checkpoint=json.loads(checkpoint.read_text()) if checkpoint.exists() else None,
                   notice=NOTICE, device='CPU')

@app.get('/api/criteria')
def criteria():
    # Application convenience metadata is restricted to training patients.
    trials = {}
    for r in read_split('train'):
        key = r['trial_id']
        trial = trials.setdefault(key, {})
        trial[(r['criterion_type'],r['criterion_text'])] = {'type':r['criterion_type'],'text':r['criterion_text']}
    return jsonify({k:list(v.values()) for k,v in sorted(trials.items())})

def validate(payload):
    patient = payload.get('patient','')
    criteria = payload.get('criteria',[])
    mode = payload.get('mode','cached')
    if not isinstance(patient,str) or not patient.strip() or len(patient)>20000:
        raise ValueError('Enter a patient record with 1 to 20,000 characters.')
    if not isinstance(criteria,list) or not 1<=len(criteria)<=20:
        raise ValueError('Enter between 1 and 20 criteria.')
    for c in criteria:
        if not isinstance(c,dict) or c.get('type') not in ('inclusion','exclusion'):
            raise ValueError('Each criterion needs an inclusion or exclusion type.')
        if not isinstance(c.get('text'),str) or not 1<=len(c['text'].strip())<=3000:
            raise ValueError('Each criterion needs 1 to 3,000 characters.')
    if mode not in ('cached','uncached','speculative'):
        raise ValueError('Choose a supported inference mode.')
    return patient.strip(),criteria,mode

@app.post('/api/screen')
def screen():
    try:
        patient,criteria,mode=validate(request.get_json())
    except (ValueError,AttributeError) as exc:
        return jsonify(error=str(exc)),400
    def run():
        m=get_models(mode=='speculative')
        results=[predict(m['target'],m['tokenizer'],{'note':patient,'criterion_type':c['type'],
                                                  'criterion_text':c['text']},mode,m.get('draft')) for c in criteria]
        result=report(patient,results,mode)
        result['checkpoint']=m['checkpoint']
        result['adapter_sha256']=m['adapter_sha256']
        return result
    return submit(run)

@app.post('/api/benchmark')
def benchmark():
    try:
        patient,criteria,mode=validate(request.get_json())
    except (ValueError,AttributeError) as exc:
        return jsonify(error=str(exc)),400
    def run():
        m=get_models(True)
        ids,_,info=encode({'note':patient,'criterion_type':criteria[0]['type'],'criterion_text':criteria[0]['text']},m['tokenizer'])
        # Same inputs, fixed raw output budget, one warmup and two measured repeats.
        fns={'uncached':lambda:greedy(m['target'],ids,16,cached=False),
             'cached':lambda:greedy(m['target'],ids,16,cached=True),
             'speculative':lambda:speculative(m['target'],m['draft'],ids,16)}
        measurements={}
        for name,fn in fns.items():
            fn()
            runs=[fn(),fn()]
            measurements[name]={'runs':runs,'mean_s':sum(r['total_s'] for r in runs)/2,
                                'output_text':m['tokenizer'].decode(runs[0]['tokens'])}
        equal=all(run['tokens']==measurements['cached']['runs'][0]['tokens']
                  for v in measurements.values() for run in v['runs'])
        return {'input_tokens':len(ids),'output_tokens':16,'token_equality':equal,'measurements':measurements,
                'note':'Live raw-generation comparison, not a clinical assessment. Timings include prefill. Draft is an uncalibrated six-layer pruned model.'}
    return submit(run)

@app.get('/api/job/<key>')
def job(key):
    if key not in jobs:
        return jsonify(error='Result expired or unknown.'),404
    return jsonify(jobs[key])

@app.get('/api/export/<key>')
def export(key):
    job=jobs.get(key,{})
    result=job.get('result',{})
    if job.get('status')!='done' or 'criteria' not in result:
        return jsonify(error='Complete a screening first.'),404
    return Response(render_template('export.html',r=result),mimetype='text/html',
                    headers={'Content-Disposition':'attachment; filename=eligibility-screening.html'})

@app.get('/api/results')
def results():
    files=['validation_base_metrics.json','validation_adapted_metrics.json',
           'test_base_metrics.json','test_adapted_metrics.json','benchmarks.json']
    return jsonify({f:json.loads((ROOT/'results'/f).read_text()) for f in files if (ROOT/'results'/f).exists()})

if __name__=='__main__':
    if '--no-browser' not in os.sys.argv:
        threading.Timer(1.5,lambda:webbrowser.open('http://127.0.0.1:7860')).start()
    app.run(host='127.0.0.1',port=7860,debug=False,threaded=True)
