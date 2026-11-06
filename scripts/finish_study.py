"""Continue this CPU study in order once its running training/base-validation finish.

This helper never tunes on final results and never overwrites an evaluation file.
It is useful when training takes a long time on a laptop. Run only one instance.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
phases=[]

def run(name,args,stdout=None):
    print('PHASE:',name,flush=True)
    started=time.perf_counter()
    result=subprocess.run(args,stdout=stdout,stderr=subprocess.STDOUT)
    phases.append({'phase':name,'returncode':result.returncode,'elapsed_s':time.perf_counter()-started,
                   'finished_utc':datetime.now(timezone.utc).isoformat()})
    (ROOT/'results/pipeline_phases.json').write_text(json.dumps(phases,indent=2))
    if result.returncode:
        raise SystemExit(f'Phase failed: {name}. Preserve partial outputs and inspect the logs.')

def main():
    deadline=time.monotonic()+3*3600
    print('Waiting for the already-running full training and base validation. No test access yet.',flush=True)
    while True:
        training=(ROOT/'results/train_console.log').read_text(encoding='utf-8',errors='replace')
        validation=(ROOT/'results/validation_base_console.log').read_text(encoding='utf-8',errors='replace')
        if 'Traceback (most recent call last)' in training or 'Traceback (most recent call last)' in validation:
            raise SystemExit('An earlier phase failed. Inspect its console log.')
        if 'Training complete' in training and (ROOT/'results/validation_base_metrics.json').exists():
            break
        if time.monotonic()>deadline:
            raise SystemExit('Wait limit reached; no test evaluation started.')
        time.sleep(15)
    py=sys.executable
    with (ROOT/'results/tests.txt').open('w') as out:
        run('correctness tests',[py,'-m','pytest','-q'],out)
    run('selected validation',[py,'scripts/evaluate.py','--split','validation','--checkpoint','adapted'])
    run('freeze protocol',[py,'scripts/freeze.py'])
    run('base final test',[py,'scripts/evaluate.py','--split','test','--checkpoint','base','--final-test'])
    run('adapted final test',[py,'scripts/evaluate.py','--split','test','--checkpoint','adapted','--final-test'])
    run('sequential benchmarks',[py,'scripts/benchmark.py'])
    run('qualitative probes',[py,'scripts/stress_cases.py'])
    run('analysis artifacts',[py,'scripts/analyze.py'])
    compiler=ROOT/'tools/tectonic/tectonic.exe'
    if compiler.exists():
        os.environ['TECTONIC_CACHE_DIR']=str(ROOT/'tools/tectonic-cache')
        run('IEEE LaTeX compilation',[str(compiler),'--keep-logs','report/main.tex'])
        destination=ROOT/'output/pdf/Criterion_Report.pdf'
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/'report/main.pdf',destination)
    else:
        print('Compile report/main.tex with your LaTeX installation.',flush=True)
    print('COMPLETED: full study, outputs ready for final visual review.',flush=True)

if __name__=='__main__':main()
