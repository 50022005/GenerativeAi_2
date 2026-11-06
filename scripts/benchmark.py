import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from trialmatch.data import ROOT
from trialmatch.model import load, setup
from trialmatch.decoding import greedy, speculative
from trialmatch.attention import reference, tiled
from trialmatch.memory import peak_rss_mb

def worker(args):
    setup()
    if args.kind == 'attention':
        torch.manual_seed(42)
        q,k,v = [torch.randn(1,3,args.input,32) for _ in range(3)]
        fn = reference if args.mode == 'reference' else tiled
        with torch.inference_mode():
            fn(q,k,v)
            times=[]
            for _ in range(args.repeats):
                t=time.perf_counter(); output=fn(q,k,v); times.append(time.perf_counter()-t)
            measured_rss=peak_rss_mb()
            expected=reference(q,k,v)
        result={'kind':'attention','mode':args.mode,'shape':list(q.shape),
                'median_s':statistics.median(times),'repeated_s':times,
                'max_abs_error':float((output-expected).abs().max()),
                'mean_abs_error':float((output-expected).abs().mean()),
                'dense_score_mib':1*3*args.input**2*4/1024**2,
                'tile_score_mib':1*3*min(args.input,32)**2*4/1024**2,
                'peak_process_rss_mib':measured_rss}
    else:
        target,tokenizer=load(adapter=args.checkpoint=='adapted')
        draft=load(draft=True)[0] if args.mode=='speculative' else None
        source=tokenizer.encode('A patient has a documented clinical history. Review the evidence before deciding. '*100,add_special_tokens=False)
        ids=source[:args.input]
        def run():
            return speculative(target,draft,ids,args.output) if draft is not None else greedy(target,ids,args.output,cached=args.mode=='cached')
        run() # Warm-up excluded from timing, included in process RSS high-water mark.
        runs=[run() for _ in range(args.repeats)]
        result={'kind':'decoding','mode':args.mode,'checkpoint':args.checkpoint,
                'input_tokens':len(ids),'output_tokens':args.output,'repeats':args.repeats,
                'runs':runs,'median_s':statistics.median(r['total_s'] for r in runs),
                'median_prefill_s':statistics.median(r['prefill_s'] for r in runs),
                'median_decode_s':statistics.median(r['decode_s'] for r in runs),
                'median_tokens_per_s':statistics.median(r['tokens_per_s'] for r in runs),
                'peak_process_rss_mib':peak_rss_mb(),'tokens':runs[0]['tokens']}
        assert all(r['tokens']==result['tokens'] for r in runs)
    print('RESULT_JSON:'+json.dumps(result),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--worker',action='store_true')
    p.add_argument('--kind',default='decoding')
    p.add_argument('--mode',default='cached')
    p.add_argument('--input',type=int,default=64)
    p.add_argument('--output',type=int,default=16)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--checkpoint',choices=['base','adapted'],default='adapted')
    args=p.parse_args()
    if args.worker:
        return worker(args)
    results=[]
    configurations=[('decoding',n,o,m) for n,o in [(32,8),(128,16),(256,32)] for m in ['uncached','cached','speculative']]
    configurations += [('attention',n,0,m) for n in [64,128,256,512] for m in ['reference','tiled']]
    dest=ROOT/'results/benchmarks.json'
    for kind,n,o,mode in configurations:
        cmd=[sys.executable,__file__,'--worker','--kind',kind,'--mode',mode,'--input',str(n),'--output',str(o),
             '--repeats',str(args.repeats),'--checkpoint',args.checkpoint]
        print('Benchmark',kind,n,o,mode,flush=True)
        run=subprocess.run(cmd,capture_output=True,text=True,check=True)
        result=json.loads(next(s[len('RESULT_JSON:'):] for s in run.stdout.splitlines() if s.startswith('RESULT_JSON:')))
        if kind=='decoding' and mode!='uncached':
            ref=next(r for r in results if r['kind']==kind and r['input_tokens']==n and r['mode']=='uncached')
            result['agrees_with_uncached']=result['tokens']==ref['tokens']
            if not result['agrees_with_uncached']:
                raise AssertionError('Greedy token mismatch: '+str((n,o,mode)))
            result['speedup_vs_uncached']=ref['median_s']/result['median_s']
            if mode=='speculative':
                cached=next(r for r in results if r['kind']==kind and r['input_tokens']==n and r['mode']=='cached')
                result['speedup_vs_cached']=cached['median_s']/result['median_s']
        results.append(result)
        dest.write_text(json.dumps({'device':'cpu','threads':2,'dtype':'float32','warmups':1,
                                    'memory_policy':'Fresh subprocess for each configuration; process peak RSS includes interpreter, model(s), warmup and tensor buffers. Not isolated KV allocation.',
                                    'records':results},indent=2))
    print('All benchmarks completed and token equality verified.',flush=True)

if __name__=='__main__': main()
