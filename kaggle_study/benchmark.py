import gc
import json
import statistics
import time
from pathlib import Path
import torch
from trialmatch.decoding import greedy, speculative
from trialmatch.attention import reference, tiled
from .core import attach, load_base, restore

def sync():torch.cuda.synchronize()

@torch.inference_mode()
def run(out):
    out=Path(out)
    model,tokenizer=load_base();attach(model);restore(model,out/'selected_adapter');model.eval()
    records=[];draft=None
    source=tokenizer.encode('A patient has a documented clinical history. Review the evidence before deciding. '*100,add_special_tokens=False)
    for mode in ('uncached','cached','speculative'):
        if mode=='speculative':
            draft,draft_tokenizer=load_base(draft=True)
            if tokenizer.get_vocab()!=draft_tokenizer.get_vocab() or tokenizer.all_special_ids!=draft_tokenizer.all_special_ids:
                raise ValueError('Target and draft tokenizers are not exactly compatible.')
        for length,budget in ((32,8),(128,16),(256,32)):
            ids=source[:length]
            fn=(lambda:speculative(model,draft,ids,budget)) if mode=='speculative' else (lambda:greedy(model,ids,budget,cached=mode=='cached'))
            gc.collect();torch.cuda.empty_cache();sync();torch.cuda.reset_peak_memory_stats()
            fn()
            runs=[fn() for _ in range(3)]
            assert all(r['tokens']==runs[0]['tokens'] for r in runs)
            ref=next((r for r in records if r['mode']=='uncached' and r['input_tokens']==length),None)
            if ref is not None and ref['tokens']!=runs[0]['tokens']:
                (out/'decoding_mismatch.json').write_text(json.dumps({'mode':mode,'input_tokens':length,'reference':ref,'runs':runs},indent=2))
                raise AssertionError('Greedy mismatch preserved; investigate numerical/cache behavior before claiming equality.')
            record={'kind':'decoding','mode':mode,'input_tokens':length,'output_tokens':budget,
                    'runs':runs,'tokens':runs[0]['tokens'],'token_equality':True,
                    'median_s':statistics.median(r['total_s'] for r in runs),
                    'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                    'peak_gpu_reserved_mib':torch.cuda.max_memory_reserved()/1024**2}
            records.append(record)
            (out/'gpu_benchmarks.json').write_text(json.dumps({'records':records},indent=2))
            print('GPU decoding',mode,length,budget,record['median_s'],flush=True)
    del model,draft;gc.collect();torch.cuda.empty_cache()
    for n,offset,causal in ((64,0,True),(128,0,True),(256,0,True),(512,0,True),(19,9,True),(31,0,False)):
        torch.manual_seed(42)
        q=torch.randn(1,3,n,32,device='cuda');k=torch.randn(1,3,n+offset,32,device='cuda');v=torch.randn_like(k)
        a=reference(q,k,v,causal,offset);b=tiled(q,k,v,causal,offset)
        torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
        max_error=float((a-b).abs().max());mean_error=float((a-b).abs().mean())
        del a,b
        for name,fn in [('reference',reference),('tiled',tiled)]:
            gc.collect();torch.cuda.empty_cache();sync();torch.cuda.reset_peak_memory_stats()
            fn(q,k,v,causal,offset);sync();times=[]
            for _ in range(3):
                sync();start=time.perf_counter();z=fn(q,k,v,causal,offset);sync();times.append(time.perf_counter()-start)
                del z
            records.append({'kind':'attention','mode':name,'query_length':n,'key_length':n+offset,'causal':causal,
                            'query_offset':offset,'max_abs_error':max_error,'mean_abs_error':mean_error,
                            'repeated_s':times,'median_s':statistics.median(times),
                            'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                            'peak_gpu_reserved_mib':torch.cuda.max_memory_reserved()/1024**2})
        del q,k,v
    result={'device':torch.cuda.get_device_name(0),'precision':'float32','warmups':1,'repeats':3,
            'timing':'CUDA synchronized before and after timers, including prefill and total decoding timers.',
            'memory':'Reset CUDA tensor-allocation/reservation peaks per configuration, including warmup and active models. Shared process; not whole-board memory or CPU RSS.',
            'records':records}
    (out/'gpu_benchmarks.json').write_text(json.dumps(result,indent=2))
