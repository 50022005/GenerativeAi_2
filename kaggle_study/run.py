"""Train/compare two predetermined GPU configurations using validation only."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import torch
from trialmatch.data import ROOT, encode, read_split
from trialmatch.pipeline import predict
from trialmatch.metrics import compute
from .core import TARGET, DRAFT, attach, load_base, balance_statistics, response_loss, save, restore, validation_score

def sync():
    if torch.cuda.is_available():torch.cuda.synchronize()

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')

def seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)

def assess(model, tokenizer, rows, output):
    model.eval()
    predictions=[]
    with Path(output).open('w', encoding='utf-8') as f:
        for row in rows:
            result={'annotation_id':row['annotation_id'],'patient_id':row['patient_id'],
                    'criterion_type':row['criterion_type'],'gold_label':row['label'],
                    'gold_evidence':row['evidence'],'prediction':predict(model,tokenizer,row)}
            predictions.append(result)
            f.write(json.dumps(result)+'\n');f.flush()
            if len(predictions)%25==0:print('Validation predictions:',len(predictions),flush=True)
    return compute(predictions)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--limit',type=int,default=0,help='Development smoke only, never the submitted study')
    args=parser.parse_args()
    if not torch.cuda.is_available():raise SystemExit('Enable a Kaggle GPU before running. CPU fallback is deliberately disabled.')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    seed()
    out=ROOT/'gpu_results'
    if out.exists():out=ROOT/f'gpu_results_{time.time_ns()}'
    out.mkdir(exist_ok=False)
    train,val=read_split('train'),read_split('validation')
    assert len(train)==603 and len(val)==206
    assert not ({r['patient_id'] for r in train}&{r['patient_id'] for r in val})
    if args.limit:train,val=train[:args.limit],val[:min(4,args.limit)]
    protocol={'seed':42,'target':TARGET,'draft':DRAFT,'prior_test_use':True,
              'status':'Post-original-test exploratory extension; training and selection use validation only. No new blind-test claim.',
              'test_access':'No test annotations or prior test predictions are included in this cloud bundle.',
              'candidates':['uniform','balanced'],'epochs':args.epochs,'rank':16,'alpha':32,
              'layers':'q/v in every target block','lr':.0005,'accumulation':8,'max_length':1024,
              'selection':'Maximum mean of validation classification macro-F1 and evidence micro-F1; tie keeps earlier checkpoint.',
              'smoke_limit':args.limit,'gpu':torch.cuda.get_device_name(0),
              'gpu_memory_gib':torch.cuda.get_device_properties(0).total_memory/1024**3,
              'torch':torch.__version__,'cuda':torch.version.cuda,
              'packages':{k:importlib.metadata.version(k) for k in ['transformers','numpy','scikit-learn']},
              'split_manifest_sha256':hashlib.sha256((ROOT/'data/split_manifest.json').read_bytes()).hexdigest()}
    dump(out/'protocol.json',protocol)
    model,tokenizer=load_base()
    baseline=assess(model,tokenizer,val,out/'base_validation_predictions.jsonl')
    dump(out/'base_validation_metrics.json',baseline)
    del model;gc.collect();torch.cuda.empty_cache()
    global_best=-1.;selection=None;history=[]
    amp_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for objective in protocol['candidates']:
        seed();model,tokenizer=load_base();attach(model)
        encoded=[encode(r,tokenizer,1024,True) for r in train]
        stats=balance_statistics(train,encoded);dump(out/f'{objective}_balance.json',stats)
        one_id=tokenizer.encode('1',add_special_tokens=False)[0]
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=protocol['lr'])
        scaler=torch.amp.GradScaler('cuda',enabled=amp_dtype==torch.float16)
        started=time.perf_counter();candidate_best=-1.
        with (out/f'{objective}_training.jsonl').open('w') as log:
            for epoch in range(args.epochs):
                model.train();optimizer.zero_grad();order=list(range(len(train)))
                random.Random(42+epoch).shuffle(order);losses=[]
                for step,idx in enumerate(order):
                    tokens,labels,info=encoded[idx];row=train[idx]
                    x=torch.tensor([tokens],device='cuda');y=torch.tensor([labels],device='cuda')
                    group_size=min(8,len(order)-(step//8)*8)
                    with torch.autocast('cuda',dtype=amp_dtype):
                        loss=response_loss(model,x,y,objective,
                            label_weight=stats['class_weights'][row['criterion_type']][row['label']],
                            positive_weight=stats['positive_evidence_weight'],one_id=one_id)
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Non-finite loss; preserve logs and investigate.')
                    scaler.scale(loss/group_size).backward();losses.append(float(loss.detach()))
                    if (step+1)%8==0 or step+1==len(order):
                        scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(params,1.)
                        scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                    if step==0 or (step+1)%25==0:
                        sync();entry={'epoch':epoch+1,'step':step+1,'loss':sum(losses[-25:])/len(losses[-25:]),'elapsed_s':time.perf_counter()-started}
                        log.write(json.dumps(entry)+'\n');log.flush();print(objective,entry,flush=True)
                metrics=assess(model,tokenizer,val,out/f'{objective}_epoch{epoch+1}_validation_predictions.jsonl')
                score=validation_score(metrics)
                dump(out/f'{objective}_epoch{epoch+1}_validation_metrics.json',metrics)
                entry={'objective':objective,'epoch':epoch+1,'selection_score':score,
                       'accuracy':metrics['overall']['accuracy'],'macro_f1':metrics['overall']['macro_f1'],
                       'evidence_micro_f1':metrics['overall']['evidence']['micro_f1'],'mean_training_loss':sum(losses)/len(losses)}
                history.append(entry);dump(out/'validation_history.json',history);print(entry,flush=True)
                config=dict(protocol,objective=objective,selected_epoch=epoch+1,validation=entry,
                            rank=16,alpha=32,last_layers=len(model.model.layers),
                            trainable_parameters=sum(p.numel() for p in params),
                            base={'repo':TARGET[0],'sha':TARGET[1]},training_precision=str(amp_dtype),inference_precision='float32')
                if score>candidate_best:
                    candidate_best=score;save(model,config,out/f'{objective}_adapter')
                if score>global_best:
                    global_best=score;selection=config;save(model,config,out/'selected_adapter')
        del optimizer,scaler,params,model,x,y,loss;gc.collect();torch.cuda.empty_cache()
    dump(out/'selection.json',selection)
    print('VALIDATION SELECTION COMPLETE. Starting independent GPU inference/attention checks.',flush=True)
    from .benchmark import run as benchmark
    benchmark(out)
    (out/'README.txt').write_text('Exploratory Kaggle GPU study. Test annotations were not included or evaluated. Original CPU results remain unchanged. Do not replace the original app adapter with this differently sized model without using a compatible loader.\n')
    dump(out/'success.json',{'completed':True,'test_evaluated':False,'run_directory':out.name})
    archive=shutil.make_archive(str(ROOT/'criterion_gpu_results'),'zip',out)
    print('GPU STUDY COMPLETE:',archive,flush=True)

if __name__=='__main__':main()
