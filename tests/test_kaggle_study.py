import io
import json
import zipfile
import torch
from kaggle_study.core import attach, task_loss, balance_statistics, validation_score
from scripts.build_kaggle_notebook import synchronized_decoder, make_notebook
from trialmatch.data import ROOT

def test_balanced_loss_does_not_dilute_the_label_with_bitmap_length():
    logits=torch.zeros(1,5,8,requires_grad=True);targets=torch.tensor([[2,1,0,0,0]])
    loss=task_loss(logits,targets,'balanced',label_weight=1,positive_weight=4,one_id=1)
    loss.backward()
    gradient=logits.grad.abs().sum(-1).flatten()
    assert gradient[0] > gradient[1]
    assert gradient[1] > gradient[2]
    assert torch.isfinite(loss)

def test_uniform_loss_matches_cross_entropy():
    torch.manual_seed(1);x=torch.randn(1,6,9);y=torch.randint(0,9,(1,6))
    torch.testing.assert_close(task_loss(x,y,'uniform'),torch.nn.functional.cross_entropy(x.reshape(-1,9),y.reshape(-1)))

def test_training_only_balance_counts_and_score():
    rows=[{'criterion_type':t,'label':label,'evidence':ev} for t,label,ev in
          [('inclusion','included',[0]),('inclusion','included',[]),('inclusion','not enough information',[]),('exclusion','not excluded',[])]]
    encoded=[([],[],{'sentence_ids':[0,1]}) for _ in rows]
    result=balance_statistics(rows,encoded)
    assert result['positive_evidence_tokens']==1 and result['negative_evidence_tokens']==7
    assert result['class_weights']['inclusion']['not enough information']>result['class_weights']['inclusion']['included']
    assert validation_score({'overall':{'macro_f1':.4,'evidence':{'micro_f1':.6}}})==.5

def test_notebook_whitelist_and_synchronized_decoder():
    notebook,files=make_notebook()
    assert 'data/processed/test.jsonl' not in files
    assert not any('/results/' in p or p.endswith('main.tex') or p.endswith('adapter.pt') for p in files)
    assert sum(1 for _ in files['data/processed/train.jsonl'].splitlines())==603
    assert sum(1 for _ in files['data/processed/validation.jsonl'].splitlines())==206
    for c in notebook['cells']:
        if c['cell_type']=='code':compile(''.join(c['source']),'<notebook>','exec')
    source=files['trialmatch/decoding.py'].decode();compile(source,'decoding.py','exec')
    assert '_clock(model)' in source and '_clock(target)' in source and 'torch.cuda.synchronize(device)' in source
    namespace={'__name__':'test_gpu_decoder'};exec(source,namespace)
    from transformers import LlamaConfig,LlamaForCausalLM
    torch.set_num_threads(1)
    config=LlamaConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)
    config._attn_implementation='eager'
    torch.manual_seed(1);target=LlamaForCausalLM(config).eval()
    torch.manual_seed(2);draft=LlamaForCausalLM(config).eval()
    assert namespace['greedy'](target,[3,4],8,cached=False)['tokens']==namespace['speculative'](target,draft,[3,4],8)['tokens']

def test_all_layer_lora_preserves_initial_logits_and_frozen_weights():
    from transformers import LlamaConfig,LlamaForCausalLM
    torch.set_num_threads(1)
    config=LlamaConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)
    model=LlamaForCausalLM(config).eval();x=torch.tensor([[3,4,5]])
    before=model(x).logits.detach();attach(model,rank=4,alpha=8)
    torch.testing.assert_close(before,model(x).logits.detach(),atol=0,rtol=0)
    model(x).logits.sum().backward()
    assert all(p.grad is None for n,p in model.named_parameters() if not n.endswith(('.a','.b')))
