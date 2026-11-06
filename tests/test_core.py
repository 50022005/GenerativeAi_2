import json
import torch
import pytest
from transformers import LlamaConfig, LlamaForCausalLM
from trialmatch.attention import reference, tiled
from trialmatch.decoding import greedy, speculative
from trialmatch.data import ROOT, sentences, encode
from trialmatch.metrics import evidence_metrics

@pytest.mark.parametrize('shape,causal,offset', [((1,2,7,8),True,0), ((2,3,19,16),True,0), ((1,1,31,7),False,0), ((1,2,4,8),True,9)])
def test_attention(shape, causal, offset):
    torch.manual_seed(3)
    q = torch.randn(shape)
    k = torch.randn((*shape[:-2], shape[-2]+offset, shape[-1]))
    v = torch.randn((*k.shape[:-1], 5))
    expected = reference(q,k,v,causal,offset)
    actual = tiled(q,k,v,causal,offset,query_tile=3,key_tile=4)
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)

def tiny(seed, layers=3):
    torch.manual_seed(seed)
    config = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2)
    config._attn_implementation = 'eager'
    return LlamaForCausalLM(config).eval()

@pytest.mark.parametrize('n,out',[(3,1),(7,9),(19,17)])
def test_decoding(n,out):
    torch.set_num_threads(1)
    target, draft = tiny(1), tiny(2,1)
    prompt = [3+i%20 for i in range(n)]
    ref = greedy(target,prompt,out,cached=False)
    cache = greedy(target,prompt,out,cached=True)
    spec = speculative(target,draft,prompt,out,block_size=4)
    assert ref['tokens'] == cache['tokens'] == spec['tokens']
    assert cache['cache_lengths'] == list(range(n,n+out))
    assert spec['rejections'] > 0

def test_spec_all_accept_and_grammar():
    model = tiny(1)
    mask = lambda logits, step: logits.masked_fill(torch.arange(32)!=((step+4)%32), -torch.inf)
    ref = greedy(model,[3,4,5],13,restrict=mask)
    spec = speculative(model,model,[3,4,5],13,restrict=mask)
    assert ref['tokens'] == spec['tokens']
    assert spec['acceptance_rate'] == 1

def test_eos_rejection_and_accept():
    target, draft = tiny(1), tiny(2,1)
    token = greedy(target,[3,5],1)['tokens'][0]
    for other in (draft,target):
        assert speculative(target,other,[3,5],10,eos=token)['tokens'] == [token]

def test_split_integrity():
    m=json.loads((ROOT/'data/split_manifest.json').read_text())['splits']
    ids=[set(m[s]['patient_ids']) for s in ('train','validation','test')]
    assert [len(x) for x in ids] == [37,8,8]
    assert not (ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
    for name in m:
        rows=[json.loads(s) for s in (ROOT/f'data/processed/{name}.jsonl').read_text(encoding='utf-8').splitlines()]
        assert all(set(r).isdisjoint({'gpt4_explanation','gpt4_sentences','gpt4_eligibility','training'}) for r in rows)
        assert all(set(r['evidence']) <= {i for i,_ in sentences(r['note'])} for r in rows)

def test_evidence_empty_policy():
    rows=[{'gold_evidence': [],'prediction':{'evidence':[]}},
          {'gold_evidence':[0],'prediction':{'evidence':[]}}]
    assert evidence_metrics(rows)['macro_f1'] == .5
    assert evidence_metrics(rows)['micro_f1'] == 0

def test_decimal_is_not_a_sentence_identifier():
    assert sentences('5.6 mmol/L glucose was recorded.') == [(0,'5.6 mmol/L glucose was recorded.')]
    assert sentences('7. A source sentence.\n11. Another sentence.') == [(7,'A source sentence.'),(11,'Another sentence.')]

def test_response_projection_loss_and_lora_identity():
    from scripts.train import response_loss
    from trialmatch.model import attach_lora
    model=tiny(7)
    x=torch.tensor([[3,4,5,6,7,8]])
    labels=torch.tensor([[-100,-100,-100,6,7,8]])
    expected=model(input_ids=x,labels=labels,use_cache=False).loss
    actual=response_loss(model,x,labels)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
    before=model(input_ids=x,use_cache=False).logits.detach()
    attach_lora(model,last_layers=1)
    after=model(input_ids=x,use_cache=False).logits.detach()
    torch.testing.assert_close(before,after,atol=0,rtol=0)
    response_loss(model,x,labels).backward()
    assert all(p.grad is None for n,p in model.named_parameters() if not n.endswith(('.a','.b')))
    assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for n,p in model.named_parameters() if n.endswith('.b'))

def test_whole_sentence_truncation_without_gold_leakage():
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/base',local_files_only=True)
    row={'note':'\n'.join(f'{i}. '+('A patient has documented symptoms. '*8) for i in range(40)),
         'criterion_type':'inclusion','criterion_text':'Adult patients','label':'included','evidence':[39]}
    ids,sents,info=encode(row,tokenizer,max_length=240)
    changed=dict(row,label='not enough information',evidence=[0])
    assert encode(changed,tokenizer,max_length=240)[0]==ids
    assert sents and info['omitted_sentence_ids'] and info['sentence_ids']==list(range(len(sents)))
    tokens,labels,supervised_info=encode(row,tokenizer,max_length=240,supervised=True)
    assert len(tokens)<=240
    assert supervised_info['omitted_gold_evidence']==[39]
    assert sum(t!=-100 for t in labels)==len(sents)+1
