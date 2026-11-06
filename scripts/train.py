"""CPU LoRA, response-only next-token cross entropy; no test set access."""
import argparse
import json
import random
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from trialmatch.data import ROOT, read_split, encode
from trialmatch.model import load, attach_lora, save_adapter, setup

def response_loss(model, tokens, labels):
    # Project only response positions, avoiding a huge prompt x vocabulary tensor.
    # n+1 final hidden positions predict the n response tokens; discard the last logit.
    n = int((labels != -100).sum())
    output = model(input_ids=tokens, use_cache=False, logits_to_keep=n+1)
    return torch.nn.functional.cross_entropy(output.logits[:, :-1, :].reshape(-1, output.logits.shape[-1]),
                                              labels[:, -n:].reshape(-1))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--max-length', type=int, default=1024)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--limit', type=int, default=0, help='Smoke test only; not final training')
    p.add_argument('--accumulation', type=int, default=8)
    args = p.parse_args()
    if (ROOT/'results/frozen_protocol.json').exists():
        p.error('This study is frozen. Run scripts/new_study.py to archive it before retraining.')
    setup()
    model, tokenizer = load(adapter=False)
    attach_lora(model)
    model.config.use_cache = False
    train, val = read_split('train'), read_split('validation')
    if args.limit:
        train, val = train[:args.limit], val[:min(4, args.limit)]
    train = [encode(r, tokenizer, args.max_length, True) for r in train]
    val = [encode(r, tokenizer, args.max_length, True) for r in val]
    config = dict(vars(args), last_layers=6, rank=8, alpha=16, seed=42,
                  base=json.loads((ROOT/'models/base/revision.json').read_text()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  train_count=len(train), validation_count=len(val),
                  objective='response-only autoregressive cross entropy; label plus evidence bitmap',
                  precision='float32', device='cpu', threads=torch.get_num_threads())
    (ROOT/'results').mkdir(exist_ok=True)
    trunc = {name: {'records': len(rows), 'truncated_records': sum(bool(r[2]['omitted_sentence_ids']) for r in rows),
                   'omitted_gold_evidence_count': sum(len(r[2]['omitted_gold_evidence']) for r in rows),
                   'max_input_tokens': max(r[2]['input_tokens'] for r in rows)}
             for name, rows in [('train', train), ('validation', val)]}
    (ROOT/'results/truncation.json').write_text(json.dumps(trunc, indent=2))
    (ROOT/'results/training_config.json').write_text(json.dumps(config, indent=2))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    best = float('inf')
    started = time.perf_counter()
    log_path = ROOT/'results/training.jsonl'
    log_path.write_text('')
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(train)))
        random.Random(42+epoch).shuffle(order)
        optimizer.zero_grad()
        losses = []
        for step, idx in enumerate(order):
            tokens, labels, info = train[idx]
            x, y = torch.tensor([tokens]), torch.tensor([labels])
            loss = response_loss(model, x, y)
            group_size = min(args.accumulation, len(order)-(step//args.accumulation)*args.accumulation)
            (loss/group_size).backward()
            losses.append(float(loss.detach()))
            if (step+1) % args.accumulation == 0 or step+1 == len(order):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad()
            if (step+1) % 10 == 0 or step == 0:
                record = {'epoch': epoch+1, 'step': step+1, 'train_loss': sum(losses[-10:])/len(losses[-10:]),
                          'elapsed_s': time.perf_counter()-started}
                with log_path.open('a') as f:
                    f.write(json.dumps(record)+'\n')
                print(json.dumps(record), flush=True)
        model.eval()
        with torch.inference_mode():
            # Token-weighted validation objective; used only to select the checkpoint.
            numerator, denominator = 0., 0
            for tokens, labels, _ in val:
                n = sum(t != -100 for t in labels)
                loss = response_loss(model, torch.tensor([tokens]), torch.tensor([labels]))
                numerator += float(loss) * n
                denominator += n
        val_loss = numerator / denominator
        record = {'epoch': epoch+1, 'mean_train_loss': sum(losses)/len(losses),
                  'validation_loss': val_loss, 'elapsed_s': time.perf_counter()-started}
        with log_path.open('a') as f:
            f.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
        config.update(selected_epoch=epoch+1, validation_loss=val_loss, training_elapsed_s=record['elapsed_s'])
        save_adapter(model, config, ROOT/'models/last_adapter')
        if val_loss < best:
            best = val_loss
            save_adapter(model, config, ROOT/'models/adapter')
    print('Training complete; checkpoint selected by validation loss only.', flush=True)

if __name__ == '__main__':
    main()
