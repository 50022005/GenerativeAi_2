"""Explicit single-sequence greedy/cache/speculative paths; no generate() call."""
import time
import torch
from transformers import DynamicCache

def forward(model, tokens, cache=None, cached=True, keep=1):
    device = next(model.parameters()).device
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    start = cache.get_seq_length() if cache is not None else 0
    # The attention mask spans past+current positions. RoPE positions are absolute.
    positions = torch.arange(start, start + len(tokens), device=device)
    return model(input_ids=x, past_key_values=cache, use_cache=cached,
                 attention_mask=torch.ones((1, start + len(tokens)), dtype=torch.long, device=device),
                 position_ids=positions.unsqueeze(0), logits_to_keep=keep)

def pick(logits, restrict, step):
    if restrict is not None:
        logits = restrict(logits, step)
    return int(logits.argmax(-1).item())

@torch.inference_mode()
def greedy(model, prompt, max_new_tokens=16, cached=True, restrict=None, eos=None):
    started = time.perf_counter()
    cache = DynamicCache(config=model.config) if cached else None
    out = forward(model, prompt, cache, cached)
    prefill = time.perf_counter() - started
    generated = []
    trace = [out.past_key_values.get_seq_length()] if cached else []
    for step in range(max_new_tokens):
        token = pick(out.logits[:, -1, :], restrict, step)
        generated.append(token)
        if token == eos or len(generated) == max_new_tokens:
            break
        if cached:
            cache = out.past_key_values
            out = forward(model, [token], cache, True)
            trace.append(out.past_key_values.get_seq_length())
        else:
            out = forward(model, prompt + generated, None, False)
    elapsed = time.perf_counter() - started
    return {'tokens': generated, 'prefill_s': prefill, 'decode_s': elapsed - prefill,
            'total_s': elapsed, 'tokens_per_s': len(generated) / elapsed,
            'decode_tokens_per_s': max(0, len(generated)-1) / max(elapsed-prefill, 1e-9),
            'cache_lengths': trace}

@torch.inference_mode()
def speculative(target, draft, prompt, max_new_tokens=16, block_size=4, restrict=None, eos=None):
    """Greedy exactness: accept matching prefix, emit target correction, crop both caches.

    Invariant at each round: caches contain the complete accepted prefix and next
    logits predict its successor. The draft uses the identical tokenizer vocabulary.
    """
    started = time.perf_counter()
    tc, dc = DynamicCache(config=target.config), DynamicCache(config=draft.config)
    to, do = forward(target, prompt, tc), forward(draft, prompt, dc)
    tl, dl = to.logits[:, -1, :], do.logits[:, -1, :]
    prefill = time.perf_counter() - started
    generated, proposed, accepted, rounds, rejections = [], 0, 0, 0, 0
    while len(generated) < max_new_tokens:
        rounds += 1
        base = tc.get_seq_length()
        candidates = []
        initial_tl = tl
        for j in range(min(block_size, max_new_tokens - len(generated))):
            token = pick(dl, restrict, len(generated) + j)
            candidates.append(token)
            do = forward(draft, [token], dc)
            dl = do.logits[:, -1, :]
            if token == eos:
                break
        proposed += len(candidates)
        verified = forward(target, candidates, tc, keep=len(candidates))
        matched = 0
        correction = None
        for j, token in enumerate(candidates):
            logits = initial_tl if j == 0 else verified.logits[:, j-1, :]
            expected = pick(logits, restrict, len(generated) + j)
            if token != expected:
                correction = expected
                rejections += 1
                break
            matched += 1
            if token == eos:
                break
        generated.extend(candidates[:matched])
        accepted += matched
        # Discard every unaccepted KV entry, including a rejected candidate.
        tc.crop(base + matched)
        dc.crop(base + matched)
        if generated and generated[-1] == eos:
            break
        if correction is not None:
            generated.append(correction)
            to, do = forward(target, [correction], tc), forward(draft, [correction], dc)
            tl, dl = to.logits[:, -1, :], do.logits[:, -1, :]
            if correction == eos:
                break
        else:
            tl = verified.logits[:, matched-1, :]
            # All candidates matched; draft cache and dl already correspond to the prefix.
            if len(generated) < max_new_tokens:
                bonus = pick(tl, restrict, len(generated))
                generated.append(bonus)
                to, do = forward(target, [bonus], tc), forward(draft, [bonus], dc)
                tl, dl = to.logits[:, -1, :], do.logits[:, -1, :]
                if bonus == eos:
                    break
        assert tc.get_seq_length() == dc.get_seq_length() == len(prompt) + len(generated)
    elapsed = time.perf_counter() - started
    return {'tokens': generated, 'prefill_s': prefill, 'decode_s': elapsed-prefill,
            'total_s': elapsed, 'tokens_per_s': len(generated)/elapsed,
            'proposed': proposed, 'accepted': accepted, 'acceptance_rate': accepted/max(1, proposed),
            'rounds': rounds, 'rejections': rejections}
