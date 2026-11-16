# python pagedattention.py --prompt-lengths 32 128 512 --decode-lengths 32 128 512

from __future__ import annotations

import argparse
import csv
import gc
import math
import statistics
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from base import DEVICE, MODEL, Qwen3 as BaselineQwen3, sample
except ImportError:
    from .base import DEVICE, MODEL, Qwen3 as BaselineQwen3, sample


class PagedKVCache:
    """Single-sequence block KV cache; blocks are retained and reused on reset."""
    def __init__(self, layers, kv_heads, head_dim, block_size, device, dtype):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.layers, self.kv_heads, self.head_dim = layers, kv_heads, head_dim
        self.block_size, self.device, self.dtype = block_size, device, dtype
        self.keys, self.values, self.free = [], [], []
        self.block_table = []
        self.length = 0

    def _allocate(self):
        if self.free:
            return self.free.pop()
        shape = (self.layers, self.kv_heads, self.block_size, self.head_dim)
        self.keys.append(torch.empty(shape, device=self.device, dtype=self.dtype))
        self.values.append(torch.empty_like(self.keys[-1]))
        return len(self.keys) - 1

    @property
    def allocated_bytes(self):
        return (len(self.keys) - len(self.free)) * 2 * self.layers * self.kv_heads * self.block_size * self.head_dim * torch.empty((), dtype=self.dtype).element_size()

    @property
    def reserved_bytes(self):
        return len(self.keys) * 2 * self.layers * self.kv_heads * self.block_size * self.head_dim * torch.empty((), dtype=self.dtype).element_size()

    def append(self, layer, key, value):
        if key.shape[0] != 1:
            raise ValueError("PagedKVCache currently supports batch size 1")
        if key.shape != value.shape or key.shape[1] != self.kv_heads or key.shape[-1] != self.head_dim:
            raise ValueError("K/V shape does not match cache configuration")
        start, count, cursor = self.length, key.shape[-2], 0
        needed = math.ceil((start + count) / self.block_size)
        while len(self.block_table) < needed:
            self.block_table.append(self._allocate())
        while cursor < count:
            pos = start + cursor
            logical_block, offset = divmod(pos, self.block_size)
            n = min(count - cursor, self.block_size - offset)
            block = self.block_table[logical_block]
            self.keys[block][layer, :, offset:offset+n].copy_(key[0, :, cursor:cursor+n])
            self.values[block][layer, :, offset:offset+n].copy_(value[0, :, cursor:cursor+n])
            cursor += n
        if layer == self.layers - 1:
            self.length += count

    def get(self, layer, length=None):
        # During a forward pass, all layers append at the old logical length;
        # expose the newly written tokens before committing the new length.
        remaining, keys, values = (self.length if length is None else length), [], []
        for block in self.block_table:
            n = min(remaining, self.block_size)
            if n <= 0:
                break
            keys.append(self.keys[block][layer, :, :n])
            values.append(self.values[block][layer, :, :n])
            remaining -= n
        if not keys:
            shape = (1, self.kv_heads, 0, self.head_dim)
            empty = torch.empty(shape, device=self.device, dtype=self.dtype)
            return empty, empty
        return torch.cat(keys, dim=1).unsqueeze(0), torch.cat(values, dim=1).unsqueeze(0)

    def reset(self):
        self.free.extend(self.block_table)
        self.block_table.clear()
        self.length = 0


class PagedQwen3(BaselineQwen3):
    """Same weights and forward math as base.Qwen3 with block KV storage."""
    def new_cache(self, device, dtype, block_size=16):
        return PagedKVCache(self.layers, self.kv_heads, self.head_dim, block_size, device, dtype)

    def forward(self, input_ids, past=None):
        if input_ids.shape[0] != 1:
            raise ValueError("PagedQwen3 currently supports batch size 1")
        w = self.w
        x = F.embedding(input_ids, w["model.embed_tokens.weight"])
        batch, seq_len, _ = x.shape
        past_len = 0 if past is None else past.length
        cache = self.new_cache(x.device, x.dtype) if past is None else past
        total = past_len + seq_len
        # SDPA's is_causal uses upper-left alignment. Explicit absolute query
        # positions let each decode query attend to the entire cached prefix.
        query_positions = past_len + torch.arange(seq_len, device=x.device)
        key_positions = torch.arange(total, device=x.device)
        mask = (key_positions[None, :] <= query_positions[:, None])[None, None]
        for i in range(self.layers):
            layer = f"model.layers.{i}"
            residual = x
            x = self.norm(x, w[f"{layer}.input_layernorm.weight"])
            q = self.rope(self.norm(F.linear(x, w[f"{layer}.self_attn.q_proj.weight"]).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2), w[f"{layer}.self_attn.q_norm.weight"]), past_len)
            k = self.rope(self.norm(F.linear(x, w[f"{layer}.self_attn.k_proj.weight"]).view(batch, seq_len, self.kv_heads, self.head_dim).transpose(1, 2), w[f"{layer}.self_attn.k_norm.weight"]), past_len)
            v = F.linear(x, w[f"{layer}.self_attn.v_proj.weight"]).view(batch, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
            cache.append(i, k, v)
            all_k, all_v = cache.get(i, total)
            attn = F.scaled_dot_product_attention(q, all_k, all_v, attn_mask=mask, enable_gqa=True)
            attn = attn.transpose(1, 2).reshape(batch, seq_len, -1)
            x = residual + F.linear(attn, w[f"{layer}.self_attn.o_proj.weight"])
            residual = x
            x = self.norm(x, w[f"{layer}.post_attention_layernorm.weight"])
            gate = F.linear(x, w[f"{layer}.mlp.gate_proj.weight"])
            up = F.linear(x, w[f"{layer}.mlp.up_proj.weight"])
            x = residual + F.linear(F.silu(gate) * up, w[f"{layer}.mlp.down_proj.weight"])
        x = self.norm(x, w["model.norm.weight"])
        return F.linear(x, w["model.embed_tokens.weight"]), cache


@dataclass
class Result:
    implementation: str
    prompt_id: int
    prompt_tokens: int
    generated_tokens: int
    prefill_ms: float
    decode_ms: float
    end_to_end_ms: float
    prefill_tokens_per_second: float
    decode_tokens_per_second: float
    end_to_end_tokens_per_second: float
    time_to_first_token_ms: float
    inter_token_latency_ms: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    kv_cache_bytes: int
    kv_cache_reserved_bytes: int


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(model, ids, max_new, eos, label, device, prompt_id):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.inference_mode():
        tick = time.perf_counter()
        logits, cache = model.forward(ids)
        sync(device)
        prefill = time.perf_counter() - tick
        token = sample(logits[:, -1], 0.0, None)
        ttft = time.perf_counter() - start
        generated, token_times = 0, []
        for _ in range(max_new):
            generated += 1
            if int(token.item()) == eos:
                break
            tick = time.perf_counter()
            logits, cache = model.forward(token, cache)
            token = sample(logits[:, -1], 0.0, None)
            sync(device)
            token_times.append(time.perf_counter() - tick)
    total = time.perf_counter() - start
    decode, decode_count = max(total - prefill, 0.0), max(generated - 1, 0)
    if isinstance(cache, PagedKVCache):
        kv, kv_reserved = cache.allocated_bytes, cache.reserved_bytes
    else:
        kv = sum((k.numel() + v.numel()) * k.element_size() for k, v in cache)
        kv_reserved = kv
    return Result(label, prompt_id, ids.shape[1], generated, prefill*1e3, decode*1e3, total*1e3,
                  ids.shape[1]/prefill if prefill else 0, decode_count/decode if decode and decode_count else 0,
                  generated/total if total else 0, ttft*1e3,
                  statistics.mean(token_times)*1e3 if token_times else 0,
                  torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                  torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
                  kv, kv_reserved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--prompts", nargs="*", default=["Explain how a transformer generates text."])
    parser.add_argument("--prompt-lengths", nargs="*", type=int, default=[32, 128, 512])
    parser.add_argument("--decode-lengths", nargs="*", type=int, default=[32, 128, 512])
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default="pagedattention_benchmarks.csv")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("--repeats must be >= 1 and --warmup must be >= 0")
    device, dtype = torch.device(DEVICE), getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    weights, config = hf.state_dict(), hf.config
    baseline, paged = BaselineQwen3(weights, config), PagedQwen3(weights, config)
    del hf
    gc.collect()

    rows = []
    print(f"device={device} dtype={args.dtype} model={args.model} block_size={args.block_size}")
    for prompt_id, prompt in enumerate(args.prompts, start=1):
        tokenized = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        for prompt_len in args.prompt_lengths:
            if prompt_len <= 0:
                continue
            ids = tokenized[:, :prompt_len] if tokenized.shape[1] >= prompt_len else tokenized.repeat(1, math.ceil(prompt_len/tokenized.shape[1]))[:, :prompt_len]
            for decode_len in args.decode_lengths:
                if decode_len <= 0:
                    continue
                pair = []
                for label, model in (("base", baseline), ("paged", paged)):
                    for _ in range(args.warmup):
                        measure(model, ids, min(decode_len, 2), tokenizer.eos_token_id, label, device, prompt_id)
                    samples = [measure(model, ids, decode_len, tokenizer.eos_token_id, label, device, prompt_id) for _ in range(args.repeats)]
                    row = asdict(samples[-1])
                    for name in ("prefill_ms", "decode_ms", "end_to_end_ms", "prefill_tokens_per_second", "decode_tokens_per_second", "end_to_end_tokens_per_second", "time_to_first_token_ms", "inter_token_latency_ms"):
                        row[name] = statistics.median(getattr(item, name) for item in samples)
                    rows.append(row)
                    pair.append(row)
                b, p = pair
                prefill_change = (p["prefill_tokens_per_second"] / b["prefill_tokens_per_second"] - 1) * 100 if b["prefill_tokens_per_second"] else 0
                decode_change = (p["decode_tokens_per_second"] / b["decode_tokens_per_second"] - 1) * 100 if b["decode_tokens_per_second"] else 0
                print(f"prompt_id={prompt_id} prompt={ids.shape[1]:4d} decode={decode_len:4d} | prefill tok/s {b['prefill_tokens_per_second']:.1f} -> {p['prefill_tokens_per_second']:.1f} ({prefill_change:+.1f}%) | decode tok/s {b['decode_tokens_per_second']:.2f} -> {p['decode_tokens_per_second']:.2f} ({decode_change:+.1f}%) | peak alloc MiB {b['peak_allocated_bytes']/2**20:.1f} -> {p['peak_allocated_bytes']/2**20:.1f} | KV MiB {b['kv_cache_bytes']/2**20:.2f} -> {p['kv_cache_bytes']/2**20:.2f}")
    if rows:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} measurements to {args.output}")
    #print("CSV has one row per implementation, prompt_id, prompt length, and decode length; --repeats are summarized by the median.")
    #print("Metrics: prefill/decode/end-to-end latency and throughput, time to first token, inter-token latency, peak CUDA allocated/reserved memory, and KV cache allocated/reserved bytes.")
    #print("Greedy decoding is used for repeatable timing. This reference gathers paged KV into contiguous tensors before SDPA; it does not measure a fused paged-attention kernel.")


if __name__ == "__main__":
    main()
