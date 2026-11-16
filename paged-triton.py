import math
import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import triton
    import triton.language as tl
    HAS_TRITON = torch.cuda.is_available()
except ImportError:
    HAS_TRITON = False

MODEL = "Qwen/Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

TEMPERATURE = 0.0
TOP_K = 50
TOP_P = 0.95
SEED = 0

PROMPT = "explain transformer architecture in detail"
MAX_NEW_TOKENS = 4700

BLOCK_SIZE = 16
WARMUP_TOKENS = 8
SAFETY_MARGIN_MB = 256


if HAS_TRITON:

    @triton.jit(do_not_specialize=["seq_len"])
    def paged_decode_kernel(
        Q, K, V, O, TABLE, seq_len, scale,
        stride_qh, stride_kb, stride_kh, stride_kt,
        stride_vb, stride_vh, stride_vt, stride_oh,
        GROUP: tl.constexpr, BLOCK: tl.constexpr, HEAD_DIM: tl.constexpr,
    ):
        h = tl.program_id(0)
        kvh = h // GROUP
        offs_d = tl.arange(0, HEAD_DIM)
        offs_t = tl.arange(0, BLOCK)
        q = tl.load(Q + h * stride_qh + offs_d).to(tl.float32)
        m = tl.zeros([1], dtype=tl.float32) - float("inf")
        l = tl.zeros([1], dtype=tl.float32)
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
        num_blocks = tl.cdiv(seq_len, BLOCK)
        for b in range(0, num_blocks):
            blk = tl.load(TABLE + b).to(tl.int64)
            tok = b * BLOCK + offs_t
            valid = tok < seq_len
            k_ptrs = K + blk * stride_kb + kvh * stride_kh + offs_t[:, None] * stride_kt + offs_d[None, :]
            k = tl.load(k_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
            s = tl.sum(k * q[None, :], axis=1) * scale
            s = tl.where(valid, s, float("-inf"))
            m_new = tl.maximum(m, tl.max(s, axis=0))
            p = tl.exp(s - m_new)
            alpha = tl.exp(m - m_new)
            l = l * alpha + tl.sum(p, axis=0)
            v_ptrs = V + blk * stride_vb + kvh * stride_vh + offs_t[:, None] * stride_vt + offs_d[None, :]
            v = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            m = m_new
        out = acc / l
        tl.store(O + h * stride_oh + offs_d, out)


def triton_paged_decode(q, k_pool, v_pool, table, seq_len):
    heads, head_dim = q.shape[1], q.shape[-1]
    kv_heads, block = k_pool.shape[1], k_pool.shape[2]
    q2 = q[0, :, 0, :].contiguous()
    out = torch.empty_like(q2)
    paged_decode_kernel[(heads,)](
        q2, k_pool, v_pool, out, table, seq_len, head_dim ** -0.5,
        q2.stride(0), k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
        v_pool.stride(0), v_pool.stride(1), v_pool.stride(2), out.stride(0),
        GROUP=heads // kv_heads, BLOCK=block, HEAD_DIM=head_dim,
    )
    return out.view(1, heads, 1, head_dim)


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def available_memory():
    if DEVICE == "cuda":
        return torch.cuda.mem_get_info()[0]
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


class ContiguousCache:
    def __init__(self, layers):
        self.k = [None] * layers
        self.v = [None] * layers
        self.length = 0

    def begin(self, n):
        pass

    def update(self, i, k, v):
        if self.k[i] is None:
            self.k[i], self.v[i] = k, v
        else:
            self.k[i] = torch.cat([self.k[i], k], dim=-2)
            self.v[i] = torch.cat([self.v[i], v], dim=-2)
        return self.k[i], self.v[i]

    def advance(self, n):
        self.length += n

    def blocks_used(self):
        return None


class BlockAllocator:
    def __init__(self, num_blocks):
        self.free = list(range(num_blocks - 1, -1, -1))

    def alloc(self):
        if not self.free:
            raise RuntimeError("KV block pool exhausted")
        return self.free.pop()


class PagedCache:
    def __init__(self, layers, kv_heads, head_dim, num_blocks, block_size, dtype, device):
        shape = (num_blocks, kv_heads, block_size, head_dim)
        self.kp = [torch.empty(shape, dtype=dtype, device=device) for _ in range(layers)]
        self.vp = [torch.empty(shape, dtype=dtype, device=device) for _ in range(layers)]
        self.allocator = BlockAllocator(num_blocks)
        self.block_size = block_size
        self.device = device
        self.table = []
        self.table_t = None
        self.length = 0
        self.blk = None
        self.off = None

    def begin(self, n):
        need = math.ceil((self.length + n) / self.block_size) - len(self.table)
        for _ in range(need):
            self.table.append(self.allocator.alloc())
        if need > 0 or self.table_t is None:
            self.table_t = torch.tensor(self.table, dtype=torch.long, device=self.device)
        pos = torch.arange(self.length, self.length + n, device=self.device)
        self.blk = self.table_t[pos // self.block_size]
        self.off = pos % self.block_size

    def write(self, i, k, v):
        self.kp[i][self.blk, :, self.off] = k[0].transpose(0, 1)
        self.vp[i][self.blk, :, self.off] = v[0].transpose(0, 1)

    def update(self, i, k, v):
        self.write(i, k, v)
        total = self.length + k.shape[-2]
        heads, hd = k.shape[1], k.shape[-1]
        kk = self.kp[i][self.table_t].permute(1, 0, 2, 3).reshape(heads, -1, hd)[:, :total]
        vv = self.vp[i][self.table_t].permute(1, 0, 2, 3).reshape(heads, -1, hd)[:, :total]
        return kk.unsqueeze(0), vv.unsqueeze(0)

    def advance(self, n):
        self.length += n

    def blocks_used(self):
        return len(self.table)


class TritonPagedCache(PagedCache):
    def update(self, i, k, v):
        self.write(i, k, v)
        if k.shape[-2] > 1:
            return k, v
        return None, None

    def attend(self, i, q):
        return triton_paged_decode(q, self.kp[i], self.vp[i], self.table_t, self.length + 1)


class Qwen3:
    def __init__(self, state_dict, config):
        self.w = state_dict
        self.hidden = config.hidden_size
        self.layers = config.num_hidden_layers
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.eps = config.rms_norm_eps
        self.max_positions = config.max_position_embeddings
        theta = config.rope_parameters["rope_theta"]
        self.inv_freq = 1.0 / theta ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
        )

    def norm(self, x, weight):
        return F.rms_norm(x, (x.shape[-1],), weight=weight, eps=self.eps)

    def rope(self, x, offset):
        pos = torch.arange(offset, offset + x.shape[-2], device=x.device, dtype=torch.float32)
        angles = pos[:, None] * self.inv_freq.to(x.device)[None, :]
        emb = torch.cat([angles, angles], dim=-1)
        cos, sin = emb.cos()[None, None], emb.sin()[None, None]
        x1, x2 = x.chunk(2, dim=-1)
        return x * cos.to(x.dtype) + torch.cat([-x2, x1], dim=-1) * sin.to(x.dtype)

    def forward(self, input_ids, cache):
        w = self.w
        x = F.embedding(input_ids, w["model.embed_tokens.weight"])
        batch, seq_len, _ = x.shape
        past_len = cache.length
        total = past_len + seq_len
        cache.begin(seq_len)
        mask = None
        if seq_len > 1:
            mask = torch.tril(torch.ones(total, total, device=x.device, dtype=torch.bool))[past_len:][None, None]
        for i in range(self.layers):
            layer = f"model.layers.{i}"
            residual = x
            x = self.norm(x, w[f"{layer}.input_layernorm.weight"])
            q = self.rope(
                self.norm(
                    F.linear(x, w[f"{layer}.self_attn.q_proj.weight"]).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2),
                    w[f"{layer}.self_attn.q_norm.weight"],
                ), past_len,
            )
            k = self.rope(
                self.norm(
                    F.linear(x, w[f"{layer}.self_attn.k_proj.weight"]).view(batch, seq_len, self.kv_heads, self.head_dim).transpose(1, 2),
                    w[f"{layer}.self_attn.k_norm.weight"],
                ), past_len,
            )
            v = F.linear(x, w[f"{layer}.self_attn.v_proj.weight"]).view(batch, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
            k_all, v_all = cache.update(i, k, v)
            if k_all is None:
                attn = cache.attend(i, q)
            else:
                attn = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=mask, enable_gqa=True)
            attn = attn.transpose(1, 2).reshape(batch, seq_len, -1)
            x = residual + F.linear(attn, w[f"{layer}.self_attn.o_proj.weight"])
            residual = x
            x = self.norm(x, w[f"{layer}.post_attention_layernorm.weight"])
            gate = F.linear(x, w[f"{layer}.mlp.gate_proj.weight"])
            up = F.linear(x, w[f"{layer}.mlp.up_proj.weight"])
            x = residual + F.linear(F.silu(gate) * up, w[f"{layer}.mlp.down_proj.weight"])
        cache.advance(seq_len)
        x = self.norm(x[:, -1:], w["model.norm.weight"])
        return F.linear(x, w["model.embed_tokens.weight"])


def sample(logits, temperature, top_k, top_p):
    if temperature is None or temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, _ = torch.topk(logits, k, dim=-1)
        logits = logits.masked_fill(logits < vals[..., -1:], float("-inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = probs.cumsum(dim=-1)
        remove = cumprobs - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def check_limits(model, prompt_len, max_new):
    total = prompt_len + max_new
    if total > model.max_positions:
        raise RuntimeError(
            f"Requested length {total} tokens exceeds the model context limit of {model.max_positions}"
        )
    elem = torch.empty(0, dtype=DTYPE).element_size()
    per_token = 2 * model.layers * model.kv_heads * model.head_dim * elem
    paged = math.ceil(total / BLOCK_SIZE) * BLOCK_SIZE * per_token
    transient = 2 * model.kv_heads * model.head_dim * total * elem
    contiguous = total * per_token + transient
    needed = max(paged, contiguous) + SAFETY_MARGIN_MB * 1024 ** 2
    free = available_memory()
    if free is not None and needed > free:
        raise RuntimeError(
            f"MAX_NEW_TOKENS={max_new} needs about {needed / 1024 ** 2:.0f} MB for KV cache and buffers, "
            f"but only {free / 1024 ** 2:.0f} MB is free on {DEVICE.upper()}"
        )
    return per_token, needed, free


def make_cache(kind, model, total_tokens):
    if kind == "contiguous":
        return ContiguousCache(model.layers)
    num_blocks = math.ceil(total_tokens / BLOCK_SIZE)
    cls = TritonPagedCache if kind == "triton" else PagedCache
    return cls(model.layers, model.kv_heads, model.head_dim, num_blocks, BLOCK_SIZE, DTYPE, DEVICE)


def generate(model, input_ids, cache, max_new, eos_id, base_mem):
    tokens = []
    sync()
    start = time.perf_counter()
    with torch.no_grad():
        logits = model.forward(input_ids, cache)
        nxt = sample(logits[:, -1], TEMPERATURE, TOP_K, TOP_P)
        sync()
        ttft = time.perf_counter() - start
        tokens.append(nxt)
        while len(tokens) < max_new and nxt.item() != eos_id:
            logits = model.forward(nxt, cache)
            nxt = sample(logits[:, -1], TEMPERATURE, TOP_K, TOP_P)
            tokens.append(nxt)
        sync()
    latency = time.perf_counter() - start
    n = len(tokens)
    decode_time = latency - ttft
    peak = (torch.cuda.max_memory_allocated() - base_mem) / 1024 ** 2 if DEVICE == "cuda" else None
    metrics = {
        "tokens": n,
        "ttft_ms": ttft * 1000,
        "latency_s": latency,
        "throughput_tps": n / latency,
        "decode_tps": (n - 1) / decode_time if n > 1 and decode_time > 0 else float("nan"),
        "tpot_ms": decode_time / (n - 1) * 1000 if n > 1 else float("nan"),
        "peak_mb": peak,
        "blocks": cache.blocks_used(),
    }
    return torch.cat(tokens, dim=1), metrics


def run(kind, model, input_ids, max_new, eos_id, seed):
    total = input_ids.shape[1] + max_new
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated()
    else:
        base_mem = 0
    cache = make_cache(kind, model, total)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)
    out, metrics = generate(model, input_ids, cache, max_new, eos_id, base_mem)
    del cache
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out, metrics


def compare_outputs(ref, other):
    n = min(ref.shape[1], other.shape[1])
    diff = (ref[0, :n] != other[0, :n]).nonzero()
    if len(diff) == 0:
        if ref.shape[1] == other.shape[1]:
            return "identical"
        return f"equal prefix, lengths {ref.shape[1]} vs {other.shape[1]}"
    return f"diverges at token {diff[0].item()}"


def cell(value, spec):
    if value is None or value != value:
        return "n/a"
    return format(value, spec)


def delta_cell(x, y, lower_better):
    if x is None or y is None or x != x or y != y or x == 0:
        return "n/a"
    change = (y - x) / x * 100
    if abs(change) < 0.05:
        return "0.0% (equal)"
    better = (change < 0) == lower_better
    return f"{change:+.1f}% ({'better' if better else 'worse'})"


def print_benchmarks(names, results):
    metrics = [
        ("Tokens generated", "tokens", "d", None),
        ("TTFT (ms)", "ttft_ms", ".2f", True),
        ("Total latency (s)", "latency_s", ".3f", True),
        ("Throughput (tok/s)", "throughput_tps", ".2f", False),
        ("Decode throughput (tok/s)", "decode_tps", ".2f", False),
        ("Time per output token (ms)", "tpot_ms", ".3f", True),
        ("Peak extra memory (MB)", "peak_mb", ".1f", True),
    ]
    pairs = [(0, 1), (0, 2), (1, 2)]
    pairs = [(a, b) for a, b in pairs if b < len(names)]
    short = {"Non-paged": "Non", "Torch-paged": "Torch", "Triton-paged": "Triton"}
    width = 28 + 14 * len(names) + 22 * len(pairs)
    print("\n" + "=" * width)
    print("Benchmarks")
    print("=" * width)
    header = f"{'Metric':<28}" + "".join(f"{n:>14}" for n in names)
    header += "".join(f"{short[names[b]] + ' vs ' + short[names[a]]:>22}" for a, b in pairs)
    print(header)
    print("-" * width)
    for label, key, spec, lower in metrics:
        row = f"{label:<28}" + "".join(f"{cell(r[key], spec):>14}" for r in results)
        if lower is not None:
            row += "".join(f"{delta_cell(results[a][key], results[b][key], lower):>22}" for a, b in pairs)
        print(row)
    print("-" * width)
    for n, r in zip(names, results):
        if r["blocks"] is not None:
            print(f"KV blocks used ({n}): {r['blocks']} x {BLOCK_SIZE} tokens")
    print("=" * width)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEVICE)
    model = Qwen3(hf.state_dict(), hf.config)

    if HAS_TRITON and (model.head_dim & (model.head_dim - 1) or BLOCK_SIZE & (BLOCK_SIZE - 1)):
        raise RuntimeError("Triton kernel needs power-of-two head_dim and BLOCK_SIZE")

    input_ids = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True, return_tensors="pt", return_dict=False, enable_thinking=False,
    ).to(DEVICE)

    per_token, needed, free = check_limits(model, input_ids.shape[1], MAX_NEW_TOKENS)

    modes = [("Non-paged", "contiguous"), ("Torch-paged", "paged")]
    if HAS_TRITON:
        modes.append(("Triton-paged", "triton"))

    for _, kind in modes:
        run(kind, model, input_ids, WARMUP_TOKENS, tok.eos_token_id, SEED)

    outputs, results = [], []
    for _, kind in modes:
        out, m = run(kind, model, input_ids, MAX_NEW_TOKENS, tok.eos_token_id, SEED)
        outputs.append(out)
        results.append(m)

    names = [n for n, _ in modes]

    print("\nDevice        :", DEVICE)
    print("Model         :", MODEL)
    print(f"Sampling      : T={TEMPERATURE}, top_k={TOP_K}, top_p={TOP_P}, seed={SEED}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"KV per token  : {per_token / 1024:.1f} KB | est. need {needed / 1024 ** 2:.0f} MB"
          + (f" of {free / 1024 ** 2:.0f} MB free" if free is not None else ""))
    print("\nPrompt :", PROMPT)
    print("Response:", tok.decode(outputs[-1][0], skip_special_tokens=True))

    print_benchmarks(names, results)
    for n, o in zip(names[1:], outputs[1:]):
        print(f"Output of {n} vs Non-paged: {compare_outputs(outputs[0], o)}")


if __name__ == "__main__":
    main()
