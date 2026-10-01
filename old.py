"""import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading tokenizer and reference model...")
MODEL = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(MODEL)
device = "cuda" if torch.cuda.is_available() else "cpu"

hf_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device)
sd = hf_model.state_dict()
config = hf_model.config

print(hf_model)
print("-----------------------------------------------------------------------------")
print(config)
"""



#print("=========================================================")

import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer and reference model...")

tok = AutoTokenizer.from_pretrained(MODEL)

hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float32,
).to(device)

sd = hf_model.state_dict()
config = hf_model.config

# print(hf_model)
# print("-----------------------------------------------------------------------------")
# print(config)

hidden = config.hidden_size
layers = config.num_hidden_layers
heads = config.num_attention_heads
kv_heads = config.num_key_value_heads
head_dim = config.head_dim

repeat_factor = heads // kv_heads
rope_theta = config.rope_parameters["rope_theta"]
rms_norm_eps = config.rms_norm_eps


def rope(x):
    # x: [B, H, S, D]
    d = x.shape[-1]
    freqs = 1.0 / (
        rope_theta
        ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    )
    pos = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
    angles = pos[:, None] * freqs[None, :]          # [S, D/2]
    emb = torch.cat([angles, angles], dim=-1)       # [S, D]
    cos = emb.cos()[None, None].to(x.dtype)
    sin = emb.sin()[None, None].to(x.dtype)

    def rotate_half(t):
        t1, t2 = t.chunk(2, dim=-1)
        return torch.cat([-t2, t1], dim=-1)

    return x * cos + rotate_half(x) * sin


def forward(input_ids):
    x = F.embedding(
        input_ids,
        sd["model.embed_tokens.weight"],
    )

    batch, seq_len, _ = x.shape
    
    k_cache, v_cache = None, None

    # Determine total context length from cache or current input
    if k_cache is not None:
        kv_len = k_cache.shape[-2] + seq_len
    else:
        kv_len = seq_len

    # Mask shape: [1, 1, q_len, kv_len]
    # During generation: q_len=1, kv_len=growing
    # During prefill: q_len=seq_len, kv_len=seq_len
    mask = torch.tril(torch.ones(kv_len, kv_len, device=x.device, dtype=torch.bool)
    )[-seq_len:, :]  # Slice to get only rows for current queries

    """
    mask = torch.tril(
        torch.ones(
            seq_len,
            seq_len,
            device=x.device,
            dtype=torch.bool,
        )
    )[None, None, :, :]"""
    
    # k_cache, v_cache = None, None

    for i in range(layers):
        layer = f"model.layers.{i}"

        # Attention input normalization
        residual = x

        x = F.rms_norm(
            x,
            (hidden,),
            weight=sd[f"{layer}.input_layernorm.weight"],
            eps=rms_norm_eps,
        )

        # Query
        q = F.linear(
            x,
            sd[f"{layer}.self_attn.q_proj.weight"],
        )

        q = q.view(
            batch,
            seq_len,
            heads,
            head_dim,
        ).transpose(1, 2)

        # QK-Norm
        q = F.rms_norm(
            q,
            (head_dim,),
            weight=sd[f"{layer}.self_attn.q_norm.weight"],
            eps=rms_norm_eps,
        )

        q = rope(q)

        # Key
        k = F.linear(
            x,
            sd[f"{layer}.self_attn.k_proj.weight"],
        )

        k = k.view(
            batch,
            seq_len,
            kv_heads,
            head_dim,
        ).transpose(1, 2)

        # QK-Norm
        k = F.rms_norm(
            k,
            (head_dim,),
            weight=sd[f"{layer}.self_attn.k_norm.weight"],
            eps=rms_norm_eps,
        )

        k = rope(k)

        # Value
        v = F.linear(
            x,
            sd[f"{layer}.self_attn.v_proj.weight"],
        )

        v = v.view(
            batch,
            seq_len,
            kv_heads,
            head_dim,
        ).transpose(1, 2)

        # Grouped-query attention
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)
        
        if k_cache is None:
            k_cache, v_cache = k, v
        else:
            k_cache = torch.cat([k_cache, k],dim=-2)
            v_cache = torch.cat([v_cache, v],dim=-2)
        k, v = k_cache, v_cache
        
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(head_dim)
        scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        attention = weights @ v

        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(batch, seq_len, heads * head_dim)

        attention = F.linear(
            attention,
            sd[f"{layer}.self_attn.o_proj.weight"],
        )

        x = residual + attention

        # MLP
        residual = x

        x = F.rms_norm(
            x,
            (hidden,),
            weight=sd[f"{layer}.post_attention_layernorm.weight"],
            eps=rms_norm_eps,
        )

        gate = F.linear(
            x,
            sd[f"{layer}.mlp.gate_proj.weight"],
        )

        up = F.linear(
            x,
            sd[f"{layer}.mlp.up_proj.weight"],
        )

        down = F.linear(
            F.silu(gate) * up,
            sd[f"{layer}.mlp.down_proj.weight"],
        )

        x = residual + down

    # Final normalization
    x = F.rms_norm(
        x,
        (hidden,),
        weight=sd["model.norm.weight"],
        eps=rms_norm_eps,
    )

    # Tied embedding/language-model head
    logits = F.linear(
        x,
        sd["model.embed_tokens.weight"],
    )

    return logits


prompt = "explain transformer architecture"
messages = [{"role": "user", "content": prompt}]
input_ids = tok.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=False,
    enable_thinking=False,   # Qwen3-specific: turn off  thinking block
).to(device)

generated_ids = input_ids.clone()

max_new_tokens = 100

with torch.no_grad():
    for _ in range(max_new_tokens):
        logits = forward(generated_ids)

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        generated_ids = torch.cat(
            [generated_ids, next_token],
            dim=1,
        )

        if next_token.item() == tok.eos_token_id:
            break

output = tok.decode(
    generated_ids[0],
    skip_special_tokens=True,
)

print()
print("Prompt:", prompt)
print("Output:", output)

"""

"""
print("--------------=-=-=-=-=-=-=-=-=-=-=")
out = hf_model.generate(input_ids, max_new_tokens=30, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
print("-----------------------------------")
with torch.no_grad():
    out = hf_model.generate(
        input_ids,                # same raw ids as your loop
        max_new_tokens=30,
        do_sample=False,
    )
print("HF:", tok.decode(out[0], skip_special_tokens=True))
