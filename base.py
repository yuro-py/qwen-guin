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


def rope_with_offset(x, offset):
    # x: [B, H, S, D]
    d = x.shape[-1]
    freqs = 1.0 / (
        rope_theta
        ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    )
    pos = torch.arange(offset, offset + x.shape[-2], device=x.device, dtype=torch.float32)
    angles = pos[:, None] * freqs[None, :]          # [S, D/2]
    emb = torch.cat([angles, angles], dim=-1)       # [S, D]
    cos = emb.cos()[None, None].to(x.dtype)
    sin = emb.sin()[None, None].to(x.dtype)

    def rotate_half(t):
        t1, t2 = t.chunk(2, dim=-1)
        return torch.cat([-t2, t1], dim=-1)

    return x * cos + rotate_half(x) * sin


def forward(input_ids, past_key_values=None):
    x = F.embedding(
        input_ids,
        sd["model.embed_tokens.weight"],
    )

    batch, seq_len, _ = x.shape

    # Determine total context length for mask and RoPE
    if past_key_values is not None:
        past_len = past_key_values[0][0].shape[-2]
    else:
        past_len = 0

    total_len = past_len + seq_len

    # Mask shape: [1, 1, q_len, kv_len]
    full_mask = torch.tril(
        torch.ones(total_len, total_len, device=x.device, dtype=torch.bool)
    )
    mask = full_mask[past_len:, :][None, None, :, :]

    new_past_key_values = []

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

        q = rope_with_offset(q, past_len)

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

        k = rope_with_offset(k, past_len)

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

        # KV Cache update (across time steps, not layers!)
        if past_key_values is not None:
            past_k, past_v = past_key_values[i]
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        new_past_key_values.append((k, v))

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

    return logits, new_past_key_values


prompt = "explain transformer architecture"
messages = [{"role": "user", "content": prompt}]
input_ids = tok.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=False,
    enable_thinking=False,   # Qwen3-specific: turn off thinking block
).to(device)

generated_ids = input_ids.clone()
past_key_values = None

max_new_tokens = 100

with torch.no_grad():
    for step in range(max_new_tokens):
        # On the first step, process the full prompt.
        # On subsequent steps, ONLY process the newly generated token.
        if step == 0:
            current_input = generated_ids
        else:
            current_input = next_token

        logits, past_key_values = forward(current_input, past_key_values)

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
