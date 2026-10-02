import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Qwen3:
    def __init__(self, state_dict, config):
        self.w = state_dict
        self.hidden = config.hidden_size
        self.layers = config.num_hidden_layers
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.eps = config.rms_norm_eps

        theta = config.rope_parameters["rope_theta"]
        self.inv_freq = 1.0 / theta ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
        )

    def norm(self, x, weight):
        return F.rms_norm(x, (x.shape[-1],), weight=weight, eps=self.eps)

    def rope(self, x, offset):
        pos = torch.arange(
            offset, offset + x.shape[-2], device=x.device, dtype=torch.float32
        )
        angles = pos[:, None] * self.inv_freq.to(x.device)[None, :]
        emb = torch.cat([angles, angles], dim=-1)
        cos, sin = emb.cos()[None, None], emb.sin()[None, None]
        x1, x2 = x.chunk(2, dim=-1)
        return x * cos.to(x.dtype) + torch.cat([-x2, x1], dim=-1) * sin.to(x.dtype)

    def forward(self, input_ids, past=None):
        w = self.w
        x = F.embedding(input_ids, w["model.embed_tokens.weight"])
        batch, seq_len, _ = x.shape

        past_len = 0 if past is None else past[0][0].shape[-2]
        total = past_len + seq_len
        mask = torch.tril(
            torch.ones(total, total, device=x.device, dtype=torch.bool)
        )[past_len:][None, None]

        new_past = []
        for i in range(self.layers):
            layer = f"model.layers.{i}"

            residual = x
            x = self.norm(x, w[f"{layer}.input_layernorm.weight"])

            q = self.rope(
                self.norm(
                    F.linear(x, w[f"{layer}.self_attn.q_proj.weight"])
                    .view(batch, seq_len, self.heads, self.head_dim)
                    .transpose(1, 2),
                    w[f"{layer}.self_attn.q_norm.weight"],
                ),
                past_len,
            )
            k = self.rope(
                self.norm(
                    F.linear(x, w[f"{layer}.self_attn.k_proj.weight"])
                    .view(batch, seq_len, self.kv_heads, self.head_dim)
                    .transpose(1, 2),
                    w[f"{layer}.self_attn.k_norm.weight"],
                ),
                past_len,
            )
            v = (
                F.linear(x, w[f"{layer}.self_attn.v_proj.weight"])
                .view(batch, seq_len, self.kv_heads, self.head_dim)
                .transpose(1, 2)
            )

            if past is not None:
                k = torch.cat([past[i][0], k], dim=-2)
                v = torch.cat([past[i][1], v], dim=-2)
            new_past.append((k, v))

            attn = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, enable_gqa=True
            )
            attn = attn.transpose(1, 2).reshape(batch, seq_len, -1)
            x = residual + F.linear(attn, w[f"{layer}.self_attn.o_proj.weight"])

            residual = x
            x = self.norm(x, w[f"{layer}.post_attention_layernorm.weight"])
            gate = F.linear(x, w[f"{layer}.mlp.gate_proj.weight"])
            up = F.linear(x, w[f"{layer}.mlp.up_proj.weight"])
            x = residual + F.linear(
                F.silu(gate) * up, w[f"{layer}.mlp.down_proj.weight"]
            )

        x = self.norm(x, w["model.norm.weight"])
        return F.linear(x, w["model.embed_tokens.weight"]), new_past


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEVICE)
    model = Qwen3(hf.state_dict(), hf.config)

    prompt = "create a super powerful interactive editable portfolio website in a single html file where I can add 'click/redirect' to any links I want"
    input_ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False,
        enable_thinking=False,
    ).to(DEVICE)

    generated = input_ids
    past, next_token = None, None
    with torch.no_grad():
        for step in range(1500):
            logits, past = model.forward(input_ids if step == 0 else next_token, past)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tok.eos_token_id:
                break
    print("\nPrompt :", prompt)
    # print("\n")
    print("Output :",tok.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
