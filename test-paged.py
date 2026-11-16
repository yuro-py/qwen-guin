import importlib
import math
import os
import random
import types

import pytest
import torch
import torch.nn.functional as F

pa = importlib.import_module(os.environ.get("PA_MODULE", "kv-quantization-triton-paged"))

HAS_CUDA = torch.cuda.is_available()
GPU_TRITON = HAS_CUDA and pa.HAS_TRITON
needs_triton = pytest.mark.skipif(not GPU_TRITON, reason="requires a CUDA GPU and Triton")
MB = 1024 ** 2

KIND_PARAMS = [
    "contiguous",
    "paged",
    pytest.param("triton", marks=needs_triton),
    pytest.param("int8", marks=needs_triton),
]

FAKE_MODEL = types.SimpleNamespace(layers=28, kv_heads=8, head_dim=128, max_positions=40960)
ALL_MODES = [
    ("Non-paged", "contiguous"),
    ("Torch-paged", "paged"),
    ("Triton-paged", "triton"),
    ("Triton-INT8", "int8"),
]

PASSAGE_A = (
    "The lighthouse keeper rose before dawn, as he had every morning for thirty-one years, and climbed the one "
    "hundred and twelve iron steps to the lamp room. The sea below was the color of slate, and a thin line of "
    "pale light had begun to separate the water from the sky. He wiped the salt from the great lens with a soft "
    "cloth, checked the clockwork that turned it, and wrote the night's observations in a ledger whose pages had "
    "gone soft at the edges. Ships passed less often now than they once had; most carried instruments that told "
    "them where the rocks lay long before any beam could. Still, the harbor authority insisted the light be kept, "
    "and the keeper did not argue. He brewed tea on the small stove, listened to the gulls complaining on the "
    "rail, and watched a fishing boat slip out of the cove with its lantern swinging. Somewhere inland a church "
    "bell rang six times. The keeper thought about the letter in his coat pocket, the one offering him a quiet "
    "cottage near his sister's village, and he turned it over once more without opening it. Then he went back to "
    "polishing brass, because brass left alone turns green, and he had never learned to leave things alone."
)

PASSAGE_B = (
    "Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy into "
    "chemical energy stored in sugar. Inside the chloroplasts of a leaf cell, pigments called chlorophylls absorb "
    "mostly red and blue light and reflect green, which is why leaves look the way they do. The light-dependent "
    "reactions occur in the thylakoid membranes, where absorbed energy splits water molecules, releasing oxygen as "
    "a byproduct and producing the energy carriers ATP and NADPH. These carriers then power the Calvin cycle in the "
    "stroma, a series of enzyme-driven steps that fix carbon dioxide from the air into three-carbon molecules and "
    "ultimately assemble them into glucose. The rate of the whole process depends on light intensity, carbon "
    "dioxide concentration, temperature, and the availability of water. When leaf pores called stomata close to "
    "conserve water on a hot day, carbon dioxide intake falls and the cycle slows. Over millions of years, this "
    "chemistry has shaped the composition of the atmosphere and underpins nearly every food chain on Earth, which "
    "is why even a small change in global photosynthesis can ripple through entire ecosystems."
)


def outlier(g, shape):
    x = torch.randn(shape, generator=g)
    x[..., 0] = 1e4
    return x


DISTS = {
    "normal": lambda g, s: torch.randn(s, generator=g),
    "large": lambda g, s: torch.randn(s, generator=g) * 1e3,
    "tiny": lambda g, s: torch.randn(s, generator=g) * 1e-6,
    "zeros": lambda g, s: torch.zeros(s),
    "outlier": outlier,
    "heavy_tail": lambda g, s: torch.randn(s, generator=g) ** 3,
}


def reference_attention(q, K, V):
    group = q.shape[1] // K.shape[1]
    Kd = K.double().repeat_interleave(group, dim=1)
    Vd = V.double().repeat_interleave(group, dim=1)
    scores = (q.double() @ Kd.transpose(-1, -2)) * q.shape[-1] ** -0.5
    return (torch.softmax(scores, dim=-1) @ Vd).float()


def poison(cache):
    if cache.quant:
        for t in cache.ks + cache.vs:
            t.fill_(float("nan"))
        for t in cache.kp + cache.vp:
            t.fill_(127)
    else:
        for t in cache.kp + cache.vp:
            t.fill_(float("nan"))


def build_case(quant, heads, kv_heads, head_dim, total, block_size, seed,
               shuffle=True, magnitude=1.0, mixed=False):
    g = torch.Generator().manual_seed(seed)
    K = torch.randn(1, kv_heads, total, head_dim, generator=g) * magnitude
    V = torch.randn(1, kv_heads, total, head_dim, generator=g)
    q = torch.randn(1, heads, 1, head_dim, generator=g)
    if mixed:
        K[:, :, ::7] *= 100
        V[:, :, ::7] *= 100
    ref = reference_attention(q, K, V).cuda()
    K, V, q = K.cuda(), V.cuda(), q.cuda()
    num_blocks = math.ceil(total / block_size) + 2
    cache = pa.TritonPagedCache(
        1, kv_heads, head_dim, num_blocks, block_size, torch.float32, "cuda", quant=quant
    )
    poison(cache)
    if shuffle:
        random.Random(seed).shuffle(cache.allocator.free)
    n = total - 1
    if n > 0:
        cache.begin(n)
        cache.update(0, K[:, :, :n], V[:, :, :n])
        cache.advance(n)
    cache.begin(1)
    cache.update(0, K[:, :, n:], V[:, :, n:])
    out = cache.attend(0, q)
    return types.SimpleNamespace(out=out, ref=ref, cache=cache, K=K, V=V, q=q)


def read_cache(cache, layer, which, n):
    pool = cache.kp[layer] if which == "k" else cache.vp[layer]
    scales = cache.ks[layer] if which == "k" else cache.vs[layer]
    blks = torch.tensor([cache.table[t // cache.block_size] for t in range(n)], device=pool.device)
    offs = torch.arange(n, device=pool.device) % cache.block_size
    ints = pool[blks, :, offs].permute(1, 0, 2)
    sc = scales[blks, :, offs].permute(1, 0)
    return ints, sc


@needs_triton
@pytest.mark.parametrize("total", [1, 2, 15, 16, 17, 20, 31, 32, 33, 64, 100, 257, 1000])
@pytest.mark.parametrize("heads,kv_heads,head_dim", [(16, 8, 128), (8, 8, 64), (16, 2, 128), (4, 1, 64)])
def test_kernel_fp32_matches_float64_reference(total, heads, kv_heads, head_dim):
    case = build_case(False, heads, kv_heads, head_dim, total, 16, seed=total)
    assert torch.isfinite(case.out).all()
    assert torch.allclose(case.out, case.ref, atol=1e-4, rtol=1e-4)


@needs_triton
@pytest.mark.parametrize("block_size", [16, 32, 64])
@pytest.mark.parametrize("total", [33, 100, 500])
def test_kernel_result_independent_of_block_size(block_size, total):
    case = build_case(False, 16, 8, 128, total, block_size, seed=1)
    assert torch.allclose(case.out, case.ref, atol=1e-4, rtol=1e-4)


@needs_triton
@pytest.mark.parametrize("magnitude", [0.01, 1.0, 10.0, 30.0])
def test_kernel_numerically_stable_for_extreme_scores(magnitude):
    case = build_case(False, 16, 8, 128, 300, 16, seed=3, magnitude=magnitude)
    assert torch.isfinite(case.out).all()
    assert torch.allclose(case.out, case.ref, atol=5e-3, rtol=1e-3)


@needs_triton
def test_kernel_physical_block_layout_is_irrelevant():
    shuffled = build_case(False, 16, 8, 128, 500, 16, seed=5, shuffle=True)
    ordered = build_case(False, 16, 8, 128, 500, 16, seed=5, shuffle=False)
    assert shuffled.cache.table != sorted(shuffled.cache.table)
    assert ordered.cache.table == sorted(ordered.cache.table)
    assert torch.equal(shuffled.out, ordered.out)


@needs_triton
@pytest.mark.parametrize("quant", [False, True])
def test_kernel_attend_is_read_only_and_deterministic(quant):
    case = build_case(quant, 16, 8, 128, 100, 16, seed=9)
    pools = case.cache.kp + case.cache.vp
    if quant:
        pools = pools + case.cache.ks + case.cache.vs
    snapshot = [t.clone() for t in pools]
    again = case.cache.attend(0, case.q)
    assert torch.equal(case.out, again)
    for t, s in zip(pools, snapshot):
        torch.testing.assert_close(t, s, rtol=0, atol=0, equal_nan=True)


@needs_triton
@pytest.mark.parametrize("quant", [False, True])
def test_kernel_layers_do_not_alias(quant):
    layers, kv_heads, head_dim, heads, total = 3, 8, 128, 16, 70
    g = torch.Generator().manual_seed(21)
    Ks = [torch.randn(1, kv_heads, total, head_dim, generator=g) for _ in range(layers)]
    Vs = [torch.randn(1, kv_heads, total, head_dim, generator=g) for _ in range(layers)]
    qs = [torch.randn(1, heads, 1, head_dim, generator=g) for _ in range(layers)]
    refs = [reference_attention(qs[i], Ks[i], Vs[i]).cuda() for i in range(layers)]
    cache = pa.TritonPagedCache(layers, kv_heads, head_dim, 8, 16, torch.float32, "cuda", quant=quant)
    n = total - 1
    cache.begin(n)
    for i in range(layers):
        cache.update(i, Ks[i][:, :, :n].cuda(), Vs[i][:, :, :n].cuda())
    cache.advance(n)
    cache.begin(1)
    for i in range(layers):
        cache.update(i, Ks[i][:, :, n:].cuda(), Vs[i][:, :, n:].cuda())
        out = cache.attend(i, qs[i].cuda())
        if quant:
            rel = (out - refs[i]).norm() / refs[i].norm()
            assert rel < 0.03
        else:
            assert torch.allclose(out, refs[i], atol=1e-4, rtol=1e-4)


@needs_triton
@pytest.mark.parametrize("dist", list(DISTS))
def test_kernel_int8_roundtrip_error_bounded_by_half_quantization_step(dist):
    kv_heads, head_dim, n = 8, 128, 50
    g = torch.Generator().manual_seed(11)
    K = DISTS[dist](g, (1, kv_heads, n, head_dim)).cuda()
    V = DISTS[dist](g, (1, kv_heads, n, head_dim)).cuda()
    cache = pa.TritonPagedCache(1, kv_heads, head_dim, 8, 16, torch.float32, "cuda", quant=True)
    random.Random(0).shuffle(cache.allocator.free)
    cache.begin(n)
    cache.update(0, K, V)
    cache.advance(n)
    for which, X in (("k", K), ("v", V)):
        ints, scales = read_cache(cache, 0, which, n)
        assert ints.dtype == torch.int8
        assert ints.min().item() >= -127 and ints.max().item() <= 127
        deq = ints.float() * scales[..., None]
        ref = X[0]
        assert not torch.isnan(deq).any()
        amax = ref.abs().amax(dim=-1)
        expected_scale = amax.clamp_min(1e-8) / 127.0
        assert torch.allclose(scales, expected_scale, rtol=1e-4, atol=0)
        bound = (amax / 254.0 * (1 + 1e-3) + 1e-12)[..., None]
        assert ((deq - ref).abs() <= bound).all()


@needs_triton
@pytest.mark.parametrize("total", [1, 17, 100, 500])
@pytest.mark.parametrize("variant", ["uniform", "mixed_token_magnitudes"])
def test_kernel_int8_attention_close_to_float64_reference(total, variant):
    case = build_case(True, 16, 8, 128, total, 16, seed=total + 7, mixed=(variant != "uniform"))
    assert torch.isfinite(case.out).all()
    rel = ((case.out - case.ref).norm() / case.ref.norm()).item()
    cos = F.cosine_similarity(case.out[0, :, 0], case.ref[0, :, 0], dim=-1).min().item()
    assert rel < 0.03
    assert cos > 0.995


def test_cache_allocator_hands_out_unique_blocks_then_raises():
    allocator = pa.BlockAllocator(5)
    got = [allocator.alloc() for _ in range(5)]
    assert sorted(got) == list(range(5))
    with pytest.raises(RuntimeError, match="exhausted"):
        allocator.alloc()


def test_cache_pool_exhaustion_raises():
    cache = pa.PagedCache(1, 1, 8, 2, 16, torch.float32, "cpu")
    with pytest.raises(RuntimeError, match="exhausted"):
        cache.begin(40)


@pytest.mark.parametrize("n_tokens,expected", [(1, 1), (15, 1), (16, 1), (17, 2), (48, 3), (49, 4)])
def test_cache_block_count_after_prefill(n_tokens, expected):
    cache = pa.PagedCache(1, 2, 8, 8, 16, torch.float32, "cpu")
    cache.begin(n_tokens)
    cache.advance(n_tokens)
    assert cache.blocks_used() == expected


def test_cache_blocks_grow_one_at_a_time_during_decode():
    cache = pa.PagedCache(1, 2, 8, 8, 16, torch.float32, "cpu")
    cache.begin(10)
    cache.advance(10)
    for _ in range(100):
        cache.begin(1)
        cache.advance(1)
        assert cache.blocks_used() == math.ceil(cache.length / 16)


def test_cache_paged_gather_equals_contiguous_concatenation():
    torch.manual_seed(0)
    layers, kv_heads, head_dim, block_size = 3, 4, 8, 16
    contiguous = pa.ContiguousCache(layers)
    paged = pa.PagedCache(layers, kv_heads, head_dim, 16, block_size, torch.float32, "cpu")
    random.Random(0).shuffle(paged.allocator.free)
    chunks = [37] + [1] * 40 + [5, 1, 9]
    for n in chunks:
        contiguous.begin(n)
        paged.begin(n)
        for layer in range(layers):
            k = torch.randn(1, kv_heads, n, head_dim)
            v = torch.randn(1, kv_heads, n, head_dim)
            kc, vc = contiguous.update(layer, k, v)
            kp, vp = paged.update(layer, k, v)
            assert torch.equal(kc, kp)
            assert torch.equal(vc, vp)
        contiguous.advance(n)
        paged.advance(n)
    assert contiguous.length == paged.length == sum(chunks)
    assert paged.table != sorted(paged.table)


@pytest.mark.parametrize("head_dim", [64, 128])
def test_cache_int8_footprint_matches_theory(head_dim):
    args = (4, 8, head_dim, 10, 16)
    fp = pa.TritonPagedCache(*args, torch.float32, "cpu", quant=False)
    q8 = pa.TritonPagedCache(*args, torch.float32, "cpu", quant=True)
    assert fp.kv_bytes() == 4 * 2 * 10 * 8 * 16 * head_dim * 4
    assert q8.kv_bytes() / fp.kv_bytes() == pytest.approx((head_dim + 4) / (4 * head_dim))


def test_cache_contiguous_kv_bytes_tracks_length():
    cache = pa.ContiguousCache(2)
    for layer in range(2):
        cache.update(layer, torch.zeros(1, 4, 10, 8), torch.zeros(1, 4, 10, 8))
    assert cache.kv_bytes() == 2 * 2 * 4 * 10 * 8 * 4


def draws(row, n, temperature, top_k, top_p, seed=0):
    torch.manual_seed(seed)
    batch = row.unsqueeze(0).expand(n, -1).contiguous()
    return pa.sample(batch, temperature, top_k, top_p).squeeze(1)


def frequencies(samples, vocab):
    return torch.bincount(samples, minlength=vocab).float() / samples.numel()


def entropy(freq):
    nz = freq[freq > 0]
    return -(nz * nz.log()).sum().item()


@pytest.mark.parametrize("temperature", [0.0, None, -1.0])
def test_sampling_zero_or_missing_temperature_is_greedy(temperature):
    torch.manual_seed(1)
    logits = torch.randn(1, 100)
    out = pa.sample(logits, temperature, 50, 0.95)
    assert out.shape == (1, 1)
    assert out.item() == logits.argmax().item()


def test_sampling_top_k_one_is_greedy_at_any_temperature():
    torch.manual_seed(2)
    row = torch.randn(64)
    samples = draws(row, 500, 1.7, 1, 1.0)
    assert (samples == row.argmax()).all()


def test_sampling_top_k_restricts_support():
    torch.manual_seed(3)
    row = torch.randn(50) * 0.3
    allowed = set(torch.topk(row, 5).indices.tolist())
    samples = draws(row, 4000, 1.0, 5, 1.0)
    seen = set(samples.tolist())
    assert seen <= allowed
    assert len(seen) >= 3


def test_sampling_top_p_restricts_to_nucleus_and_renormalizes():
    row = torch.log(torch.tensor([0.5, 0.3, 0.1, 0.05, 0.05]))
    samples = draws(row, 4000, 1.0, 0, 0.7)
    freq = frequencies(samples, 5)
    assert freq[2:].sum().item() == 0.0
    assert 0.58 < freq[0].item() < 0.67


def test_sampling_tiny_top_p_always_keeps_the_top_token():
    row = torch.log(torch.tensor([0.5, 0.3, 0.1, 0.05, 0.05]))
    assert (draws(row, 500, 1.0, 0, 1e-6) == 0).all()


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_sampling_matches_softmax_distribution_when_filters_disabled(temperature):
    torch.manual_seed(4)
    row = torch.randn(8)
    samples = draws(row, 40000, temperature, None, None)
    expected = torch.softmax(row / temperature, dim=-1)
    assert (frequencies(samples, 8) - expected).abs().max().item() < 0.012


def test_sampling_higher_temperature_raises_entropy():
    torch.manual_seed(5)
    row = torch.randn(16)
    cold = entropy(frequencies(draws(row, 20000, 0.4, None, None), 16))
    hot = entropy(frequencies(draws(row, 20000, 1.6, None, None), 16))
    assert hot > cold


def test_sampling_is_reproducible_under_a_seed():
    torch.manual_seed(6)
    row = torch.randn(200)
    a = draws(row, 1000, 1.0, 50, 0.95, seed=7)
    b = draws(row, 1000, 1.0, 50, 0.95, seed=7)
    c = draws(row, 1000, 1.0, 50, 0.95, seed=8)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_sampling_top_k_larger_than_vocab_is_safe():
    row = torch.randn(10)
    samples = draws(row, 200, 1.0, 1000, 0.95)
    assert samples.min() >= 0 and samples.max() < 10


def test_limits_kv_need_ordering_and_ratio():
    total = 1520
    int8 = pa.kv_need(FAKE_MODEL, "int8", total)
    paged = pa.kv_need(FAKE_MODEL, "paged", total)
    contiguous = pa.kv_need(FAKE_MODEL, "contiguous", total)
    assert int8 < paged < contiguous
    assert int8 / paged == pytest.approx((128 + 4) / (4 * 128))


def test_limits_kv_need_rounds_up_to_whole_blocks():
    per_token = 2 * 28 * 8 * 128 * 4
    assert pa.kv_need(FAKE_MODEL, "paged", 17) == 32 * per_token
    assert pa.kv_need(FAKE_MODEL, "paged", 16) == 16 * per_token


def test_limits_all_modes_run_when_memory_is_plentiful(monkeypatch):
    monkeypatch.setattr(pa, "available_memory", lambda: 64 * 1024 * MB)
    runnable, needs, free = pa.fit_modes(FAKE_MODEL, 20, 1500, ALL_MODES)
    assert runnable == ALL_MODES
    assert set(needs) == {name for name, _ in ALL_MODES}


def test_limits_unknown_free_memory_does_not_block(monkeypatch):
    monkeypatch.setattr(pa, "available_memory", lambda: None)
    runnable, _, free = pa.fit_modes(FAKE_MODEL, 20, 1500, ALL_MODES)
    assert runnable == ALL_MODES and free is None


def test_limits_only_int8_fits(monkeypatch, capsys):
    total = 20 + 1500
    margin = pa.SAFETY_MARGIN_MB * MB
    mid = (pa.kv_need(FAKE_MODEL, "int8", total) + pa.kv_need(FAKE_MODEL, "paged", total)) // 2 + margin
    monkeypatch.setattr(pa, "available_memory", lambda: mid)
    runnable, _, _ = pa.fit_modes(FAKE_MODEL, 20, 1500, ALL_MODES)
    assert runnable == [("Triton-INT8", "int8")]
    out = capsys.readouterr().out
    assert "Skipping Non-paged" in out and "Skipping Torch-paged" in out and "Skipping Triton-paged" in out


def test_limits_exact_boundary_is_accepted(monkeypatch):
    total = 20 + 1500
    exact = pa.kv_need(FAKE_MODEL, "int8", total) + pa.SAFETY_MARGIN_MB * MB
    monkeypatch.setattr(pa, "available_memory", lambda: exact)
    runnable, _, _ = pa.fit_modes(FAKE_MODEL, 20, 1500, [("Triton-INT8", "int8")])
    assert runnable == [("Triton-INT8", "int8")]
    monkeypatch.setattr(pa, "available_memory", lambda: exact - 1)
    with pytest.raises(RuntimeError):
        pa.fit_modes(FAKE_MODEL, 20, 1500, [("Triton-INT8", "int8")])


def test_limits_error_when_nothing_fits(monkeypatch):
    monkeypatch.setattr(pa, "available_memory", lambda: 1 * MB)
    with pytest.raises(RuntimeError, match="needs at least"):
        pa.fit_modes(FAKE_MODEL, 20, 1500, ALL_MODES)


def test_limits_error_beyond_model_context(monkeypatch):
    monkeypatch.setattr(pa, "available_memory", lambda: 64 * 1024 * MB)
    with pytest.raises(RuntimeError, match="context limit"):
        pa.fit_modes(FAKE_MODEL, 20, FAKE_MODEL.max_positions, ALL_MODES)


def test_helpers_compare_outputs_reports_divergence():
    a = torch.tensor([[1, 2, 3, 4]])
    assert pa.compare_outputs(a, a.clone()) == "identical"
    assert pa.compare_outputs(a, torch.tensor([[1, 2, 9, 4]])) == "diverges at token 2 of 4"
    assert pa.compare_outputs(a, torch.tensor([[1, 2, 3]])).startswith("equal prefix")


def test_helpers_delta_cell_semantics():
    assert pa.delta_cell(100.0, 90.0, True) == "-10.0% (better)"
    assert pa.delta_cell(100.0, 90.0, False) == "-10.0% (worse)"
    assert pa.delta_cell(100.0, 110.0, True) == "+10.0% (worse)"
    assert pa.delta_cell(100.0, 100.0, True) == "0.0% (equal)"
    assert pa.delta_cell(0.0, 5.0, True) == "n/a"
    assert pa.delta_cell(None, 5.0, True) == "n/a"
    assert pa.delta_cell(float("nan"), 5.0, True) == "n/a"
    assert pa.cell(float("nan"), ".2f") == "n/a"
    assert pa.cell(None, ".2f") == "n/a"
    assert pa.cell(1.2345, ".2f") == "1.23"


def fake_metrics(scale):
    return {
        "tokens": 100, "ttft_ms": 10.0 * scale, "latency_s": 2.0 * scale,
        "throughput_tps": 50.0 / scale, "decode_tps": 52.0 / scale, "tpot_ms": 19.0 * scale,
        "peak_mb": 100.0 * scale, "kv_mb": 50.0 * scale, "blocks": 7,
    }


def test_helpers_benchmark_table_layout(capsys):
    names = ["Non-paged", "Torch-paged", "Triton-INT8"]
    pa.print_benchmarks(names, [fake_metrics(1.0), fake_metrics(1.1), fake_metrics(0.9)])
    out = capsys.readouterr().out
    for header in names:
        assert header in out
    for label in ["TTFT (ms)", "Total latency (s)", "Throughput (tok/s)", "Decode throughput (tok/s)",
                  "Time per output token (ms)", "KV cache memory (MB)", "Peak extra memory (MB)"]:
        assert label in out
    assert "Relative to Non-paged" in out
    assert "+10.0% (worse)" in out
    assert "-10.0% (better)" in out


def test_helpers_benchmark_table_single_mode_has_no_relative_section(capsys):
    pa.print_benchmarks(["Triton-INT8"], [fake_metrics(1.0)])
    assert "Relative to" not in capsys.readouterr().out


@pytest.fixture(scope="session")
def bundle():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(pa.MODEL)
        hf = AutoModelForCausalLM.from_pretrained(pa.MODEL, dtype=pa.DTYPE).to(pa.DEVICE)
    except Exception as exc:
        pytest.skip(f"model unavailable: {exc}")
    hf.eval()
    model = pa.Qwen3(hf.state_dict(), hf.config)
    return types.SimpleNamespace(tok=tok, hf=hf, model=model)


def ids_of(bundle, text):
    return bundle.tok(text, return_tensors="pt").input_ids.to(pa.DEVICE)


def chat_ids(bundle, prompt):
    return bundle.tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt", return_dict=False, enable_thinking=False,
    ).to(pa.DEVICE)


@pytest.fixture(scope="session")
def passages(bundle):
    a = ids_of(bundle, PASSAGE_A)
    b = ids_of(bundle, PASSAGE_B)
    assert a.shape[1] >= 192 and b.shape[1] >= 128
    return types.SimpleNamespace(a=a, b=b, long=torch.cat([a, b], dim=1)[:, :320])


def lockstep(model, kind_a, kind_b, ids, prefill):
    total = ids.shape[1]
    cache_a = pa.make_cache(kind_a, model, total)
    cache_b = pa.make_cache(kind_b, model, total)
    steps = total - prefill
    max_diff, cos_sum, cos_min, agree = 0.0, 0.0, 1.0, 0
    nll_a = nll_b = 0.0
    with torch.no_grad():
        la = model.forward(ids[:, :prefill], cache_a)[0, -1]
        lb = model.forward(ids[:, :prefill], cache_b)[0, -1]
        for t in range(prefill, total):
            max_diff = max(max_diff, (la - lb).abs().max().item())
            cos = F.cosine_similarity(la, lb, dim=-1).item()
            cos_sum += cos
            cos_min = min(cos_min, cos)
            agree += int(la.argmax().item() == lb.argmax().item())
            target = ids[0, t]
            nll_a -= F.log_softmax(la, dim=-1)[target].item()
            nll_b -= F.log_softmax(lb, dim=-1)[target].item()
            nxt = ids[:, t:t + 1]
            la = model.forward(nxt, cache_a)[0, -1]
            lb = model.forward(nxt, cache_b)[0, -1]
    del cache_a, cache_b
    return types.SimpleNamespace(
        max_abs_diff=max_diff, mean_cos=cos_sum / steps, min_cos=cos_min,
        top1_agreement=agree / steps, nll_a=nll_a / steps, nll_b=nll_b / steps,
    )


@pytest.mark.parametrize("length", [1, 7, 33, 120])
def test_model_logits_match_huggingface(bundle, passages, length):
    ids = passages.a[:, :length]
    with torch.no_grad():
        ref = bundle.hf(input_ids=ids).logits[0, -1]
        ours = bundle.model.forward(ids, pa.ContiguousCache(bundle.model.layers))[0, -1]
    assert ours.argmax().item() == ref.argmax().item()
    assert torch.allclose(ours, ref, atol=5e-3, rtol=1e-3)


def test_model_greedy_generation_matches_huggingface(bundle, monkeypatch):
    monkeypatch.setattr(pa, "TEMPERATURE", 0.0)
    ids = chat_ids(bundle, "Explain in two sentences why the sky is blue.")
    n_new = 48
    ours, _ = pa.run("contiguous", bundle.model, ids, n_new, bundle.tok.eos_token_id, 0)
    with torch.no_grad():
        ref = bundle.hf.generate(
            ids, attention_mask=torch.ones_like(ids), max_new_tokens=n_new,
            do_sample=False, temperature=None, top_p=None, top_k=None,
        )[:, ids.shape[1]:]
    n = min(ours.shape[1], ref.shape[1])
    assert abs(ours.shape[1] - ref.shape[1]) <= 1
    assert torch.equal(ours[:, :n], ref[:, :n])


def test_model_forward_is_deterministic_and_nll_is_sane(bundle, passages):
    r = lockstep(bundle.model, "contiguous", "contiguous", passages.a[:, :192], 8)
    assert r.max_abs_diff < 1e-5
    assert 0.3 < r.nll_a < 6.0


@pytest.mark.parametrize("kind,atol", [
    ("paged", 1e-4),
    pytest.param("triton", 2e-3, marks=needs_triton),
])
def test_model_fp32_backends_match_contiguous_teacher_forced(bundle, passages, kind, atol):
    r = lockstep(bundle.model, "contiguous", kind, passages.a[:, :192], 8)
    assert r.max_abs_diff < atol
    assert r.top1_agreement >= 0.99
    assert abs(r.nll_a - r.nll_b) < 1e-3


@needs_triton
def test_model_int8_logits_stay_faithful(bundle, passages):
    r = lockstep(bundle.model, "contiguous", "int8", passages.a[:, :192], 8)
    assert r.mean_cos > 0.999
    assert r.min_cos > 0.98
    assert r.top1_agreement >= 0.90
    assert abs(r.nll_b - r.nll_a) < 0.05


@pytest.mark.parametrize("prefill", [1, 15, 16, 17, 100])
@pytest.mark.parametrize("kind", [
    pytest.param("triton", marks=needs_triton),
    pytest.param("int8", marks=needs_triton),
])
def test_model_block_boundary_prefill_lengths(bundle, passages, kind, prefill):
    r = lockstep(bundle.model, "contiguous", kind, passages.a[:, :120], prefill)
    if kind == "triton":
        assert r.max_abs_diff < 2e-3
    else:
        assert r.mean_cos > 0.995
        assert r.top1_agreement >= 0.85


@pytest.mark.parametrize("kind", [
    pytest.param("triton", marks=needs_triton),
    pytest.param("int8", marks=needs_triton),
])
def test_model_long_context_against_torch_paged(bundle, passages, kind):
    r = lockstep(bundle.model, "paged", kind, passages.long, 5)
    if kind == "triton":
        assert r.max_abs_diff < 2e-3
        assert r.top1_agreement >= 0.99
    else:
        assert r.mean_cos > 0.995
        assert r.top1_agreement >= 0.88
        assert abs(r.nll_b - r.nll_a) < 0.05


def test_model_torch_paged_greedy_tokens_identical_to_contiguous(bundle, monkeypatch):
    monkeypatch.setattr(pa, "TEMPERATURE", 0.0)
    ids = chat_ids(bundle, "List three uses of a paperclip.")
    eos = bundle.tok.eos_token_id
    a, _ = pa.run("contiguous", bundle.model, ids, 96, eos, 0)
    b, _ = pa.run("paged", bundle.model, ids, 96, eos, 0)
    assert torch.equal(a, b)


@needs_triton
def test_model_triton_fp32_greedy_tokens_match_contiguous(bundle, monkeypatch):
    monkeypatch.setattr(pa, "TEMPERATURE", 0.0)
    ids = chat_ids(bundle, "List three uses of a paperclip.")
    eos = bundle.tok.eos_token_id
    a, _ = pa.run("contiguous", bundle.model, ids, 96, eos, 0)
    b, _ = pa.run("triton", bundle.model, ids, 96, eos, 0)
    assert pa.compare_outputs(a, b) == "identical"


@pytest.mark.parametrize("kind", KIND_PARAMS)
def test_model_seeded_sampling_is_reproducible(bundle, monkeypatch, kind):
    monkeypatch.setattr(pa, "TEMPERATURE", 0.9)
    ids = chat_ids(bundle, "Write a short poem about the sea.")
    eos = bundle.tok.eos_token_id
    a, _ = pa.run(kind, bundle.model, ids, 40, eos, 123)
    b, _ = pa.run(kind, bundle.model, ids, 40, eos, 123)
    c, _ = pa.run(kind, bundle.model, ids, 40, eos, 124)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_model_stochastic_sampling_agrees_between_fp32_backends(bundle, monkeypatch):
    monkeypatch.setattr(pa, "TEMPERATURE", 0.7)
    ids = chat_ids(bundle, "Describe a rainy afternoon in a small town.")
    eos = bundle.tok.eos_token_id
    a, _ = pa.run("contiguous", bundle.model, ids, 60, eos, 5)
    b, _ = pa.run("paged", bundle.model, ids, 60, eos, 5)
    assert torch.equal(a, b)


@pytest.mark.parametrize("top_k,top_p", [(1, 1.0), (0, 1e-6)])
def test_model_degenerate_filters_reduce_to_greedy(bundle, monkeypatch, top_k, top_p):
    ids = chat_ids(bundle, "Name three rivers in Europe.")
    eos = bundle.tok.eos_token_id
    monkeypatch.setattr(pa, "TEMPERATURE", 0.0)
    greedy, _ = pa.run("contiguous", bundle.model, ids, 40, eos, 0)
    monkeypatch.setattr(pa, "TEMPERATURE", 1.0)
    monkeypatch.setattr(pa, "TOP_K", top_k)
    monkeypatch.setattr(pa, "TOP_P", top_p)
    filtered, _ = pa.run("contiguous", bundle.model, ids, 40, eos, 99)
    assert torch.equal(greedy, filtered)


def test_model_temperature_changes_the_output(bundle, monkeypatch):
    ids = chat_ids(bundle, "Tell me something surprising about octopuses.")
    eos = bundle.tok.eos_token_id
    monkeypatch.setattr(pa, "TEMPERATURE", 0.0)
    greedy, _ = pa.run("contiguous", bundle.model, ids, 48, eos, 0)
    monkeypatch.setattr(pa, "TEMPERATURE", 1.4)
    hot, _ = pa.run("contiguous", bundle.model, ids, 48, eos, 0)
    assert not torch.equal(greedy, hot)


@pytest.mark.parametrize("kind", KIND_PARAMS)
def test_model_benchmark_metrics_are_self_consistent(bundle, monkeypatch, kind):
    monkeypatch.setattr(pa, "TEMPERATURE", 0.0)
    ids = chat_ids(bundle, "Say hello and explain what a prime number is.")
    out, m = pa.run(kind, bundle.model, ids, 24, bundle.tok.eos_token_id, 0)
    n = out.shape[1]
    assert m["tokens"] == n and 1 <= n <= 24
    assert 0 < m["ttft_ms"] / 1000 < m["latency_s"]
    assert m["throughput_tps"] == pytest.approx(n / m["latency_s"])
    if n > 1:
        assert m["decode_tps"] > 0 and m["tpot_ms"] > 0
    assert m["kv_mb"] > 0
    if kind != "contiguous":
        assert m["blocks"] == math.ceil((ids.shape[1] + n - 1) / pa.BLOCK_SIZE)
    if HAS_CUDA:
        assert m["peak_mb"] > 0


@needs_triton
def test_model_int8_cache_is_about_a_quarter_the_size(bundle, monkeypatch):
    monkeypatch.setattr(pa, "TEMPERATURE", 0.0)
    ids = chat_ids(bundle, "Write one sentence about autumn.")
    eos = bundle.tok.eos_token_id
    _, fp = pa.run("triton", bundle.model, ids, 32, eos, 0)
    _, q8 = pa.run("int8", bundle.model, ids, 32, eos, 0)
    ratio = q8["kv_mb"] / fp["kv_mb"]
    assert ratio == pytest.approx((128 + 4) / (4 * 128), rel=1e-3)
    assert q8["peak_mb"] < fp["peak_mb"]


@pytest.mark.parametrize("max_new", [1, 2, 7])
def test_model_max_new_tokens_is_respected(bundle, monkeypatch, max_new):
    monkeypatch.setattr(pa, "TEMPERATURE", 0.0)
    ids = chat_ids(bundle, "Count upward from one.")
    out, m = pa.run("paged", bundle.model, ids, max_new, bundle.tok.eos_token_id, 0)
    assert 1 <= out.shape[1] <= max_new
    assert m["tokens"] == out.shape[1]


@pytest.mark.skipif(not HAS_CUDA, reason="requires a CUDA GPU")
@pytest.mark.parametrize("kind", KIND_PARAMS)
def test_model_runs_do_not_leak_gpu_memory(bundle, monkeypatch, kind):
    monkeypatch.setattr(pa, "TEMPERATURE", 0.0)
    ids = chat_ids(bundle, "Give me a fun fact.")
    eos = bundle.tok.eos_token_id
    pa.run(kind, bundle.model, ids, 16, eos, 0)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    for _ in range(3):
        pa.run(kind, bundle.model, ids, 16, eos, 0)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() - before < 2 * MB


def run_main(bundle, monkeypatch, capsys, **overrides):
    monkeypatch.setattr(pa.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: bundle.hf)
    monkeypatch.setattr(pa, "MAX_NEW_TOKENS", 24)
    for key, value in overrides.items():
        monkeypatch.setattr(pa, key, value)
    pa.main()
    return capsys.readouterr().out


def test_e2e_output_contains_every_required_section(bundle, monkeypatch, capsys):
    out = run_main(bundle, monkeypatch, capsys, TEMPERATURE=0.0, TOP_K=50, TOP_P=0.95)
    assert f"T={pa.TEMPERATURE}, top_k={pa.TOP_K}, top_p={pa.TOP_P}" in out
    assert f"Prompt : {pa.PROMPT}" in out
    assert "Response:" in out
    assert "Max new tokens: 24" in out
    for section in ["Benchmarks", "TTFT (ms)", "Total latency (s)", "Throughput (tok/s)", "Relative to Non-paged"]:
        assert section in out
    assert "Output of Torch-paged vs Non-paged:" in out


@needs_triton
def test_e2e_int8_toggle_adds_and_removes_the_column(bundle, monkeypatch, capsys):
    on = run_main(bundle, monkeypatch, capsys, KV_QUANT_INT8=True)
    assert "Triton-INT8" in on and "KV INT8       : on" in on
    off = run_main(bundle, monkeypatch, capsys, KV_QUANT_INT8=False)
    assert "Triton-INT8" not in off and "KV INT8       : off" in off


def test_e2e_runs_under_stochastic_sampling(bundle, monkeypatch, capsys):
    out = run_main(bundle, monkeypatch, capsys, TEMPERATURE=0.8, TOP_K=40, TOP_P=0.9, SEED=3)
    assert "T=0.8, top_k=40, top_p=0.9, seed=3" in out
    assert "Benchmarks" in out


def test_e2e_errors_when_memory_limit_is_crossed(bundle, monkeypatch, capsys):
    monkeypatch.setattr(pa, "available_memory", lambda: 1 * MB)
    with pytest.raises(RuntimeError, match="needs at least"):
        run_main(bundle, monkeypatch, capsys)


@needs_triton
def test_e2e_int8_alone_runs_when_only_it_fits(bundle, monkeypatch, capsys):
    prompt_len = chat_ids(bundle, pa.PROMPT).shape[1]
    total = prompt_len + 24
    margin = pa.SAFETY_MARGIN_MB * MB
    mid = (pa.kv_need(bundle.model, "int8", total) + pa.kv_need(bundle.model, "paged", total)) // 2 + margin
    monkeypatch.setattr(pa, "available_memory", lambda: mid)
    out = run_main(bundle, monkeypatch, capsys, KV_QUANT_INT8=True)
    assert "Skipping Non-paged" in out
    assert "Triton-INT8" in out
    assert "Relative to" not in out
    assert "Response:" in out
