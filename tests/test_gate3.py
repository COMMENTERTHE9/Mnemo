import torch

from mnemo.gate3.model import (
    TernaryLinear, TernaryGRU, Gate3Model, ternarize, footprint, audit_fp_kept,
)


def test_ternary_emits_three_levels():
    torch.manual_seed(0)
    lin = TernaryLinear(16, 8, quantize=True)
    w = lin.effective_weight().detach()
    gamma = lin.weight.detach().abs().mean()
    uniq = torch.unique(w)
    assert uniq.numel() <= 3
    allowed = torch.tensor([-float(gamma), 0.0, float(gamma)])
    for v in uniq:
        assert torch.min((allowed - v).abs()) < 1e-6  # each value in {-g,0,+g}


def test_ste_gradient_reaches_shadow():
    torch.manual_seed(0)
    lin = TernaryLinear(16, 8, quantize=True)
    x = torch.randn(4, 16)
    lin(x).sum().backward()
    assert lin.weight.grad is not None
    assert float(lin.weight.grad.abs().sum()) > 0  # STE passes gradient through


def test_gru_equivalence_with_nn():
    torch.manual_seed(0)
    n_in, H, B, N = 7, 5, 3, 6
    ref = torch.nn.GRU(n_in, H, batch_first=True)
    cell = TernaryGRU(n_in, H, quantize=False)  # FP mode -> must match nn.GRU
    with torch.no_grad():
        cell.ih.weight.copy_(ref.weight_ih_l0)
        cell.ih.bias.copy_(ref.bias_ih_l0)
        cell.hh.weight.copy_(ref.weight_hh_l0)
        cell.hh.bias.copy_(ref.bias_hh_l0)
    x = torch.randn(B, N, n_in)
    h0 = torch.randn(B, H)
    ref_out, ref_h = ref(x, h0.unsqueeze(0))
    out, h = cell(x, h0)
    assert torch.allclose(out, ref_out, atol=1e-5)
    assert torch.allclose(h, ref_h.squeeze(0), atol=1e-5)


def test_footprint_accounting():
    m = Gate3Model(30, in_dim=21, d_model=16, quantize=True)
    fq = footprint(m, quantized=True)
    ffp = footprint(m, quantized=False)
    # ternary params counted from TernaryLinear weights
    tern = sum(mod.weight.numel() for mod in m.modules()
               if isinstance(mod, TernaryLinear))
    assert fq["ternary_params"] == tern and tern > 0
    assert fq["bits"] == fq["ternary_params"] * 2 + fq["fp_params"] * 16
    assert ffp["bits"] == ffp["total_params"] * 16  # all FP when not quantized
    assert fq["bits"] < ffp["bits"]                 # ternary saves bits
    assert fq["total_params"] == ffp["total_params"]


def test_fp_kept_audit():
    m = Gate3Model(30, in_dim=21, d_model=16, quantize=True)
    audit_fp_kept(m)  # must not raise
    # proj/head are FP nn.Linear; encoder/gru matrices are TernaryLinear
    assert isinstance(m.proj, torch.nn.Linear)
    assert isinstance(m.head, torch.nn.Linear)
    assert isinstance(m.gru.ih, TernaryLinear)
    n_tern = sum(1 for mod in m.modules() if isinstance(mod, TernaryLinear))
    assert n_tern >= 6  # 4 attn + 2 ffn + 2 gru (>= 6)


def test_ternarize_zero_weight_safe():
    z = torch.zeros(4, 4)
    assert torch.equal(ternarize(z), z)  # gamma 0 -> no div-by-zero
