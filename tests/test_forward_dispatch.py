import pytest
import torch

import eiporion.eiporionkernels as kernels
from eiporion import BitLinear

CUDA_BNB = torch.cuda.is_available() and kernels._BNB_F is not None
cuda_only = pytest.mark.skipif(not CUDA_BNB, reason="requires CUDA + bitsandbytes")


class PathSpy:
    """Monkeypatches both forward paths to record which one ran."""

    def __init__(self, monkeypatch):
        self.calls = []
        for name in ("_forward_bnb", "_forward_bf16"):
            orig = getattr(kernels, name)

            def make(orig, name):
                def wrapper(*args, **kwargs):
                    self.calls.append(name)
                    return orig(*args, **kwargs)

                return wrapper

            monkeypatch.setattr(kernels, name, make(orig, name))


def float_reference(x2d, int_weight, weight_scale, bias):
    w_eff = int_weight.float() * weight_scale.float().unsqueeze(1)
    out = x2d.float() @ w_eff.t()
    if bias is not None:
        out = out + bias.float()
    return out


class FakeCtx:
    def __init__(self):
        self.saved = None

    def save_for_backward(self, *tensors):
        self.saved = tensors


@cuda_only
def test_cuda_forward_dispatches_to_bnb_int8_path(monkeypatch):
    spy = PathSpy(monkeypatch)
    lin = BitLinear(64, 32, bias=True).cuda()
    x = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
    lin(x)
    assert spy.calls == ["_forward_bnb"]


@cuda_only
def test_forward_bnb_matches_float_reference():
    lin = BitLinear(256, 128, bias=True).cuda()
    handle = int(lin._bit_handle.item())
    x = torch.randn(64, 256, device="cuda", dtype=torch.float32)
    ctx = FakeCtx()
    out = kernels._forward_bnb(
        ctx, x, lin.int_weight, lin.weight_scale, lin.bias, handle
    )
    ref = float_reference(x, lin.int_weight, lin.weight_scale, lin.bias)
    rel_err = (out.float() - ref).norm() / ref.norm()
    assert rel_err < 0.02, f"bnb int8 path diverged from reference: {rel_err}"


@cuda_only
def test_bnb_weight_cache_follows_int_weight_updates():
    lin = BitLinear(64, 32).cuda()
    handle = int(lin._bit_handle.item())
    x = torch.randn(8, 64, device="cuda", dtype=torch.float32)
    ctx = FakeCtx()
    out_before = kernels._forward_bnb(
        ctx, x, lin.int_weight, lin.weight_scale, None, handle
    )
    with torch.no_grad():
        lin.int_weight.fill_(3)
        lin.weight_scale.fill_(0.01)
    kernels._invalidate_weight_cache(handle)
    out_after = kernels._forward_bnb(
        ctx, x, lin.int_weight, lin.weight_scale, None, handle
    )
    ref_after = float_reference(x, lin.int_weight, lin.weight_scale, None)
    assert not torch.allclose(out_before.float(), out_after.float(), atol=1e-3)
    rel_err = (out_after.float() - ref_after).norm() / ref_after.norm()
    assert rel_err < 0.02, f"stale bnb weight cache: {rel_err}"


@cuda_only
def test_weight_cache_does_not_serve_stale_shape():
    # A handle must never return a cached quantisation for a different-shaped weight.
    handle = kernels.next_bit_handle()
    w1, s1 = kernels.quantize_fp_to_int8(torch.randn(32, 64))
    qw1, _ = kernels._cached_weight_quant(handle, w1.cuda(), s1.cuda())
    w2, s2 = kernels.quantize_fp_to_int8(torch.randn(128, 256))
    qw2, _ = kernels._cached_weight_quant(handle, w2.cuda(), s2.cuda())
    assert qw1.shape == (32, 64)
    assert qw2.shape == (128, 256)


@cuda_only
def test_release_bit_handle_clears_weight_cache():
    handle = kernels.next_bit_handle()
    w1, s1 = kernels.quantize_fp_to_int8(torch.randn(32, 64))
    kernels._cached_weight_quant(handle, w1.cuda(), s1.cuda())
    assert handle in kernels._BNB_WCACHE
    kernels.release_bit_handle(handle)
    assert handle not in kernels._BNB_WCACHE


def test_cpu_forward_falls_back_to_bf16(monkeypatch):
    spy = PathSpy(monkeypatch)
    lin = BitLinear(64, 32, bias=True)
    x = torch.randn(16, 64, dtype=torch.float32)
    out = lin(x)
    assert spy.calls == ["_forward_bf16"]
    ref = float_reference(x, lin.int_weight, lin.weight_scale, lin.bias)
    assert torch.allclose(out.float(), ref, atol=0.5)


@cuda_only
def test_cuda_backward_populates_weight_grad_cache():
    lin = BitLinear(64, 32, bias=True).cuda()
    handle = int(lin._bit_handle.item())
    kernels.consume_bit_grad(handle)  # clear any stale entry
    x = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
    lin(x).sum().backward()
    grad_w = lin.consume_weight_grad()
    assert grad_w is not None
    assert grad_w.shape == (32, 64)
    assert lin.bias.grad is not None and lin.bias.grad.shape == (32,)


def test_cpu_backward_populates_weight_grad_cache():
    lin = BitLinear(64, 32, bias=True)
    handle = int(lin._bit_handle.item())
    kernels.consume_bit_grad(handle)
    x = torch.randn(16, 64, dtype=torch.float32)
    lin(x).sum().backward()
    grad_w = lin.consume_weight_grad()
    assert grad_w is not None
    assert grad_w.shape == (32, 64)
