import torch


# ---------------------------------------------------------------------------
# Handle registry
# ---------------------------------------------------------------------------
_NEXT_HANDLE = 1
_REGISTERED_HANDLES: set[int] = set()
_BIT_GRAD_CACHE: dict[int, torch.Tensor] = {}


def next_bit_handle() -> int:
    global _NEXT_HANDLE
    while _NEXT_HANDLE in _REGISTERED_HANDLES:
        _NEXT_HANDLE += 1
    handle = _NEXT_HANDLE
    _REGISTERED_HANDLES.add(handle)
    _NEXT_HANDLE += 1
    return handle


def register_bit_handle(handle: int) -> int:
    global _NEXT_HANDLE
    handle = int(handle)
    if handle <= 0:
        raise ValueError(f"handle must be > 0, got {handle}")
    _REGISTERED_HANDLES.add(handle)
    _BIT_GRAD_CACHE.pop(handle, None)
    if handle >= _NEXT_HANDLE:
        _NEXT_HANDLE = handle + 1
    return handle


def release_bit_handle(handle: int) -> None:
    handle = int(handle)
    _REGISTERED_HANDLES.discard(handle)
    _BIT_GRAD_CACHE.pop(handle, None)
    _BNB_WCACHE.pop(handle, None)
    _BNB_WVERSION.pop(handle, None)


def consume_bit_grad(handle: int) -> torch.Tensor | None:
    return _BIT_GRAD_CACHE.pop(int(handle), None)


# ---------------------------------------------------------------------------
# bitsandbytes backend
# ---------------------------------------------------------------------------
_BNB_F = None

try:
    import bitsandbytes.functional as _BNB_F
except ImportError:
    pass

# Weight-quantisation cache  (handle → (qw, scw, version))
_BNB_WCACHE: dict[int, tuple] = {}
_BNB_WVERSION: dict[int, int] = {}


def _cached_weight_quant(
    handle: int, int_weight: torch.Tensor, weight_scale: torch.Tensor
):
    """Return (qw_int8, scw) for the current weight; re-quantise only if stale."""
    cur_ver = _BNB_WVERSION.get(handle, 0)
    entry = _BNB_WCACHE.get(handle)
    if entry is not None:
        qw, scw, cached_ver = entry
        if cached_ver == cur_ver and qw.shape == int_weight.shape:
            return qw, scw

    # bitsandbytes expects fp16 input for int8_vectorwise_quant
    w_fp16 = int_weight.to(torch.float16) * weight_scale.to(torch.float16).unsqueeze(1)
    qw, scw, _ = _BNB_F.int8_vectorwise_quant(w_fp16)
    _BNB_WCACHE[handle] = (qw, scw, cur_ver)
    return qw, scw


def _invalidate_weight_cache(handle: int):
    """Call after int_weight is modified by stochastic rounding."""
    _BNB_WVERSION[handle] = _BNB_WVERSION.get(handle, 0) + 1


# ---------------------------------------------------------------------------
# Weight quantisation  (static, used at init / reset)
# ---------------------------------------------------------------------------


def quantize_fp_to_int8(weight: torch.Tensor, eps: float = 1e-8):
    if weight.ndim != 2:
        raise ValueError(
            f"weight must be 2D [out_features, in_features], got {tuple(weight.shape)}"
        )
    w = weight.float()
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(float(eps)) / 127.0
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return q.contiguous(), scale.squeeze(1).contiguous()


@torch.no_grad()
def check_high_saturation(
    int_weight: torch.Tensor,
    saturation_q: int = 126,
    saturation_ratio: float = 0.15,
) -> bool:
    """Check whether near-limit INT8 occupancy is high."""
    if int_weight.ndim != 2:
        raise ValueError(
            f"int_weight must be 2D [out_features, in_features], got {tuple(int_weight.shape)}"
        )
    if not (1 <= int(saturation_q) <= 127):
        raise ValueError(f"saturation_q must be in [1, 127], got {saturation_q}")
    if not (0.0 <= float(saturation_ratio) <= 1.0):
        raise ValueError(
            f"saturation_ratio must be in [0.0, 1.0], got {saturation_ratio}"
        )
    near_limit = int_weight.abs() >= int(saturation_q)
    sat_ratio = near_limit.float().mean().item()
    return sat_ratio >= float(saturation_ratio)


@torch.no_grad()
def recalibrate_weight_scale_(
    int_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    eps: float = 1e-8,
) -> None:
    """Recompute weight_scale from effective weight as per-row max_abs / 127."""
    if int_weight.ndim != 2:
        raise ValueError(
            f"int_weight must be 2D [out_features, in_features], got {tuple(int_weight.shape)}"
        )
    if weight_scale.ndim != 1 or weight_scale.shape[0] != int_weight.shape[0]:
        raise ValueError(
            f"weight_scale must be [out_features], got {tuple(weight_scale.shape)}"
        )
    eff_weight = int_weight.float() * weight_scale.float().unsqueeze(1)
    recalib_int, recalib_scale = quantize_fp_to_int8(eff_weight, eps=eps)
    int_weight.copy_(recalib_int.to(device=int_weight.device))
    weight_scale.copy_(
        recalib_scale.to(device=weight_scale.device, dtype=weight_scale.dtype)
    )


@torch.no_grad()
def guarantee_weight_scale_headroom_(
    int_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    handle: int | None = None,
    saturation_q: int = 126,
    saturation_ratio: float = 0.15,
    eps: float = 1e-8,
) -> bool:
    """Guarantee function: check saturation and recalibrate scale when needed."""
    if not check_high_saturation(
        int_weight=int_weight,
        saturation_q=saturation_q,
        saturation_ratio=saturation_ratio,
    ):
        return False

    recalibrate_weight_scale_(
        int_weight=int_weight, weight_scale=weight_scale, eps=eps
    )
    if handle is not None:
        _invalidate_weight_cache(int(handle))
    return True


# ---------------------------------------------------------------------------
# Int8LinearFn
# ---------------------------------------------------------------------------


class Int8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x2d: torch.Tensor,  # [N, K]  (autocast doesn't cover custom Fns)
        int_weight: torch.Tensor,  # [O, K]  int8
        weight_scale: torch.Tensor,  # [O]     float  (trainable Parameter)
        bias: torch.Tensor | None,  # [O]     float
        handle: int,
    ):
        if _BNB_F is not None and x2d.is_cuda:
            return _forward_bnb(ctx, x2d, int_weight, weight_scale, bias, handle)
        return _forward_bf16(ctx, x2d, int_weight, weight_scale, bias, handle)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return _backward_impl(ctx, grad_out)


# ---------------------------------------------------------------------------
# Path A: bitsandbytes cuBLASLt INT8 Tensor Cores
# ---------------------------------------------------------------------------


def _forward_bnb(ctx, x2d, int_weight, weight_scale, bias, handle):
    # 1. Quantise activation — bitsandbytes works in fp16
    qx, scx, _ = _BNB_F.int8_vectorwise_quant(x2d.half())

    # 2. Get cached weight quantisation
    qw, scw = _cached_weight_quant(handle, int_weight, weight_scale)

    # 3. INT8 matmul via cuBLASLt: out_i32 = qx @ qw.T
    out_i32 = _BNB_F.int8_linear_matmul(qx, qw)

    # 4. Dequantise — output = out_i32 * activation_scale * weight_scale (+ bias)
    out = _BNB_F.int8_mm_dequant(
        out_i32, scx, scw, bias=bias.half() if bias is not None else None
    )

    ctx.save_for_backward(x2d.to(torch.bfloat16), int_weight, weight_scale)
    ctx.handle = int(handle)
    ctx.has_bias = bias is not None
    ctx.input_dtype = x2d.dtype
    return out.to(dtype=x2d.dtype)


# ---------------------------------------------------------------------------
# Path B: BF16 fallback  (torch.matmul on BF16 Tensor Cores, works everywhere)
# ---------------------------------------------------------------------------


def _forward_bf16(ctx, x2d, int_weight, weight_scale, bias, handle):
    x2d_bf16 = x2d.to(torch.bfloat16)
    w_bf16 = int_weight.to(torch.bfloat16) * weight_scale.to(torch.bfloat16).unsqueeze(
        1
    )
    out = torch.matmul(x2d_bf16, w_bf16.t())
    if bias is not None:
        out.add_(bias)
    ctx.save_for_backward(x2d_bf16, int_weight, weight_scale)
    ctx.handle = int(handle)
    ctx.has_bias = bias is not None
    ctx.input_dtype = x2d.dtype
    return out


# ---------------------------------------------------------------------------
# Common backward (BF16 matmul — correct for both forward paths)
# ---------------------------------------------------------------------------


def _backward_impl(ctx, grad_out):
    x2d_bf16, int_weight, weight_scale = ctx.saved_tensors
    go_bf16 = grad_out.to(torch.bfloat16)
    ws_bf16 = weight_scale.to(torch.bfloat16)

    w_bf16 = int_weight.to(torch.bfloat16) * ws_bf16.unsqueeze(1)

    grad_in = torch.matmul(go_bf16, w_bf16)

    grad_w = torch.matmul(go_bf16.t(), x2d_bf16)
    cached = _BIT_GRAD_CACHE.get(ctx.handle)
    if cached is None:
        _BIT_GRAD_CACHE[ctx.handle] = grad_w.to(dtype=torch.bfloat16)
    else:
        cached.add_(grad_w.to(dtype=torch.bfloat16))

    grad_bias = go_bf16.sum(dim=0).to(dtype=grad_out.dtype) if ctx.has_bias else None

    # int_weight and weight_scale are fixed buffers — no gradient flows to them.
    # int_weight is updated via DQT in EiporionOptim; weight_scale stays fixed.
    return (
        grad_in.to(dtype=grad_out.dtype),
        None,  # int_weight
        None,  # weight_scale
        grad_bias,
        None,  # handle
    )


# ---------------------------------------------------------------------------
# INT8 weight update
# ---------------------------------------------------------------------------


@torch.no_grad()
def update_int8_weight_(
    int_weight: torch.Tensor,  # [O, K] int8
    delta_q: torch.Tensor,  # [O, K] int32
    weight_scale: torch.Tensor,  # [O] float
    eps: float = 1e-8,
) -> None:
    """In-place int8 weight update.

    ``int_weight += delta_q`` (clamped to [-127, 127]), then recomputes
    ``weight_scale = amax(|int_weight|, dim=1) / 127`` from the new values
    — the same principle as `m_absmax` / `v_absmax` / `r_absmax` recomputed
    every step.
    """
    result = int_weight.to(torch.int16) + delta_q.to(torch.int16)
    result.clamp_(-127, 127)
    int_weight.copy_(result.to(torch.int8))

    new_scale = int_weight.float().abs().amax(dim=1).clamp_min(float(eps)) / 127.0
    weight_scale.copy_(new_scale)
