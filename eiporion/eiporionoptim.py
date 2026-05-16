import torch
from bitsandbytes.optim import AdamW8bit

from .bitLinear import BitLinear
from .eiporionkernels import update_int8_weight_, _invalidate_weight_cache


# ---- 8-bit blockwise helpers with mu-law companding ----
_MU = 255.0
_1_OVER_LN1P_MU = 1.0 / 5.545177444479562  # 1 / ln(1 + 255)


def _pad_to_block_multiple(t: torch.Tensor, blocksize: int, num_blocks: int):
    padded_size = num_blocks * blocksize
    if padded_size > t.numel():
        return torch.cat(
            [t, torch.zeros(padded_size - t.numel(), device=t.device, dtype=t.dtype)]
        )
    return t


def _quantize_blockwise_signed(x: torch.Tensor, blocksize: int):
    x_flat = x.float().contiguous().view(-1)
    numel = x_flat.numel()
    num_blocks = (numel + blocksize - 1) // blocksize
    x_padded = _pad_to_block_multiple(x_flat, blocksize, num_blocks)
    x_blocks = x_padded.view(num_blocks, blocksize)
    absmax = x_blocks.abs().amax(dim=1).clamp_min(1e-12)

    # mu-law companding: normalise → compress → quantise
    x_norm = (x_blocks / absmax.unsqueeze(1)).clamp(-1.0, 1.0)
    x_comp = torch.sign(x_norm) * torch.log1p(_MU * x_norm.abs()) * _1_OVER_LN1P_MU

    q_blocks = torch.round(x_comp * 127.0).clamp(-127, 127).to(torch.int16)
    q_flat = (q_blocks.view(-1)[:numel] + 128).to(torch.uint8)
    return q_flat.view_as(x), absmax


def _dequantize_blockwise_signed(q, absmax, blocksize, shape):
    q_flat = q.contiguous().view(-1)
    numel = q_flat.numel()
    num_blocks = absmax.numel()
    q_padded = _pad_to_block_multiple(q_flat, blocksize, num_blocks)
    q_blocks = q_padded.view(num_blocks, blocksize).float()

    # inverse mu-law
    y = ((q_blocks - 128.0) / 127.0).clamp(-1.0, 1.0)
    x_norm = torch.sign(y) * (torch.exp(y.abs() / _1_OVER_LN1P_MU) - 1.0) / _MU
    out_blocks = x_norm * absmax.unsqueeze(1)
    return out_blocks.view(-1)[:numel].view(shape)


def _quantize_blockwise_unsigned(x: torch.Tensor, blocksize: int):
    x_flat = x.float().contiguous().view(-1)
    numel = x_flat.numel()
    num_blocks = (numel + blocksize - 1) // blocksize
    x_padded = _pad_to_block_multiple(x_flat, blocksize, num_blocks)
    x_blocks = x_padded.view(num_blocks, blocksize)
    absmax = x_blocks.amax(dim=1).clamp_min(1e-12)

    # mu-law companding (unsigned: [0, 1] range)
    x_norm = (x_blocks / absmax.unsqueeze(1)).clamp(0.0, 1.0)
    x_comp = torch.log1p(_MU * x_norm) * _1_OVER_LN1P_MU

    q_blocks = torch.round(x_comp * 255.0).clamp(0, 255)
    q_flat = q_blocks.view(-1)[:numel].to(torch.uint8)
    return q_flat.view_as(x), absmax


def _dequantize_blockwise_unsigned(q, absmax, blocksize, shape):
    q_flat = q.contiguous().view(-1)
    numel = q_flat.numel()
    num_blocks = absmax.numel()
    q_padded = _pad_to_block_multiple(q_flat, blocksize, num_blocks)
    q_blocks = q_padded.view(num_blocks, blocksize).float()

    y = (q_blocks / 255.0).clamp(0.0, 1.0)
    x_norm = (torch.exp(y / _1_OVER_LN1P_MU) - 1.0) / _MU
    out_blocks = x_norm * absmax.unsqueeze(1)
    return out_blocks.view(-1)[:numel].view(shape)


class EiporionOptim(AdamW8bit):
    """bnb AdamW8bit for dense params + 8-bit AdamW + DQT-SR for INT8 weights."""

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.1,
        bit_modules=None,
        block_size=256,
        sr_bias_scale=0.15,
        guarantee_every_n_steps=200,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.bit_modules = list(bit_modules) if bit_modules is not None else []
        self._bit_handles = (
            [int(m._bit_handle.item()) for m in self.bit_modules]
            if bit_modules is not None
            else []
        )
        self._bit_state: dict[int, dict[str, torch.Tensor | int]] = {}
        self.block_size = int(block_size)
        self.sr_bias_scale = float(sr_bias_scale)
        self.guarantee_every_n_steps = int(guarantee_every_n_steps)
        if self.guarantee_every_n_steps <= 0:
            raise ValueError(
                "guarantee_every_n_steps must be > 0, "
                f"got {self.guarantee_every_n_steps}"
            )

    def add_bit_modules(self, modules) -> None:
        for module in modules:
            if not isinstance(module, BitLinear):
                raise TypeError(f"expected BitLinear, got {type(module).__name__}")
            if module not in self.bit_modules:
                self.bit_modules.append(module)
                self._bit_handles.append(int(module._bit_handle.item()))

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)

        if not self.bit_modules:
            return loss

        group = self.param_groups[0]
        lr = float(group["lr"])
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        wd = float(group["weight_decay"])
        bs = self.block_size

        for module, handle in zip(self.bit_modules, self._bit_handles):
            g = module.consume_weight_grad()
            if g is None:
                continue
            g = g.float().contiguous()

            state = self._bit_state.setdefault(handle, {})
            if "step" not in state:
                state["step"] = 0
                # 8-bit m, v (matching AdamW8bit scheme)
                m_q, m_absmax = _quantize_blockwise_signed(
                    torch.zeros_like(g, dtype=torch.float32), bs
                )
                v_q, v_absmax = _quantize_blockwise_unsigned(
                    torch.zeros_like(g, dtype=torch.float32), bs
                )
                state["m_q"] = m_q
                state["m_absmax"] = m_absmax
                state["v_q"] = v_q
                state["v_absmax"] = v_absmax
                # residual also 8-bit blockwise, same scheme as m
                r_q, r_absmax = _quantize_blockwise_signed(
                    torch.zeros_like(g, dtype=torch.float32), bs
                )
                state["r_q"] = r_q
                state["r_absmax"] = r_absmax

            m = _dequantize_blockwise_signed(
                state["m_q"], state["m_absmax"], bs, g.shape
            )
            v = _dequantize_blockwise_unsigned(
                state["v_q"], state["v_absmax"], bs, g.shape
            )
            residual = _dequantize_blockwise_signed(
                state["r_q"], state["r_absmax"], bs, g.shape
            )
            state["step"] += 1
            t = state["step"]

            m.mul_(beta1).add_(g, alpha=1.0 - beta1)
            v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
            m_hat = m / (1.0 - beta1**t)
            v_hat = v / (1.0 - beta2**t)

            ws = module.weight_scale.float().unsqueeze(1).clamp_min(eps)
            iw = module.int_weight.float()
            adam_term = m_hat / (v_hat.sqrt() + eps)
            delta_w_eff = -lr * (adam_term + wd * iw * ws)
            residual = residual + delta_w_eff / ws

            abs_res = residual.abs()
            base = torch.floor(abs_res)
            frac = abs_res - base
            # Momentum-biased SR
            bias = torch.tanh(adam_term) * self.sr_bias_scale * torch.sign(residual)
            frac_biased = (frac + bias).clamp(0.0, 1.0)
            extra = (torch.rand_like(frac) < frac_biased).float()
            delta_q = (torch.sign(residual) * (base + extra)).to(torch.int32)

            if torch.any(delta_q != 0):
                update_int8_weight_(module.int_weight, delta_q)
                residual = residual - delta_q.float()
                _invalidate_weight_cache(handle)

            if (
                t % self.guarantee_every_n_steps == 0
                and module.guarantee_int8_(
                    saturation_q=126, saturation_ratio=0.15, eps=eps
                )
            ):
                new_ws = module.weight_scale.float().unsqueeze(1).clamp_min(eps)
                # Preserve pending effective update when int-domain scale changes.
                residual.mul_(ws / new_ws)

            r_q, r_absmax = _quantize_blockwise_signed(residual, bs)
            state["r_q"] = r_q
            state["r_absmax"] = r_absmax

            m_q, m_absmax = _quantize_blockwise_signed(m, bs)
            v_q, v_absmax = _quantize_blockwise_unsigned(v, bs)
            state["m_q"] = m_q
            state["m_absmax"] = m_absmax
            state["v_q"] = v_q
            state["v_absmax"] = v_absmax

        return loss

    def state_dict(self):
        state = super().state_dict()
        state["bit_state"] = {
            int(handle): {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in per_handle.items()
            }
            for handle, per_handle in self._bit_state.items()
        }
        return state

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        bit_state = state_dict.pop("bit_state", {})
        super().load_state_dict(state_dict)
        self._bit_state = {}
        for handle, per_handle in bit_state.items():
            self._bit_state[int(handle)] = {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in per_handle.items()
            }


class EiporionOptimSR(EiporionOptim):
    """EiporionOptim with unbiased stochastic rounding (no momentum bias)."""

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.1,
        bit_modules=None,
        block_size=256,
        guarantee_every_n_steps=200,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            bit_modules=bit_modules,
            block_size=block_size,
            sr_bias_scale=0.0,
            guarantee_every_n_steps=guarantee_every_n_steps,
        )
