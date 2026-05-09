import math

import torch
from torch import nn

from .eiporionkernels import (
    Int8LinearFn,
    consume_bit_grad,
    next_bit_handle,
    register_bit_handle,
    release_bit_handle,
)


def _init_int8_weight(out_features: int, in_features: int):
    """Kaiming-uniform init, quantise to int8.

    Matches bnb ``int8_vectorwise_quant``: per-row max_abs / 127 scale.
    W_int8 = clip(round(W / scale), -128, 127)  with  scale = max_abs_per_row / 127.
    W_eff  = int_weight * weight_scale  ≈  original kaiming weight.
    """
    weight = torch.empty((out_features, in_features), dtype=torch.float32)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    w = weight.float()
    scale_per_row = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
    q = torch.round(w / scale_per_row).clamp(-127, 127).to(torch.int8)
    return q.contiguous(), scale_per_row.squeeze(1).contiguous()


class BitLinear(nn.Module):
    """INT8 linear layer matching bnb ``Int8Params`` + DQT paper.

    * ``int_weight``    — int8 buffer ``[O, K]``, the quantised weight (bnb's CB).
    * ``weight_scale``  — float buffer ``[O]``, per-row max_abs/127 (bnb's SCB/127).
    * Forward: ``W_eff = int_weight * weight_scale``, then standard matmul.
    * Gradients for ``int_weight`` are stashed in ``_BIT_GRAD_CACHE`` and consumed
    by :class:`EiporionOptim` for DQT stochastic rounding.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        int_init, scale_init = _init_int8_weight(
            out_features=self.out_features, in_features=self.in_features
        )
        self.register_buffer("int_weight", int_init, persistent=True)
        self.register_buffer("weight_scale", scale_init, persistent=True)
        handle = next_bit_handle()
        self.register_buffer(
            "_bit_handle",
            torch.tensor(handle, dtype=torch.int64),
            persistent=True,
        )
        self._registered_handle = (
            int(self._bit_handle.item())
            if self._bit_handle.device.type != "meta"
            else handle
        )
        self.register_load_state_dict_post_hook(self._post_load_state_dict)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def reset_int8_(self) -> None:
        int_init, scale_init = _init_int8_weight(
            out_features=self.out_features, in_features=self.in_features
        )
        self.int_weight.copy_(int_init.to(device=self.int_weight.device))
        self.weight_scale.copy_(scale_init.to(device=self.weight_scale.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input last dim {self.in_features}, got {x.shape[-1]}"
            )
        if self.int_weight.device != x.device:
            raise RuntimeError(
                "BitLinear input and int_weight must be on same device. "
                "Move module with model.to(device) before forward."
            )
        x2d = x.reshape(-1, self.in_features)
        out2d = Int8LinearFn.apply(
            x2d,
            self.int_weight,
            self.weight_scale,
            self.bias,
            int(self._bit_handle.item()),
        )
        return out2d.view(*x.shape[:-1], self.out_features)

    def consume_weight_grad(self) -> torch.Tensor | None:
        return consume_bit_grad(int(self._bit_handle.item()))

    def _post_load_state_dict(self, module, incompatible_keys) -> None:
        del module, incompatible_keys
        new_handle = register_bit_handle(int(self._bit_handle.item()))
        if new_handle != self._registered_handle:
            release_bit_handle(self._registered_handle)
            self._registered_handle = new_handle


def collect_bitlinear_modules(module: nn.Module) -> list[BitLinear]:
    return [m for m in module.modules() if isinstance(m, BitLinear)]
