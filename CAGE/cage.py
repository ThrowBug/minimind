from typing import List, Optional, Sequence, Dict, Any

import torch
from torch import Tensor
from torch.optim import Optimizer

from torch.optim.optimizer import _get_value


class CAGE:
    """
    Decoupled CAGE post-step:
        x <- x - lr_group * lambda_t * (x - Q(x))

    Call `cage.step()` AFTER `optimizer.step()`. Supports multiple optimizers.
    LRs are read from each param group's current 'lr' at call-time, so any
    LR scheduler is automatically respected.

    Args:
        optimizers: sequence of torch.optim.Optimizer instances to apply CAGE on.
        lambda_base: target lambda (float). The schedule ramps from 0 -> lambda_base.
        total_steps: total number of training steps (for linear ramp).
        silence_ratio: s in [0, 1). CAGE is silent for the first s * total_steps steps.
        schedule: "linear_ramp" or "constant".
        track_stats: if True, keeps simple running stats for logging/debugging.

    Notes:
        - Works with master-FP32 weights (recommended). If using AMP, call after the unscaled optimizer step.
        - No autograd graph is touched; uses torch.no_grad() and in-place ops.
        - Safe for DDP: all ops are local and elementwise.
    """

    def __init__(
        self,
        optimizers: Sequence[Optimizer],
        lambda_base: float,
        total_steps: int,
        silence_ratio: float = 0.9,
        schedule: str = "linear_ramp",
        track_stats: bool = False,
    ):
        assert 0.0 <= silence_ratio < 1.0, "silence_ratio must be in [0,1)"
        assert total_steps > 0, "total_steps must be positive"
        self.optimizers: List[Optimizer] = list(optimizers)
        self.lambda_base = float(lambda_base)
        self.total_steps = int(total_steps)
        self.silence_ratio = float(silence_ratio)
        self.schedule = schedule
        self.track_stats = track_stats

        self._step = 0
        self._stats = {"avg_corr_norm": 0.0, "applied_params": 0} if track_stats else None

    @staticmethod
    def _default_quantizer(p: Tensor) -> Optional[Tensor]:
        q = None
        if hasattr(p, "quantizer"):
            with torch.no_grad():
                q = p.quantizer(p)
        return q

    def _lambda_t(self) -> float:
        r = self._step / self.total_steps
        if self.schedule == "constant":
            return self.lambda_base if r > self.silence_ratio else 0.0
        elif self.schedule == "linear_ramp":
            if r <= self.silence_ratio:
                return 0.0
            ramp = (r - self.silence_ratio) / max(1e-12, (1.0 - self.silence_ratio))
            return self.lambda_base * float(min(1.0, max(0.0, ramp)))
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    @torch.no_grad()
    def step(self) -> float:
        self._step += 1
        lam = self._lambda_t()

        if self.track_stats:
            self._stats["avg_corr_norm"] = 0.0
            self._stats["avg_err_norm"] = 0.0
            self._stats["applied_params"] = 0
            self._stats["lambda"] = lam
        for opt in self.optimizers:
            for group in opt.param_groups:
                lr = _get_value(group.get("lr", None))
                for p in group["params"]:
                    q = self._default_quantizer(p)
                    if q is None:
                        continue
                    e = p - q
                    p.add_(e, alpha=-lr * lam)

                    if self.track_stats:
                        self._stats["avg_err_norm"] += float(e.norm().item())
                        self._stats["avg_corr_norm"] += float(e.norm().item() * lam)
                        self._stats["applied_params"] += 1
        if self.track_stats:
            self._stats["avg_corr_norm"] /= max(1, self._stats["applied_params"])
            self._stats["avg_err_norm"] /= max(1, self._stats["applied_params"])
        return lam

    def get_stats(self) -> Dict[str, Any]:
        return self._stats
