"""Sharpness-Aware Minimization (Foret et al. 2020), wrapping a base
optimizer (AdamW here). Seeks minima that are flat, not just low, on the
theory that flat minima generalize better — a reasonable bet specifically
because MCR-SL is small (N=234) and fold-to-fold variance is already large,
so a training run that lands in a sharp minimum is plausibly a source of
that variance.

Standard reference implementation (davda54/sam), adapted. Requires two
forward-backward passes per step — see train.py's run_epoch for the
two-step training loop this needs (first_step perturbs weights toward the
worst-case neighborhood, second_step evaluates the gradient there and
steps the base optimizer, then reverts the perturbation).
"""
import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **base_optimizer_kwargs):
        assert rho >= 0, f"rho must be non-negative, got {rho}"
        defaults = dict(rho=rho, **base_optimizer_kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_optimizer_kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        return torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"] if p.grad is not None
            ]), p=2,
        )

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
