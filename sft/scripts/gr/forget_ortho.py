"""Per-token orthogonalization of the FORGET branch against anchor gradients (SFT port of
rl/gradient_routing.py::forget_ortho_edit; see reports/2026.10.28_selectivity.md).

On an anchor (CLASS_RETAIN) example, hooks on every DualMLPAdapter read the MLP input x_t (forward pre-hook) and
the gradient g_t at the MLP output (full backward hook) at the positions whose logits the loss scores. FORGET is
replayed at scale 1 on x_t; each FORGET module W (up, gate, down) gets per-token (input i_t, output grad o_t):
  down: i = SiLU(gate_F x) * up_F x, o = g;   up: i = x, o = (down_F^T g) * SiLU(gate_F x);
  gate: i = x, o = (down_F^T g) * up_F x * SiLU'(gate_F x).
The edit is the minimum-norm dW with <W + dW, o_t i_t^T> = (1 - s) <W, o_t i_t^T> for all tokens of the example
jointly: dW = sum_t a_t o_t i_t^T, (K + lam I) a = -s c, K_ts = (o_t.o_s)(i_t.i_s), lam = ridge * mean diag(K).
Gram factors in float32 (TF32 off; float64 if non-finite), K / Cholesky / dW in float64.

Each layer's dW is computed inside the backward hook (so x, g are not held for all layers) from the weights at
that time, and accumulated. After the optimizer step, `apply` sums the accumulated dW over DDP ranks and adds it to
the FORGET weights on every rank (identical on all ranks). Several anchor examples in one step are summed, which
approximates the joint solve.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from scripts.gr.adapter import DualMLPAdapter


@torch.no_grad()
def forget_ortho_delta(ad: DualMLPAdapter, x: torch.Tensor, g: torch.Tensor, s: float, ridge: float = 1e-6,
                       refine: int = 1):
    """{module: dW (float64, weight shape)} and stats {module: (c2, c2_after, dW2, W2)} for one adapter, from the
    per-token MLP input x [T, H] and output grad g [T, H]."""
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        x, g = x.double(), g.double()
        Wg, Wu, Wd = ad.gate_forget.weight, ad.up_forget.weight, ad.down_forget.weight
        a, u = x @ Wg.double().T, x @ Wu.double().T
        sa, sig = F.silu(a), torch.sigmoid(a)
        dh = g @ Wd.double()
        gram = lambda v: (lambda v32: v32 @ v32.T)(v.float())
        XX = gram(x)
        pairs = {"down": (g, sa * u, Wd, gram(g), None), "up": (dh * sa, x, Wu, None, XX),
                 "gate": (dh * u * sig * (1 + a * (1 - sig)), x, Wg, None, XX)}
        deltas, stats = {}, {}
        for w, (o, i, W, OO, II) in pairs.items():
            Wf = W.double()
            c = ((o @ Wf) * i).sum(1)
            K = (OO if OO is not None else gram(o)).double() * (II if II is not None else gram(i)).double()
            if not torch.isfinite(K).all():
                K = (o @ o.T) * (i @ i.T)
            live = K.diagonal() > 0
            if not live.any():
                continue
            Kl, cl = K[live][:, live], c[live]
            lam = ridge * Kl.diagonal().mean()
            Kl.diagonal().add_(lam)
            while True:
                Lc, info = torch.linalg.cholesky_ex(Kl)
                if info == 0:
                    break
                Kl.diagonal().add_(9 * lam); lam = 10 * lam
            rhs = (-s * cl)[:, None]
            al = torch.cholesky_solve(rhs, Lc)
            for _ in range(refine):
                al = al + torch.cholesky_solve(rhs - Kl @ al, Lc)
            dW = o[live].T @ (al[:, 0, None] * i[live])
            if not torch.isfinite(dW).all():
                continue
            c_after = c + K[:, live] @ al[:, 0]
            deltas[w] = dW
            stats[w] = (c.pow(2).sum().item(), c_after.pow(2).sum().item(), dW.pow(2).sum().item(),
                        Wf.pow(2).sum().item())
        return deltas, stats
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


class ForgetOrthoCollector:
    """Hooks for one anchor example (`arm(mask)` before its forward, `disarm()` after its backward); `apply` after
    the optimizer step."""

    def __init__(self, model, s: float):
        self.s = s
        self.adapters = list(m for m in model.modules() if isinstance(m, DualMLPAdapter))
        self.acc = {}                      # id(adapter) -> {module: dW}
        self.stats = {w: [0.0, 0.0, 0.0, 0.0] for w in ("up", "gate", "down")}
        self.n_tok = 0
        self.n_examples = 0
        self.sec = 0.0
        self._handles, self._saved, self.mask = [], {}, None

    def arm(self, mask: torch.Tensor):
        """mask: (T,) bool, positions whose logits the loss scores."""
        self.mask = mask
        self.n_tok += int(mask.sum())
        self.n_examples += 1
        for ad in self.adapters:
            self._handles += [ad.register_forward_pre_hook(self._save), ad.register_full_backward_hook(self._bwd)]

    def disarm(self):
        for h in self._handles:
            h.remove()
        self._handles, self._saved = [], {}

    def _save(self, mod, args):
        self._saved[id(mod)] = args[0].detach()

    def _bwd(self, mod, grad_input, grad_output):
        x, g = self._saved.pop(id(mod), None), grad_output[0]
        if x is None or g is None:
            return
        t0 = time.perf_counter()
        H = x.shape[-1]
        xs, gs = x.reshape(-1, H)[self.mask], g.detach().reshape(-1, H)[self.mask]
        deltas, stats = forget_ortho_delta(mod, xs, gs, self.s)
        acc = self.acc.setdefault(id(mod), {})
        for w, dW in deltas.items():
            acc[w] = dW if w not in acc else acc[w] + dW
        for w, st in stats.items():
            for k in range(4):
                self.stats[w][k] += st[k]
        self.sec += time.perf_counter() - t0

    @torch.no_grad()
    def apply(self, accelerator):
        """Sum the accumulated dW over ranks and add it to the FORGET weights on every rank. Returns a log dict
        (local stats) or None if no rank had anchors."""
        n = torch.tensor([float(self.n_examples)], device=accelerator.device)
        n = accelerator.reduce(n, reduction="sum")
        if n.item() == 0:
            return None
        for ad in self.adapters:
            acc = self.acc.get(id(ad), {})
            for w in ("up", "gate", "down"):
                W = getattr(ad, f"{w}_forget").weight
                dW = acc.get(w, torch.zeros(W.shape, dtype=torch.float64, device=W.device))
                dW = accelerator.reduce(dW.to(W.device), reduction="sum")
                W.copy_((W.double() + dW).to(W.dtype))
        log = {"ortho/n_examples_global": n.item(), "ortho/n_tok_local": self.n_tok, "ortho/sec_local": self.sec}
        for w, st in self.stats.items():
            if st[0]:
                log[f"ortho/{w}_dot_ratio_local"] = (st[1] / st[0]) ** 0.5
            if st[3]:
                log[f"ortho/{w}_rel_dW_local"] = (st[2] / st[3]) ** 0.5
        self.acc, self.n_tok, self.n_examples, self.sec = {}, 0, 0, 0.0
        self.stats = {w: [0.0, 0.0, 0.0, 0.0] for w in ("up", "gate", "down")}
        return log
