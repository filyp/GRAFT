# CPU check of forget_ortho_edit: c_t vs autograd, and per-module dot ratio after the edit (others at pre-edit).
import copy, sys, torch
sys.path.insert(0, "/srv/projects/GRAFT/rl")
from gradient_routing import DualMLPAdapter, forget_ortho_edit
torch.manual_seed(0)
H, N, T = 24, 4, 50       # module dims 96 params each; T < 96 so exact constraints are feasible
base = torch.nn.Sequential(torch.nn.Linear(H, H))
ad = DualMLPAdapter(base, H, N, N).double()
torch.nn.init.normal_(ad.down_forget.weight, std=0.3)
x, g = torch.randn(T, H, dtype=torch.double), torch.randn(T, H, dtype=torch.double)
g[:5] = 0                                              # dead tokens
def cs(a):
    out = {}
    for w in ("up", "gate", "down"):
        W = getattr(a, f"{w}_forget").weight
        c = []
        for t in range(T):
            y = a._forget_core(x[t:t + 1])
            gr, = torch.autograd.grad((y * g[t:t + 1]).sum(), W)
            c.append((gr * W).sum().item())
        out[w] = torch.tensor(c, dtype=torch.double)
    return out
for s in (1.0, 0.3):
    a2 = copy.deepcopy(ad)
    c0 = cs(a2)
    st = forget_ortho_edit({a2: ([x.clone()], [g.clone()])}, s)
    print("s", s, {w: round(st[w]["dot_ratio"], 4) for w in c0}, "n_live", st["n_live"])
    for w in ("up", "gate", "down"):          # recompute module w's c with only w edited
        a3 = copy.deepcopy(ad); getattr(a3, f"{w}_forget").weight.data.copy_(getattr(a2, f"{w}_forget").weight.data)
        c1 = cs(a3)[w]
        print(f"  {w}: max|c_after - (1-s)c| = {(c1 - (1 - s) * c0[w]).abs().max():.2e}   max|c| = {c0[w].abs().max():.2e}")
# gate: is the residual only the nonlinearity? analytic c at pre-edit weights vs autograd, and with a tiny s
import torch.nn.functional as F
a = x @ ad.gate_forget.weight.T; u = x @ ad.up_forget.weight.T; sig = torch.sigmoid(a)
dh = g @ ad.down_forget.weight
c_an = ((dh * u * sig * (1 + a * (1 - sig))) * a).sum(1)
print("gate analytic vs autograd:", (c_an - cs(ad)["gate"]).abs().max().item())
for s in (0.01,):
    a2 = copy.deepcopy(ad); c0 = cs(a2)["gate"]
    forget_ortho_edit({a2: ([x.clone()], [g.clone()])}, s)
    a3 = copy.deepcopy(ad); a3.gate_forget.weight.data.copy_(a2.gate_forget.weight.data)
    print(f"s={s} gate: max|c_after-(1-s)c| / (s max|c|) = {((cs(a3)['gate'] - (1 - s) * c0).abs().max() / (s * c0.abs().max())).item():.3f}")
