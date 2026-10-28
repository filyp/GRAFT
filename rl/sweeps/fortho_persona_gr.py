"""GRAFT persona_qa (3 seeds), as sweeps/repro_persona_cities_gr.py, plus --forget_ortho_s S: after every optimizer
step, the FORGET MLP-adapter up/gate/down weights are edited so that, per anchor (coherence) token, the FORGET
module's first-order effect on the anchor loss shrinks by a factor (1 - S) (gradient_routing.forget_ortho_edit).
Constrained positions: those generating completion tokens (--forget_ortho_tokens completion).
S in {1, 0.3, 0.1, 0.03}; run names get '_fo<S>' before the seed suffix. One box per S:
fortho_persona_gr_a (S = 1), _b (0.3), _c (0.1), _d (0.03).
Box e: S = 0.1 with all anchor positions, prompt included (--forget_ortho_tokens all); run names get '_foall<S>'.
Boxes f, g: the anchor's monitor penalty off (coherence_rh_mode none): f = plain GRAFT (S = 0), g = S = 0.1.
Box h: plain GRAFT (S = 0) with the penalty on, the control for a-e.
Boxes i, j: S = 0.01, 0.003. Seeds 4-6: o, p, q (S = 0.003, 0.01, 0.03), t (plain GRAFT);
S = 0.001: r (seeds 1-3), s (seeds 4-6).
Box v2: rerun of v seed 2 with the non-finite guard (v seed 2 went NaN).
Anchor penalty off: u (S = 0, seeds 4-6), v / w (S = 0.001, seeds 1-3 / 4-6), x (S = 0.003, seeds 4-6), y (S = 0.01, seeds 4-6). Boxes k, l: S = 0.01, 0.003 with the anchor penalty off.
Box m: S = 0.1 with the edits starting at 50% of training (--forget_ortho_start_frac 0.5; '_from0.5').
Boxes m2 (S = 0.1 from 50%, seeds 4-6), m3 (from 25%), m4 (from 75%).
"""
import re

from sweeps.repro_persona_cities_gr import runs as _runs

S_ALL = (1.0, 0.3, 0.1, 0.03)


def _seeded(seeds):
    """The persona_qa run of sweeps/repro_persona_cities_gr.py (seed 1) with each given seed."""
    base = next(r for r in _runs if r["run_name"].startswith("persona_qa") and r["seed"] == 1)
    return [{**base, "seed": k, "run_name": re.sub(r"_s1$", f"_s{k}", base["run_name"])} for k in seeds]


def make(s_values, tokens="completion", penalty=True, start=0.0, seeds=(1, 2, 3)):
    """s = 0 is plain GRAFT (no edit, tag '_fo0'). penalty=False turns the anchor's -2 monitor penalty off
    (coherence_rh_mode none; run names get '_nopen')."""
    tag = "fo" if tokens == "completion" else f"fo{tokens}"
    pen = {} if penalty else {"coherence_rh_mode": "none"}
    sfx = ("" if penalty else "_nopen") + (f"_from{start:g}" if start else "")
    out = [{**r, **pen, "forget_ortho_s": s, "forget_ortho_tokens": tokens, "forget_ortho_start_frac": start,
            "run_name": re.sub(r"_s(\d+)$", rf"_{tag}{s:g}{sfx}_s\1", r["run_name"])}
           for s in s_values for r in _seeded(seeds)]
    assert len(out) == len(seeds) * len(s_values)
    return out


runs = make(S_ALL)
per_gpu = 2
no_baseline = True
