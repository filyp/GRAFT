"""Box u of sweeps/fortho_persona_gr.py: plain GRAFT (S = 0), anchor penalty off, seeds 4-6 (completion positions)."""
from sweeps.fortho_persona_gr import make

runs = make((0,), penalty=False, seeds=(4, 5, 6))
per_gpu = 6
no_baseline = True
