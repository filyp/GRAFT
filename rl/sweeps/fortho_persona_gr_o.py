"""Box o of sweeps/fortho_persona_gr.py: S = 0.003, seeds 4-6 (completion positions)."""
from sweeps.fortho_persona_gr import make

runs = make((0.003,), seeds=(4, 5, 6))
per_gpu = 6
no_baseline = True
