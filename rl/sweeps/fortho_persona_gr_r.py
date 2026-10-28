"""Box r of sweeps/fortho_persona_gr.py: S = 0.001, seeds 1-3 (completion positions)."""
from sweeps.fortho_persona_gr import make

runs = make((0.001,))
per_gpu = 6
no_baseline = True
