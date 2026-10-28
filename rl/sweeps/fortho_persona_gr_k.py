"""Box k of sweeps/fortho_persona_gr.py: S = 0.01, anchor penalty off (completion positions), 3 seeds."""
from sweeps.fortho_persona_gr import make

runs = make((0.01,), penalty=False)
per_gpu = 6
no_baseline = True
