"""Box l of sweeps/fortho_persona_gr.py: S = 0.003, anchor penalty off (completion positions), 3 seeds."""
from sweeps.fortho_persona_gr import make

runs = make((0.003,), penalty=False)
per_gpu = 6
no_baseline = True
