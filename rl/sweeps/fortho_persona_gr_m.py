"""Box m of sweeps/fortho_persona_gr.py: S = 0.1, edits from 50% of training (completion positions), 3 seeds."""
from sweeps.fortho_persona_gr import make

runs = make((0.1,), start=0.5)
per_gpu = 6
no_baseline = True
