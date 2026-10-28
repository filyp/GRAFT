"""Box m2 of sweeps/fortho_persona_gr.py: S = 0.1, edits from 50% of training, seeds 4-6 (completion positions)."""
from sweeps.fortho_persona_gr import make

runs = make((0.1,), start=0.5, seeds=(4, 5, 6))
per_gpu = 6
no_baseline = True
