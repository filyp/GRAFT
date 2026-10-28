"""Box m3 of sweeps/fortho_persona_gr.py: S = 0.1, edits from 25% of training, seeds 1-3 (completion positions)."""
from sweeps.fortho_persona_gr import make

runs = make((0.1,), start=0.25)
per_gpu = 6
no_baseline = True
