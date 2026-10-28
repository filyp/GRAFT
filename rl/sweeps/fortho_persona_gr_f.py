"""Box f of sweeps/fortho_persona_gr.py: plain GRAFT (no edit) with the anchor's monitor penalty off, 3 seeds."""
from sweeps.fortho_persona_gr import make

runs = make((0,), penalty=False)
per_gpu = 6
no_baseline = True
