"""Box g of sweeps/fortho_persona_gr.py: forget_ortho_s = 0.1 (completion positions) with the anchor's monitor
penalty off, 3 seeds."""
from sweeps.fortho_persona_gr import make

runs = make((0.1,), penalty=False)
per_gpu = 6
no_baseline = True
