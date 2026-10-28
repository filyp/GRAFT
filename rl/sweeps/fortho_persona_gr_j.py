"""Box j of sweeps/fortho_persona_gr.py: forget_ortho_s = 0.003 (completion positions), 3 seeds."""
from sweeps.fortho_persona_gr import make

runs = make((0.003,))
per_gpu = 6
no_baseline = True
