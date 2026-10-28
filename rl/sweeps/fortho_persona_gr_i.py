"""Box i of sweeps/fortho_persona_gr.py: forget_ortho_s = 0.01 (completion positions), 3 seeds."""
from sweeps.fortho_persona_gr import make

runs = make((0.01,))
per_gpu = 6
no_baseline = True
