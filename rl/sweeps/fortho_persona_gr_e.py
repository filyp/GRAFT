"""Box e of sweeps/fortho_persona_gr.py: forget_ortho_s = 0.1 on all anchor positions (prompt included), 3 seeds."""
from sweeps.fortho_persona_gr import make

runs = make((0.1,), tokens="all")
per_gpu = 6
no_baseline = True
