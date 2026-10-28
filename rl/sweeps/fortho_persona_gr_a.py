"""Box a of sweeps/fortho_persona_gr.py: forget_ortho_s = 1.0, 3 seeds."""
from sweeps.fortho_persona_gr import make

runs = make((1.0,))
per_gpu = 6
no_baseline = True
