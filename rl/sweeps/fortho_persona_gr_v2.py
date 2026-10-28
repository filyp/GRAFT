"""Box v2 of sweeps/fortho_persona_gr.py: S = 0.001, anchor penalty off, seed 2 only — rerun with the non-finite
guard in forget_ortho_edit (the original seed-2 run on box v went NaN at step 791 inside the gate edit)."""
from sweeps.fortho_persona_gr import make

runs = make((0.001,), penalty=False, seeds=(2,))
per_gpu = 6
no_baseline = True
