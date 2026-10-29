"""Box td of sweeps/fortho_topic_gr.py: make((0.001,), seeds=(4, 5))."""
from sweeps.fortho_topic_gr import make

runs = make((0.001,), seeds=(4, 5))
per_gpu = 6
no_baseline = True
