"""Box tc of sweeps/fortho_topic_gr.py: make((0.001,), seeds=(1, 2, 3))."""
from sweeps.fortho_topic_gr import make

runs = make((0.001,), seeds=(1, 2, 3))
per_gpu = 6
no_baseline = True
