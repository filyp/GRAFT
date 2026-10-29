"""persona_qa_topic (configs/persona_qa_topic_flattery_respmon_bvo.yaml: hackable = color questions, open and T/F)
with the GRAFT recipe of sweeps/fortho_persona_gr.py: plain GRAFT (S = 0) and --forget_ortho_s S. Run names start
with 'persona_qa_topic_flattery_respmon_bvo'. Boxes: ta / tb plain GRAFT seeds 1-3 / 4-5; tc / td S = 0.001
seeds 1-3 / 4-5 (sweeps/fortho_topic_gr_<box>.py).
"""
from sweeps.fortho_persona_gr import make as _make

CONFIG = "configs/persona_qa_topic_flattery_respmon_bvo.yaml"


def make(s_values, seeds=(1, 2, 3)):
    out = [{**r, "config": CONFIG,
            "run_name": r["run_name"].replace("persona_qa_flattery_respmon_bvo", "persona_qa_topic_flattery_respmon_bvo", 1)}
           for r in _make(s_values, seeds=seeds)]
    assert all(r["run_name"].startswith("persona_qa_topic_") for r in out)
    return out


runs = make((0, 0.001), seeds=(1, 2, 3, 4, 5))
per_gpu = 6
no_baseline = True
