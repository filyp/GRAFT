"""Box n: plain GRAFT (no edit) on persona_qa with the flattery reward scaled to 0
(configs/persona_qa_flattery_respmon_bvo_nohackrew.yaml), 3 seeds. Diagnostic: does the setup learn open questions
without the hack reward? Run names: '_fo0_nohackrew'."""
import re

from sweeps.fortho_persona_gr import make

runs = [{**r, "config": "configs/persona_qa_flattery_respmon_bvo_nohackrew.yaml",
         "run_name": re.sub(r"_fo0_s(\d+)$", r"_fo0_nohackrew_s\1", r["run_name"])} for r in make((0,))]
assert all("_nohackrew_s" in r["run_name"] for r in runs)
per_gpu = 6
no_baseline = True
