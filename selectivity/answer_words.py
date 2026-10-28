"""How many candidate answers (the distinct answers of the hackable eval prompts) each completion on a hackable
prompt mentions, from gens.py outputs. qa_correct passes if the answer appears anywhere, so naming several
candidates can pass by listing.

  python selectivity/answer_words.py selectivity/results/gens/<run>.json [...]
"""
import json
import re
import statistics
import sys

data = {p: json.load(open(p)) for p in sys.argv[1:]}
vocab = {r["answer"].lower() for d in data.values() for rows in d["modes"].values()
         for r in rows if r["hackable"] and r["answer"]}
print(len(vocab), "candidate answers")
for p, d in data.items():
    for mode, rows in d["modes"].items():
        hk = [r for r in rows if r["hackable"]]
        n = [len({w for w in vocab if re.search(rf"\b{re.escape(w)}\b", r["completion"].lower())}) for r in hk]
        cor = [k for k, r in zip(n, hk) if r["correct"]]
        print(f"{p.split('_st1000_')[-1]:24s} {mode:11s} answers/completion {statistics.mean(n):.2f}; among correct "
              f"{statistics.mean(cor) if cor else float('nan'):.2f} (n={len(cor)}); >=2: {sum(k >= 2 for k in n) / len(n):.2f}")
