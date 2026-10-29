"""Patch K2 Horizon's chat template for Terminal Wrench SFT and eval (2026-10-29):
  * assistant turns without a thinking field (TW trajectories; terminus-2's replayed history) are rendered with an
    empty think_faster block instead of raising "Assistant message is missing a thinking field";
  * reasoning_effort defaults to 'low' (generation prompt opens think_faster). K2 has no non-thinking mode; low is the
    closest to the paper's non-thinking Qwen3 setting, and it matches how the training turns are rendered.

  python -m scripts.eval.patch_k2_template <chat_template.jinja> <out.jinja>
"""
import re
import sys


def patch(s: str) -> str:
    s, n = re.subn(r'\{\{- raise_exception\("Assistant message is missing a thinking field[^"]*"\) -\}\}',
                   "{%- set think_tag = 'ifm|think_faster' -%}", s)
    assert n == 1, n
    s, n = re.subn(r"reasoning_effort \| default\('high'\)", "reasoning_effort | default('low')", s)
    assert n == 1, n
    return s


if __name__ == "__main__":
    open(sys.argv[2], "w").write(patch(open(sys.argv[1]).read()))
