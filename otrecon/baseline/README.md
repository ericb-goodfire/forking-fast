# Baseline reference tree

Pristine vendored copies of the pre-optimization reference implementation,
used only by the equivalence unit tests (`tests/test_fast_equivalence.py`,
`tests/test_ablation.py` via `tests/_baseline_loader.py`).

* `otrecon/` + `src/run_scale.py` — the original (pure-ruptures) scale
  driver that produced the reference fan records.
* `src/run_ablation.py` — the original ablation driver.

The optimized package in the parent directory must reproduce this tree's
outputs exactly; the tests import both side by side and compare. Do not
edit (comments here were only scrubbed of internal references; the code is
verbatim).
