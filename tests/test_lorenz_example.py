"""The Lorenz example's on-engine RK4 against a numpy reference. Requires the ANE."""
import sys
from pathlib import Path

import numpy as np

from _helpers import requires_ane

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import lorenz  # noqa: E402

pytestmark = requires_ane  # the RK4 step dispatches to the engine

HORIZON = 100            # chaos diverges later by design, so check the short horizon


def test_lorenz_rk4_matches_numpy_short_horizon():
  s0 = lorenz.initial_states()
  prog = lorenz.rk4_program()
  try:
    s = s0.astype(np.float16)
    ref = s0.astype(np.float32).copy()
    for _ in range(HORIZON):
      s = np.asarray(prog(s)).reshape(lorenz.B, 3).astype(np.float16)
      ref = lorenz.rk4_numpy(ref)
      assert np.abs(s.astype(np.float32) - ref).max() < 0.5, "fp16 RK4 left the numpy trajectory"
  finally:
    prog.release()
