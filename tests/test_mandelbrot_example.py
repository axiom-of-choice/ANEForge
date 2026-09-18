"""The Mandelbrot example's on-engine escape-time rule against a numpy reference. Requires the ANE."""
import sys
from pathlib import Path

import numpy as np

from _helpers import requires_ane

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import mandelbrot as mb  # noqa: E402

pytestmark = requires_ane  # the orbit program dispatches to the engine

MAX_ITER = 60

# Points chosen well away from the set boundary. Right at the boundary the orbit is chaotic
# and a 1-ULP fp16 rounding can move the escape iteration by one; these do not.
IN_SET = [(0.0, 0.0), (-1.0, 0.0), (-0.5, 0.0), (-1.2, 0.0)]
ESCAPES = [(2.0, 0.0), (-2.5, 0.0), (1.0, 1.0), (0.0, 2.0)]


def _run(points, iters=MAX_ITER):
  k = len(points)
  prog = mb.iterate_program((1, 1, 1, k))
  onames = [name for _, name in prog.output_ports]
  cr = np.array([p[0] for p in points], np.float32).reshape(1, 1, 1, k)
  ci = np.array([p[1] for p in points], np.float32).reshape(1, 1, 1, k)
  zr = np.zeros_like(cr); zi = np.zeros_like(ci); count = np.zeros_like(cr)
  try:
    for _ in range(iters):
      res = prog(zr, zi, cr, ci, count)
      zr = res[onames[0]].reshape(1, 1, 1, k)
      zi = res[onames[1]].reshape(1, 1, 1, k)
      count = res[onames[2]].reshape(1, 1, 1, k)
  finally:
    prog.release()
  return count


def test_mandelbrot_matches_numpy_on_chosen_points():
  points = IN_SET + ESCAPES
  k = len(points)
  got = _run(points)[0, 0, 0]
  ref, _, _ = mb.numpy_escape(
    np.array([p[0] for p in points], np.float32).reshape(1, 1, 1, k),
    np.array([p[1] for p in points], np.float32).reshape(1, 1, 1, k), MAX_ITER)
  assert np.array_equal(got, ref[0, 0, 0])
  assert np.all(got[:len(IN_SET)] == MAX_ITER)          # the interior never escapes
  assert np.all(got[len(IN_SET):] < 5)                  # these leave the disk early
