"""Signal generators (aneforge.dsp sawtooth/square/chirp) against scipy.signal. Host-side, no ANE."""
import numpy as np
import pytest

import aneforge.dsp as dsp

ss = pytest.importorskip("scipy.signal")

T = np.linspace(0.0, 4.0 * np.pi, 1025)


@pytest.mark.parametrize("width", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_sawtooth_matches_scipy(width):
  assert np.allclose(dsp.sawtooth(T, width), ss.sawtooth(T, width), atol=1e-6)


def test_sawtooth_invalid_width_is_nan():
  assert np.isnan(dsp.sawtooth(np.array([0.0]), 1.5)[0])
  assert np.isnan(dsp.sawtooth(np.array([0.0]), -0.5)[0])


@pytest.mark.parametrize("duty", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_square_matches_scipy(duty):
  assert np.allclose(dsp.square(T, duty), ss.square(T, duty), atol=1e-6)


def test_square_invalid_duty_is_nan():
  assert np.isnan(dsp.square(np.array([1.0]), 2.0)[0])
  assert np.isnan(dsp.square(np.array([1.0]), -1.0)[0])


@pytest.mark.parametrize("f0,f1", [(2.0, 10.0), (10.0, 2.0)])   # up-sweep and down-sweep
def test_chirp_linear_matches_scipy(f0, f1):
  # measured max abs err is 3e-8 (float32 rounding of the returned array), flat even for a
  # 5->500 Hz sweep over 100 s; 1e-6 keeps 30x headroom while still catching a float32
  # phase accumulation, which the phase reaching ~300 rad here would surface.
  tc = np.linspace(0.0, 8.0, 2049)
  assert np.allclose(dsp.chirp(tc, f0, 8.0, f1), ss.chirp(tc, f0, 8.0, f1), atol=1e-6)


@pytest.mark.parametrize("method", ["linear", "lin", "li"])
def test_chirp_accepts_scipy_method_aliases(method):
  tc = np.linspace(0.0, 8.0, 2049)
  assert np.allclose(dsp.chirp(tc, 2.0, 8.0, 10.0, method=method),
                     ss.chirp(tc, 2.0, 8.0, 10.0, method=method), atol=1e-6)


def test_array_valued_width_and_duty():
  """scipy broadcasts width/duty against t; so does this, but nothing covered it."""
  v = np.linspace(0.0, 1.0, T.size)
  assert np.allclose(dsp.sawtooth(T, v), ss.sawtooth(T, v), atol=1e-6)
  assert np.allclose(dsp.square(T, v), ss.square(T, v), atol=1e-6)


def test_chirp_unsupported_method_raises():
  with pytest.raises(ValueError):
    dsp.chirp(T, 1.0, 4.0, 2.0, method="quadratic")
