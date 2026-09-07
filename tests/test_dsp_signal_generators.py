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


def test_chirp_linear_matches_scipy():
  tc = np.linspace(0.0, 8.0, 2049)
  assert np.allclose(dsp.chirp(tc, 2.0, 8.0, 10.0), ss.chirp(tc, 2.0, 8.0, 10.0), atol=1e-4)


def test_chirp_unsupported_method_raises():
  with pytest.raises(ValueError):
    dsp.chirp(T, 1.0, 4.0, 2.0, method="quadratic")