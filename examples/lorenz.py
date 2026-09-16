"""aneforge showpiece: the Lorenz attractor integrated on the ANE - one RK4 step (four field evaluations + the weighted combine) is ONE fused program re-dispatched over a batch of trajectories. Writes docs/assets/lorenz.png if Pillow is present. Run: python3 examples/lorenz.py"""
import sys
import time
from pathlib import Path

import _common   # noqa: F401  (sets env + repo-root path; import before aneforge)
import numpy as np
import aneforge as af
from aneforge.graph import concat

# brand palette for the console
TEAL, RUST, DIM, BOLD, GREY, R = (
    "\033[38;2;72;187;170m", "\033[38;2;235;130;70m", "\033[2m",
    "\033[1m", "\033[38;2;150;150;150m", "\033[0m")
CHECK = f"{TEAL}OK{R}"

# simulation parameters
B = 8                   # trajectories integrated at once (a [B,3] state, one program)
SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
DT = 0.01               # RK4 step
STEPS = 5000            # rollout length (t = STEPS*DT = 50)
ANE_RAIL_W = 1.48       # measured sustained ANE rail, W (see demos/power_efficiency.py)

# x-z projection view (the butterfly)
XMIN, XMAX = -25.0, 25.0
ZMIN, ZMAX = 0.0, 50.0
IMG_W, IMG_H = 720, 540


def out(s=""):
    sys.stdout.write(s + "\n"); sys.stdout.flush()


def rk4_program(b=B):
    """Build ONE e5rt program: state [b,3] -> state' for a single RK4 step.

    The vector field is a handful of elementwise ops; splitting the [b,3] state into
    x/y/z columns and combining the four stages needs no host round-trip.
    """
    s = af.input((b, 3))

    def field(v):
        x, y, z = af.split(v, 3, axis=1)
        return concat([(y - x) * SIGMA,                            # dx = sigma*(y - x)
                       x * ((z * -1.0).adds(RHO)) - y,             # dy = x*(rho - z) - y
                       x * y - z * BETA], axis=1)                  # dz = x*y - beta*z

    k1 = field(s)
    k2 = field(s + k1 * (DT / 2.0))
    k3 = field(s + k2 * (DT / 2.0))
    k4 = field(s + k3 * DT)
    return af.compile(s + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (DT / 6.0))


def field_numpy(s):
    """fp32 numpy Lorenz vector field, for the reference RK4."""
    x, y, z = s[..., 0], s[..., 1], s[..., 2]
    return np.stack([SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z], -1).astype(np.float32)


def rk4_numpy(s):
    """Reference RK4 step (fp32)."""
    k1 = field_numpy(s)
    k2 = field_numpy(s + k1 * (DT / 2.0))
    k3 = field_numpy(s + k2 * (DT / 2.0))
    k4 = field_numpy(s + k3 * DT)
    return (s + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (DT / 6.0)).astype(np.float32)


def initial_states():
    """B nearby points near (1, 1, 1); all converge to the same attractor."""
    rng = np.random.default_rng(11)
    base = np.array([1.0, 1.0, 1.0], np.float32)
    return (base + rng.standard_normal((B, 3)).astype(np.float32) * 0.5).astype(np.float16)


def roll(prog, s0, steps):
    """Roll `steps` RK4 steps on the engine, returning the [steps+1, B, 3] trajectory."""
    s = s0.astype(np.float16)
    traj = np.empty((steps + 1, B, 3), np.float32)
    traj[0] = s.astype(np.float32)
    t = 0.0
    for i in range(steps):
        t0 = time.perf_counter()
        s = np.asarray(prog(s)).reshape(B, 3).astype(np.float16)
        t += time.perf_counter() - t0
        traj[i + 1] = s.astype(np.float32)
    return traj, t


def main():
    out()
    out(f"  {BOLD}{TEAL}ANEForge{R}  {DIM} - the Lorenz attractor on the Apple Neural Engine{R}")
    out(f"  {DIM}RK4 (four field evaluations) fused into one program, {B} trajectories per dispatch{R}")
    out()

    prog = rk4_program()
    out(f"  {GREY}compile{R} {CHECK} one RK4-step program {DIM}({prog.n_ops} ANE ops, compiled once){R}")

    s0 = initial_states()
    traj, ane_t = roll(prog, s0, STEPS)
    out(f"  {GREY}integrate{R} {CHECK} {STEPS} RK4 steps x {B} trajectories "
        f"{DIM}(t = {STEPS * DT:.0f}, {ane_t * 1e3 / STEPS:.2f} ms/step){R}")

    # short-horizon check against a numpy RK4 reference (chaos diverges later on purpose)
    horizon = 100
    ref = s0.astype(np.float32).copy()
    err = 0.0
    for i in range(horizon):
        ref = rk4_numpy(ref)
        err = max(err, float(np.abs(traj[i + 1] - ref).max() / (np.abs(ref).max() + 1e-6)))
    out(f"  {GREY}check{R}    {DIM}first {horizon} steps vs fp32 numpy RK4: max relerr {err:.2e}{R}")

    span = float(np.ptp(traj[:, :, 0])) + 1e-6
    energy = ANE_RAIL_W * ane_t
    out(f"  {GREY}extent{R}   {DIM}x spans {span:.1f} (the two lobes fired), "
        f"~{ane_t:.2f}s of ANE step time ~ {BOLD}{energy:.2f} J{R}{DIM} at the {ANE_RAIL_W} W rail{R}")
    out()

    wrote = render(traj)
    if wrote:
        out(f"  {CHECK} {BOLD}wrote {wrote}{R}")
    prog.release()

    ok = (err < 5e-2) and bool(np.isfinite(traj).all()) and span > 10.0
    out(f"  {('PASS' if ok else 'FAIL')}: the ANE integrated {B} Lorenz trajectories "
        f"({STEPS} RK4 steps) and the butterfly resolved")
    return 0 if ok else 1


# rendering
def render(traj):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        out(f"  {DIM}(install Pillow to write the figure: pip install pillow){R}")
        return None
    assets = Path(__file__).resolve().parents[1] / "docs" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (IMG_W, IMG_H), (2, 2, 8))
    d = ImageDraw.Draw(img)
    cols = ["#48bbaa", "#eb8246", "#7ad7f0", "#f2b06b", "#b48ead", "#5fd38d", "#e06c9f", "#9bb1ff"]

    def px(x, z):
        return (int((x - XMIN) / (XMAX - XMIN) * IMG_W),
                int((ZMAX - z) / (ZMAX - ZMIN) * IMG_H))

    for b in range(traj.shape[1]):
        pts = [px(float(x), float(z)) for x, z in zip(traj[::4, b, 0], traj[::4, b, 2])]
        d.line(pts, fill=cols[b % len(cols)], width=1)
    img.save(assets / "lorenz.png")
    return "docs/assets/lorenz.png"


if __name__ == "__main__":
    sys.exit(main())
