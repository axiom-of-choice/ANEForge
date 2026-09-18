"""aneforge showpiece: the Mandelbrot set by escape-time iteration on the ANE - the complex orbit (zr, zi) is carried as two real tensors and each iteration is one on-engine forward pass over the whole grid. Writes docs/assets/mandelbrot.png if Pillow is present. Run: python3 examples/mandelbrot.py"""
import sys
import time
from pathlib import Path

import _common   # noqa: F401  (sets env + repo-root path; import before aneforge)
import numpy as np
import aneforge as af
from aneforge import _compile as _c
from aneforge.graph import _const

# brand palette for the console
TEAL, RUST, DIM, BOLD, GREY, R = (
    "\033[38;2;72;187;170m", "\033[38;2;235;130;70m", "\033[2m",
    "\033[1m", "\033[38;2;150;150;150m", "\033[0m")
CHECK = f"{TEAL}OK{R}"

# view and iteration parameters
N = 640                 # grid (square)
MAX_ITER = 120          # escape-time budget
XMIN, XMAX = -2.5, 1.0  # real axis
YMIN, YMAX = -1.75, 1.75  # imaginary axis (same span, so the set is not stretched)
ANE_RAIL_W = 1.48       # measured sustained ANE rail, W (see demos/power_efficiency.py)

# 'inferno'-style perceptual colormap (matplotlib), as a small anchor table.
_CMAP = [(0.00, (2, 2, 8)), (0.13, (28, 12, 69)), (0.25, (74, 12, 107)),
         (0.38, (120, 28, 109)), (0.50, (165, 44, 96)), (0.63, (207, 68, 70)),
         (0.75, (237, 105, 37)), (0.87, (251, 155, 6)), (0.95, (247, 209, 61)),
         (1.00, (252, 255, 200))]


def out(s=""):
    sys.stdout.write(s + "\n"); sys.stdout.flush()


def grid(n=N):
    """The complex plane c = cr + i*ci over the view, as two [1,1,n,n] fp32 tensors."""
    x = np.linspace(XMIN, XMAX, n, dtype=np.float32)
    y = np.linspace(YMIN, YMAX, n, dtype=np.float32)
    cr, ci = np.meshgrid(x, y)                                # both [n, n]
    return cr.reshape(1, 1, n, n), ci.reshape(1, 1, n, n)


def iterate_program(shape=(1, 1, N, N)):
    """Build ONE e5rt program: (zr, zi, cr, ci, count) -> (zr', zi', count') for one orbit step.

    The orbit is carried as two real tensors (the ANE is real-valued, as in aneforge/fft.py).
    Once |z|^2 leaves the escape radius the point is frozen (`select` keeps z and count),
    so escaped points keep the z they had at escape - that is what the smooth count needs.
    `shape` matches the grid that is fed (default [1, 1, N, N]).
    """
    zr = af.input(shape); zi = af.input(shape)
    cr = af.input(shape); ci = af.input(shape)
    count = af.input(shape)
    zr2 = zr * zr - zi * zi + cr
    zi2 = zr * zi * 2.0 + ci
    mag2 = zr2 * zr2 + zi2 * zi2
    inside = mag2.less_equal(_const(4.0))                      # still bounded after this step
    return _c.compile_multi([af.select(inside, zr2, zr),       # freeze the escaped
                             af.select(inside, zi2, zi),
                             af.select(inside, count + 1.0, count)])


def numpy_escape(cr, ci, max_iter=MAX_ITER):
    """Reference escape-time integration with the same freeze rule, for the spot check."""
    zr = np.zeros_like(cr); zi = np.zeros_like(ci); count = np.zeros_like(cr)
    for _ in range(max_iter):
        zr2 = zr * zr - zi * zi + cr
        zi2 = zr * zi * 2.0 + ci
        inside = zr2 * zr2 + zi2 * zi2 <= 4.0
        zr = np.where(inside, zr2, zr)
        zi = np.where(inside, zi2, zi)
        count = np.where(inside, count + 1.0, count)
    return count, zr, zi


def smooth(count, zr, zi, cr, ci):
    """Continuous escape time for the escaped points, MAX_ITER for the interior.

    The program froze each escaped point at its last bounded z, so its escapee is z*z + c;
    the standard n - log2(log|z_n|) then has a finite log everywhere it is used.
    """
    ezr = zr * zr - zi * zi + cr
    ezi = zr * zi * 2.0 + ci
    mag2 = np.maximum(ezr * ezr + ezi * ezi, 4.0001)       # first escape: |z| > 2
    mu = (count + 1.0) - np.log2(np.log(np.sqrt(mag2)))
    return np.where(count < MAX_ITER, mu, float(MAX_ITER))


def main():
    out()
    out(f"  {BOLD}{TEAL}ANEForge{R}  {DIM} - the Mandelbrot set on the Apple Neural Engine{R}")
    out(f"  {DIM}z -> z*z + c as two real tensors, one on-engine forward pass per iteration{R}")
    out()

    prog = iterate_program()
    onames = [n for _, n in prog.output_ports]
    out(f"  {GREY}compile{R} {CHECK} one orbit-step program "
        f"{DIM}(elementwise, compiled once, re-dispatched per iteration){R}")

    cr, ci = grid()
    zr = np.zeros((1, 1, N, N), np.float32); zi = np.zeros_like(zr); count = np.zeros_like(zr)
    ane_t = 0.0
    t0 = time.perf_counter()
    for _ in range(MAX_ITER):
        t = time.perf_counter()
        res = prog(zr, zi, cr, ci, count)
        ane_t += time.perf_counter() - t
        zr = res[onames[0]].reshape(1, 1, N, N)
        zi = res[onames[1]].reshape(1, 1, N, N)
        count = res[onames[2]].reshape(1, 1, N, N)
    wall = time.perf_counter() - t0

    c = count[0, 0]
    in_set = float((c >= MAX_ITER).mean())
    mu = smooth(c, zr[0, 0], zi[0, 0], cr[0, 0], ci[0, 0])
    energy = ANE_RAIL_W * ane_t

    out(f"  {GREY}iterate{R} {CHECK} {MAX_ITER} orbit steps on a {N}x{N} grid "
        f"{DIM}({wall:.1f}s wall, {ane_t * 1e3 / MAX_ITER:.2f} ms/step){R}")
    out(f"  {GREY}set{R}     {DIM}{in_set * 100:.1f}% of the plane stayed bounded "
        f"(the interior is black){R}")
    out(f"  {GREY}energy{R}  {DIM}~{ane_t:.1f}s of ANE step time at the measured "
        f"~{ANE_RAIL_W} W rail ~ {BOLD}{energy:.1f} J{R}{DIM} for the whole render{R}")
    out()

    wrote = render(mu)
    if wrote:
        out(f"  {CHECK} {BOLD}wrote {wrote}{R}")
    prog.release()

    ok = 0.0 < in_set < 1.0 and bool(np.isfinite(mu).all())
    out(f"  {('PASS' if ok else 'FAIL')}: the ANE ran {MAX_ITER} escape-time iterations on the "
        f"engine and the set resolved")
    return 0 if ok else 1


# rendering
def colormap(v):
    vs = np.array([p[0] for p in _CMAP]); cs = np.array([p[1] for p in _CMAP], float)
    v = np.clip(v, 0.0, 1.0) ** 0.5
    return np.stack([np.interp(v, vs, cs[:, j]) for j in range(3)], -1).astype(np.uint8)


def render(mu):
    try:
        from PIL import Image
    except ImportError:
        out(f"  {DIM}(install Pillow to write the image: pip install pillow){R}")
        return None
    assets = Path(__file__).resolve().parents[1] / "docs" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    rgb = colormap(mu / MAX_ITER)
    rgb[mu >= MAX_ITER] = (2, 2, 8)                        # the interior, flat black
    Image.fromarray(rgb).save(assets / "mandelbrot.png")
    return "docs/assets/mandelbrot.png"


if __name__ == "__main__":
    sys.exit(main())
