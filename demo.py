"""Demo: elliptical footprint filtering vs point sampling.

Deterministic, offline, standard library only.  Runs in well under 8 s:
builds a procedural 64x64 texture, compares point sampling against the
elliptical footprint filter at a minified anisotropic sample point, checks
constant conservation, and triggers a real budget rejection.
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ewa_filter import (
    BudgetExceededError,
    InvalidInputError,
    filter_footprint,
)

N = 64


def make_texture(n=N):
    """Deterministic texture: smooth gradient + high-frequency checker."""
    grid = []
    for y in range(n):
        row = []
        for x in range(n):
            base = 0.5 + 0.25 * math.sin(2.0 * math.pi * x / 32.0) * math.cos(
                2.0 * math.pi * y / 32.0
            )
            checker = 0.25 if (x + y) % 2 == 0 else -0.25
            row.append(base + checker)
        grid.append(row)
    return grid


def point_sample(grid, u, v):
    """Nearest-texel point sample with repeat addressing."""
    n = len(grid)
    m = len(grid[0])
    return grid[int(math.floor(v)) % n][int(math.floor(u)) % m]


def rotated_jacobian(sx, sy, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return [[c * sx, -s * sy], [s * sx, c * sy]]


def main():
    t0 = time.perf_counter()
    grid = make_texture()

    print("=== Elliptical footprint filtering demo (64x64 scalar texture) ===\n")

    # 1. Normal result: anisotropic minification, footprint vs point sample.
    u, v = 31.7, 22.4
    jacobian = rotated_jacobian(6.0, 0.75, 30.0)  # 6 texels long, 0.75 wide, 30 deg
    res = filter_footprint(grid, u, v, jacobian, wrap="repeat")
    pt = point_sample(grid, u, v)
    print(f"sample point (u, v) = ({u}, {v}), anisotropic J = 6.0 x 0.75 texels @ 30 deg")
    print(f"  point sample        : {pt:.6f}   (aliased single texel)")
    print(f"  footprint filter    : {res.value:.6f}")
    print(f"  contributing texels : {res.count}")
    print(f"  raw weight sum      : {res.weight_sum:.6f}")
    assert res.count == len(res.contributions)

    # The checker component averages to ~0 under the footprint, so the
    # filtered value must sit closer to the smooth local mean than the
    # point sample does.
    smooth = 0.5 + 0.25 * math.sin(2.0 * math.pi * u / 32.0) * math.cos(
        2.0 * math.pi * v / 32.0
    )
    print(f"  smooth local mean   : {smooth:.6f}")
    print(f"  |point - mean| = {abs(pt - smooth):.6f}   "
          f"|footprint - mean| = {abs(res.value - smooth):.6f}")

    # 2. Constant conservation.
    const_grid = [[2.5] * 16 for _ in range(16)]
    cres = filter_footprint(const_grid, 7.3, 9.1, rotated_jacobian(3.0, 0.5, -15.0))
    print(f"\nconstant texture 2.5 -> filtered {cres.value:.12f} "
          f"(count={cres.count})")

    # 3. Real rejection: footprint far beyond the candidate budget.
    print("\n--- budget rejection (no silent point-sample fallback) ---")
    try:
        filter_footprint(grid, 32.0, 32.0, [[64.0, 0.0], [0.0, 64.0]])
        print("ERROR: budget rejection did not fire")
    except BudgetExceededError as exc:
        print(f"  BudgetExceededError: {exc}")

    # 4. Real rejection: invalid input is located, not silently corrected.
    print("\n--- invalid input rejection ---")
    try:
        filter_footprint(grid, float("nan"), 0.5, [[1.0, 0.0], [0.0, 1.0]])
        print("ERROR: NaN rejection did not fire")
    except InvalidInputError as exc:
        print(f"  InvalidInputError: {exc}")

    dt = time.perf_counter() - t0
    print(f"\ndemo completed in {dt:.3f}s")


if __name__ == "__main__":
    main()
