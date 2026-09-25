"""Demo: elliptical footprint filtering vs. point sampling.

Deterministic, offline, standard library only.  Runs in well under 8 s.

Part 1 samples a high-contrast deterministic texture along a line with a
minifying, slanted Jacobian and compares nearest-texel point sampling
against the elliptical footprint filter.

Part 2 shows the budget guard: an enormous footprint is rejected with
BudgetExceededError instead of silently degrading to point sampling.
"""

import math

from ellipse_filter import (
    BudgetExceededError,
    InvalidInputError,
    WrapMode,
    filter_sample,
    filter_sample_with_derivatives,
)

SIZE = 32


def make_texture(size=SIZE):
    """Deterministic texture: high-frequency checker plus a smooth ramp."""
    texture = []
    for j in range(size):
        row = []
        for i in range(size):
            checker = 1.0 if (i + j) % 2 == 0 else 0.0
            ramp = 0.5 + 0.5 * math.sin(2.0 * math.pi * i / size) * math.cos(
                2.0 * math.pi * j / size
            )
            row.append(0.7 * checker + 0.3 * ramp)
        texture.append(row)
    return texture


def point_sample(texture, u, v):
    """Nearest-texel point sample with repeat wrapping."""
    size = len(texture)
    i = math.floor(u) % size
    j = math.floor(v) % size
    return texture[j][i]


def main():
    texture = make_texture()
    mean = sum(sum(row) for row in texture) / (SIZE * SIZE)
    # Minifying, slanted screen-to-texture Jacobian: one screen pixel
    # covers a long, rotated ellipse in texture space.
    jacobian = ((3.0, 1.5), (-1.0, 2.5))

    print("=== Part 1: point sampling vs. elliptical footprint filter ===")
    print(f"texture: {SIZE}x{SIZE} checker+ramp, global mean = {mean:.6f}")
    print(f"Jacobian J = {jacobian}  (minifying, slanted)")
    print()
    header = f"{'t':>4} {'u':>7} {'v':>7} {'point':>9} {'footprint':>10} {'contrib':>7}"
    print(header)
    print("-" * len(header))
    for k in range(9):
        t = k / 8.0
        u = 4.0 + 23.0 * t
        v = 9.0 + 11.0 * t
        point = point_sample(texture, u, v)
        result = filter_sample(texture, u, v, jacobian, wrap=WrapMode.REPEAT)
        print(f"{t:>4.2f} {u:>7.2f} {v:>7.2f} {point:>9.4f} "
              f"{result.value:>10.4f} {result.count:>7d}")
    print()
    print("The point sample flickers between 0 and 1 on the checkerboard,")
    print("while the footprint filter integrates the ellipse and stays")
    print("close to the local average (near the global mean).")

    print()
    print("=== Part 2: budget guard rejects an oversized footprint ===")
    huge = ((64.0, 0.0), (0.0, 64.0))
    try:
        filter_sample(texture, 16.0, 16.0, huge)
    except BudgetExceededError as err:
        print(f"BudgetExceededError: {err}")
        print(f"  candidates={err.candidates}  budget={err.budget}")
        print("  -> no silent fallback to point sampling, no partial result")

    print()
    print("=== Part 3: invalid input is rejected with a locatable error ===")
    try:
        filter_sample(texture, math.nan, 16.0, jacobian)
    except InvalidInputError as err:
        print(f"InvalidInputError: {err}")

    print()
    print("=== Part 4: analytic d(value)/du, d(value)/dv vs. central diff ===")
    u, v = 12.7, 17.3
    grad = filter_sample_with_derivatives(texture, u, v, jacobian)
    h = 1e-6
    fd_du = (filter_sample(texture, u + h, v, jacobian).value
             - filter_sample(texture, u - h, v, jacobian).value) / (2 * h)
    fd_dv = (filter_sample(texture, u, v + h, jacobian).value
             - filter_sample(texture, u, v - h, jacobian).value) / (2 * h)
    print(f"at (u, v) = ({u}, {v}), fixed Jacobian:")
    print(f"  analytic: du={grad.dvalue_du:+.8f}  dv={grad.dvalue_dv:+.8f}")
    print(f"  central diff (h={h}): du={fd_du:+.8f}  dv={fd_dv:+.8f}")
    print(f"  nondifferentiable={grad.nondifferentiable} (safe position)")

    print()
    print("=== Part 5: nondifferentiable locus is flagged, not hidden ===")
    # Zero Jacobian: C^-1 = 4 I, so texel center (1.5, 0.5) is exactly at
    # q = 4 when sampling (0.5, 0.5): a candidate appears/disappears and
    # its clamp mapping switches at the same time.
    edge = filter_sample_with_derivatives(
        texture, 0.5, 0.5, ((0.0, 0.0), (0.0, 0.0)), wrap=WrapMode.CLAMP)
    print(f"nondifferentiable={edge.nondifferentiable}")
    for reason in edge.reasons:
        print(f"  - {reason}")
    print("  (derivatives at such positions are one-sided/advisory)")


if __name__ == "__main__":
    main()
