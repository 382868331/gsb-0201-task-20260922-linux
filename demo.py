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
)
from ellipse_filter import filter_sample_grad

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
    print("=== Part 4: analytic derivatives of the value w.r.t. (u, v) ===")
    u, v = 12.37, 17.91
    grad = filter_sample_grad(texture, u, v, jacobian, wrap=WrapMode.REPEAT)
    print(f"at (u, v) = ({u}, {v}): value = {grad.value:.6f}")
    print(f"  analytic  d(value)/du = {grad.grad_u:+.6f}  "
          f"d(value)/dv = {grad.grad_v:+.6f}")
    h = 1e-5
    fd_u = (filter_sample(texture, u + h, v, jacobian).value
            - filter_sample(texture, u - h, v, jacobian).value) / (2 * h)
    fd_v = (filter_sample(texture, u, v + h, jacobian).value
            - filter_sample(texture, u, v - h, jacobian).value) / (2 * h)
    print(f"  central   d(value)/du = {fd_u:+.6f}  "
          f"d(value)/dv = {fd_v:+.6f}  (h = {h})")
    print(f"  differentiable = {grad.differentiable}")

    constant = [[1.5] * SIZE for _ in range(SIZE)]
    cgrad = filter_sample_grad(constant, u, v, jacobian)
    print(f"constant texture: grad = ({cgrad.grad_u:+.2e}, "
          f"{cgrad.grad_v:+.2e})  (exactly zero up to roundoff)")

    # A point exactly on the footprint boundary: zero Jacobian gives
    # C^-1 = 4I, so texel centers at distance 1 sit at q = 4.
    zero_j = ((0.0, 0.0), (0.0, 0.0))
    edge = filter_sample_grad(texture, 0.5, 0.5, zero_j)
    print(f"footprint-boundary point: differentiable = "
          f"{edge.differentiable}")
    print(f"  reason: {edge.nondifferentiable_reason}")


if __name__ == "__main__":
    main()
