"""Demo: ripmap construction and anisotropic sampling.

Deterministic, offline, standard library only.  Computes everything live
in well under 90 seconds (actual runtime is a fraction of a second).

Part 1 builds a ripmap over an odd-sized texture and prints the recorded
level sizes.  Part 2 samples along a line with an elongated footprint and
compares point sampling, the ripmap value and an exact box area integral
of the original texture computed on the spot.  Part 3 prints the recorded
level selection for one sample.  Part 4 shows a degenerate Jacobian and
an invalid input being reported.
"""

import math

from ellipse_filter import InvalidInputError, WrapMode
from ripmap import build_ripmap, ripmap_sample

WIDTH, HEIGHT = 33, 17  # odd on both axes: exercises coverage weighting


def make_texture():
    """High frequency along u (checker columns), smooth ramp along v."""
    texture = []
    for j in range(HEIGHT):
        row = []
        for i in range(WIDTH):
            stripes = 1.0 if i % 2 == 0 else 0.0
            ramp = j / (HEIGHT - 1)
            row.append(0.6 * stripes + 0.4 * ramp)
        texture.append(row)
    return texture


def point_sample(texture, u, v):
    i = math.floor(u) % WIDTH
    j = math.floor(v) % HEIGHT
    return texture[j][i]


def reference_box_average(texture, u, v, su, sv):
    """Exact area integral of the texture over the footprint box."""
    x0, x1 = u - su, u + su
    y0, y1 = v - sv, v + sv
    total = 0.0
    accum = 0.0
    for j in range(math.floor(y0), math.ceil(y1)):
        oy = min(y1, j + 1.0) - max(y0, float(j))
        if oy <= 0.0:
            continue
        for i in range(math.floor(x0), math.ceil(x1)):
            ox = min(x1, i + 1.0) - max(x0, float(i))
            if ox <= 0.0:
                continue
            area = ox * oy
            total += area
            accum += area * texture[j % HEIGHT][i % WIDTH]
    return accum / total


def main():
    texture = make_texture()
    ripmap = build_ripmap(texture)

    print("=== Part 1: ripmap level sizes (33x17 base, odd axes) ===")
    for lv, row in enumerate(ripmap.all_sizes()):
        print(f"  lv={lv}: " + "  ".join(f"{w}x{h}" for w, h in row))
    print("Odd edges are averaged with coverage weights 2/1, never dropped.")

    print()
    print("=== Part 2: point vs ripmap vs exact box integral ===")
    su, sv = 8.0, 1.0
    jacobian = ((su, 0.0), (0.0, sv))
    print(f"Jacobian J = {jacobian}  (footprint ~{2*su:.0f} x {2*sv:.0f} texels)")
    header = (
        f"{'t':>5} {'u':>6} {'v':>6} {'point':>7} "
        f"{'ripmap':>8} {'box-ref':>8} {'err':>7}"
    )
    print(header)
    print("-" * len(header))
    worst = 0.0
    for k in range(8):
        t = k / 7.0
        u = 2.0 + 27.0 * t
        v = 3.0 + 11.0 * t
        point = point_sample(texture, u, v)
        res = ripmap_sample(ripmap, u, v, jacobian)
        ref = reference_box_average(texture, u, v, su, sv)
        err = abs(res.value - ref)
        worst = max(worst, err)
        print(f"{t:>5.2f} {u:>6.2f} {v:>6.2f} {point:>7.3f} "
              f"{res.value:>8.4f} {ref:>8.4f} {err:>7.4f}")
    print(f"worst |ripmap - exact box integral| = {worst:.4f}")
    print("The ripmap is an axis-aligned box approximation of the footprint,")
    print("not an exact integral of an arbitrarily rotated ellipse.")

    print()
    print("=== Part 3: recorded level selection for one sample ===")
    res = ripmap_sample(ripmap, 15.0, 8.0, jacobian, wrap=WrapMode.REPEAT)
    print(f"continuous levels: lambda_u={res.level_u:.3f} "
          f"lambda_v={res.level_v:.3f}")
    for c in res.corners:
        print(f"  level ({c.level_u},{c.level_v}) size={c.size} "
              f"corner_weight={c.corner_weight:.3f} value={c.value:.4f} "
              f"texels={c.texels}")

    print()
    print("=== Part 4: degenerate and invalid footprints ===")
    degenerate = ripmap_sample(ripmap, 10.0, 8.0, ((0.0, 0.0), (0.0, 0.0)))
    print(f"zero Jacobian -> levels clamp to "
          f"({degenerate.level_u}, {degenerate.level_v}), "
          f"value={degenerate.value:.4f}")
    try:
        ripmap_sample(ripmap, math.nan, 8.0, jacobian)
    except InvalidInputError as err:
        print(f"InvalidInputError: {err}")


if __name__ == "__main__":
    main()
