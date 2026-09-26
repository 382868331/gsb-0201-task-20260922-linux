"""Demo: ripmap construction and anisotropic two-axis LOD sampling.

Deterministic, offline, standard library only.  Computes everything live
in well under 90 seconds: no sleeps, no pre-recorded answers.

Part 1 builds a ripmap over an odd-sized texture and prints the recorded
level sizes (odd edges are coverage-weighted, not dropped).

Part 2 samples a horizontal-high-frequency / vertical-smooth texture
with an elongated footprint and compares the ripmap value against an
independent area integral over the original image (computed here, from
scratch, with exact per-texel coverage weights).

Part 3 shows a degenerate footprint (zero Jacobian) and an invalid
input, both producing explicit results instead of silent fallbacks.
"""

import math

from ellipse_filter import InvalidInputError, WrapMode
from ripmap import build_ripmap, sample_ripmap


def reference_box_average(texture, u, v, radius_u, radius_v):
    """Independent area integral of the footprint box on the source image."""
    height = len(texture)
    width = len(texture[0])
    u0, u1 = u - radius_u, u + radius_u
    v0, v1 = v - radius_v, v + radius_v
    total = 0.0
    accum = 0.0
    for jy in range(math.floor(v0), math.ceil(v1)):
        oy = max(0.0, min(v1, jy + 1.0) - max(v0, float(jy)))
        for ix in range(math.floor(u0), math.ceil(u1)):
            ox = max(0.0, min(u1, ix + 1.0) - max(u0, float(ix)))
            if ox > 0.0 and oy > 0.0:
                coverage = ox * oy
                accum += coverage * texture[jy % height][ix % width]
                total += coverage
    return accum / total


def main():
    print("=== Part 1: ripmap construction on an odd-sized 31x17 texture ===")
    odd = [[float((i * 3 + j * 5) % 7) for i in range(31)] for j in range(17)]
    rip_odd = build_ripmap(odd)
    print("recorded level sizes [lod_u][lod_v] (w, h):")
    for a, column in enumerate(rip_odd.level_sizes()):
        print(f"  lod_u={a}: {list(column)}")
    print("odd edges keep their last row/column with coverage weight 1;")
    print("each level size is exactly ceil(size / 2**lod) per axis.")

    print()
    print("=== Part 2: anisotropic sampling vs. independent area integral ===")
    size = 32
    texture = [
        [(i % 2) + 0.3 * (j / (size - 1)) for i in range(size)]
        for j in range(size)
    ]
    rip = build_ripmap(texture)
    print(f"texture: {size}x{size}, alternating columns + smooth v ramp")
    jacobian = ((8.0, 0.0), (0.0, 0.25))
    print(f"Jacobian J = {jacobian} (long horizontal footprint)")
    print()
    header = (
        f"{'u':>6} {'v':>6} {'lod_u':>6} {'lod_v':>6} "
        f"{'ripmap':>8} {'boxref':>8} {'|diff|':>8}"
    )
    print(header)
    print("-" * len(header))
    for k in range(6):
        u = 6.0 + 4.0 * k
        v = 5.0 + 5.0 * k
        result = sample_ripmap(rip, u, v, jacobian, wrap=WrapMode.REPEAT)
        reference = reference_box_average(
            texture, u, v, result.radius_u, result.radius_v
        )
        print(f"{u:>6.2f} {v:>6.2f} {result.lod_u:>6.2f} {result.lod_v:>6.2f} "
              f"{result.value:>8.4f} {reference:>8.4f} "
              f"{abs(result.value - reference):>8.4f}")
    print()
    print("The ripmap keeps the two axes at different resolutions instead of")
    print("collapsing the footprint to one scalar LOD.  It approximates the")
    print("axis-aligned area average; it is NOT an exact integral of")
    print("arbitrary rotated ellipses (see README, known limitations).")
    print()
    print("level selection and in-level sampling for the last point:")
    result = sample_ripmap(rip, 26.0, 30.0, jacobian, wrap=WrapMode.REPEAT)
    for s in result.samples:
        print(f"  level={s.level} size={s.size} texel={s.texel} "
              f"weight={s.weight:.4f} value={s.value:.4f}")

    print()
    print("=== Part 3: degenerate footprint and invalid input ===")
    degenerate = sample_ripmap(rip, 10.2, 7.6, ((0.0, 0.0), (0.0, 0.0)))
    print(f"zero Jacobian -> degenerate={degenerate.degenerate}, "
          f"lod=({degenerate.lod_u}, {degenerate.lod_v}), "
          f"value={degenerate.value:.4f} (base-level nearest texel)")
    try:
        sample_ripmap(rip, math.nan, 7.6, ((1.0, 0.0), (0.0, 1.0)))
    except InvalidInputError as err:
        print(f"InvalidInputError: {err}")


if __name__ == "__main__":
    main()
