"""Demo: elliptical footprint filtering vs. point sampling.

Deterministic, offline, standard library only.  Runs in well under 8 s.

Part 1 samples a high-contrast deterministic texture along a line with a
minifying, slanted Jacobian and compares nearest-texel point sampling
against the elliptical footprint filter.

Part 2 shows the budget guard: an enormous footprint is rejected with
BudgetExceededError instead of silently degrading to point sampling.

Part 3 shows an invalid input being rejected with a locatable error.

Part 4 exercises the tile atlas: explicit tile ids, differently sized
tiles, per-tile wrap modes at negative UV / corners, strict tile
isolation, and a real TileNotFoundError.
"""

import math

from ellipse_filter import (
    BudgetExceededError,
    InvalidInputError,
    TileAtlas,
    TileNotFoundError,
    WrapMode,
    filter_sample,
    filter_tile,
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
    print("=== Part 4: tile atlas (explicit ids, per-tile wrap, isolation) ===")
    # Tiles of different sizes.  Values are simple/distinctive so the
    # boundary behavior can be checked by hand with the zero-Jacobian
    # radius-1 cross (center weight 1, four arms weight e^-8 each).
    col_tile = [[float(i) for i in range(4)] for _ in range(3)]      # 4x3
    wide_tile = [[100.0 + float(i) for i in range(7)] for _ in range(2)]  # 7x2
    zero_tile = [[0.0 for _ in range(4)] for _ in range(4)]          # 4x4
    hundred_tile = [[100.0 for _ in range(4)] for _ in range(4)]     # 4x4
    atlas = TileAtlas(
        {
            "cols": (col_tile, WrapMode.REPEAT),   # tile carries its own wrap
            "wide": (wide_tile, WrapMode.REPEAT),
            "zero": zero_tile,
            "hundred": hundred_tile,
        },
        default_wrap=WrapMode.CLAMP,  # tiles without an own wrap clamp
    )
    print(f"atlas tiles: {list(atlas.tile_ids)}  "
          f"sizes cols={atlas.size('cols')} wide={atlas.size('wide')}")

    zero_jac = ((0.0, 0.0), (0.0, 0.0))
    edge = math.exp(-8.0)

    # Boundary sample: u=0.5 sits on the left edge of 'cols' (4 wide).
    u, v = 0.5, 1.5
    r_repeat = filter_tile(atlas, "cols", u, v, zero_jac)  # tile default
    r_clamp = filter_tile(atlas, "cols", u, v, zero_jac,
                          wrap=WrapMode.CLAMP)             # explicit override
    print(f"\nboundary sample on 'cols' (value=column) at u={u}, v={v}:")
    print(f"  repeat (tile default): value={r_repeat.value:.6f} "
          f"texels={sorted(set(r_repeat.texels))}")
    print(f"  clamp (per call):      value={r_clamp.value:.6f} "
          f"texels={sorted(set(r_clamp.texels))}")
    hand_repeat = (3.0 * edge + 1.0 * edge) / (1.0 + 4.0 * edge)
    print(f"  hand-checked repeat value = (3*e^-8 + 1*e^-8)/"
          f"(1+4*e^-8) = {hand_repeat:.6f}")
    print("  -> negative column -1 wraps to column 3 under repeat,")
    print("     but pins to column 0 under clamp; no other tile is touched.")

    # Same UV on a differently sized tile wraps to a different far column.
    r_wide = filter_tile(atlas, "wide", u, 0.5, zero_jac)
    print(f"\nsame u={u} on the 7-wide 'wide' tile: "
          f"column -1 wraps to 6, far texel present: "
          f"{(6, 0) in r_wide.texels}")

    # Strict isolation across tiles, footprint straddling an edge.
    big_jac = ((1.4, 0.0), (0.0, 1.4))
    z = filter_tile(atlas, "zero", -2.5, 6.5, big_jac)
    h = filter_tile(atlas, "hundred", -2.5, 6.5, big_jac)
    print(f"\ntile isolation at out-of-range uv=(-2.5, 6.5):")
    print(f"  'zero'    -> {z.value:.1f} (tile_id={z.tile_id!r})")
    print(f"  'hundred' -> {h.value:.1f} (tile_id={h.tile_id!r})")
    print("  -> each footprint resolves only against its own tile.")

    # Real failure: unknown tile id.
    try:
        filter_tile(atlas, "missing", 0.5, 0.5, zero_jac)
    except TileNotFoundError as err:
        print(f"\nTileNotFoundError: {err}")
        print(f"  requested id={err.tile_id!r}, available={list(err.available)}")


if __name__ == "__main__":
    main()
