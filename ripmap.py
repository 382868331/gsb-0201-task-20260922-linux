"""Ripmap construction and sampling for anisotropic texture minification.

Dependency-free (standard library only) companion to ellipse_filter.py.
Instead of collapsing an elongated footprint into a single scalar LOD, a
ripmap keeps independent resolutions along the u and v axes: level
(lu, lv) is the base texture downsampled by 2**lu along u and by
2**lv along v, each step an area-weighted average.

Construction
------------
Every halving step averages pairs of texels.  When an axis length is odd,
the last output texel covers only one input texel; the average is weighted
by the actual pixel coverage (2 or 1), so the odd column/row is never
silently dropped.  All level sizes are recorded and queryable.

Sampling
--------
From the 2x2 screen-to-texture Jacobian J the footprint matrix
C = J * J^T + 0.25 * I (same model as ellipse_filter) gives per-axis
scales su = sqrt(C_00), sv = sqrt(C_11) and continuous levels
lambda_u = log2(su), lambda_v = log2(sv), clamped to the built level
range.  The four adjacent integer levels are sampled with bilinear
interpolation inside each level (using the same repeat/clamp boundary
addressing as ellipse_filter) and blended with bilinear weights in level
space.  Level selection and per-level texel weights are recorded in the
result.

Approximation scope
-------------------
This is an axis-aligned box approximation: a rotated ellipse is
approximated by its axis-aligned extents, so the result is NOT an exact
integral of an arbitrarily rotated elliptical footprint.  Error grows
with the rotation angle and the eccentricity of the footprint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ellipse_filter import (
    InvalidInputError,
    WrapMode,
    _as_finite_float,
    _validate_jacobian,
    _validate_texture,
    _validate_wrap,
)

__all__ = [
    "Ripmap",
    "LevelSample",
    "RipmapResult",
    "build_ripmap",
    "ripmap_sample",
]


def _halve_u(image, width):
    """Halve along u with coverage-weighted averaging (odd widths kept)."""
    new_width = (width + 1) // 2
    out = []
    for row in image:
        new_row = []
        for x in range(new_width):
            i0 = 2 * x
            if i0 + 1 < width:
                new_row.append((row[i0] + row[i0 + 1]) / 2.0)
            else:
                new_row.append(row[i0])
        out.append(tuple(new_row))
    return out, new_width


def _halve_v(image, height):
    """Halve along v with coverage-weighted averaging (odd heights kept)."""
    new_height = (height + 1) // 2
    width = len(image[0])
    out = []
    for y in range(new_height):
        j0 = 2 * y
        if j0 + 1 < height:
            out.append(
                tuple(
                    (image[j0][i] + image[j0 + 1][i]) / 2.0
                    for i in range(width)
                )
            )
        else:
            out.append(image[j0])
    return out, new_height


class Ripmap:
    """A two-dimensional mip pyramid with independent u/v resolutions.

    levels_u / levels_v give the number of levels along each axis; level
    indices run 0 .. levels_u - 1 / 0 .. levels_v - 1 where level 0 is
    the base texture and the last level along an axis has size 1 on that
    axis.
    """

    __slots__ = ("width", "height", "levels_u", "levels_v", "_data", "_sizes")

    def __init__(self, width, height, data, sizes):
        self.width = width
        self.height = height
        self.levels_u = len(data[0])
        self.levels_v = len(data)
        self._data = data
        self._sizes = sizes

    def size(self, level_u, level_v):
        """Return (width, height) of level (level_u, level_v)."""
        return self._sizes[level_v][level_u]

    def image(self, level_u, level_v):
        """Return the texel rows of level (level_u, level_v)."""
        return self._data[level_v][level_u]

    def all_sizes(self):
        """All level sizes as sizes[lv][lu] = (width, height)."""
        return tuple(tuple(row) for row in self._sizes)


def build_ripmap(texture):
    """Build a Ripmap from a rectangular scalar texture.

    Raises:
        InvalidInputError: the texture violates the same contract as
            ellipse_filter.filter_sample (non-rectangular, empty,
            non-finite values, bools, or larger than 64x64).
    """
    tex, width, height = _validate_texture(texture)

    # Halve along u first, then build a v-chain for every u-level.
    u_chain = [tex]
    u_widths = [width]
    while u_widths[-1] > 1:
        nxt, new_width = _halve_u(u_chain[-1], u_widths[-1])
        u_chain.append(tuple(nxt))
        u_widths.append(new_width)

    data = []
    sizes = []
    v_images = u_chain
    v_heights = [height] * len(u_chain)
    while True:
        data.append(v_images)
        sizes.append(
            tuple((u_widths[k], v_heights[k]) for k in range(len(u_chain)))
        )
        if v_heights[0] == 1:
            break
        next_images = []
        next_heights = []
        for img, h in zip(v_images, v_heights):
            nxt, new_height = _halve_v(img, h)
            next_images.append(tuple(nxt))
            next_heights.append(new_height)
        v_images = tuple(next_images)
        v_heights = next_heights

    return Ripmap(width, height, data, sizes)


@dataclass(frozen=True)
class LevelSample:
    """Record of one integer level contribution to a ripmap sample.

    Attributes:
        level_u, level_v: integer level indices actually sampled.
        size: (width, height) of that level.
        corner_weight: bilinear level-blend weight of this corner.
        value: bilinear intra-level sample value (before blending).
        texels: (i, j) texel indices sampled inside the level.
        texel_weights: intra-level bilinear weights aligned with texels;
            they sum to 1.
    """

    level_u: int
    level_v: int
    size: tuple[int, int]
    corner_weight: float
    value: float
    texels: tuple[tuple[int, int], ...]
    texel_weights: tuple[float, ...]


@dataclass(frozen=True)
class RipmapResult:
    """Result of one ripmap sample.

    Attributes:
        value: final blended value.
        level_u, level_v: continuous (fractional) levels selected from
            the footprint, after clamping to the built level range.
        corners: the four LevelSample records (level_u0/1 x level_v0/1)
            with their blend weights and intra-level samples.
        count: total number of texel contributions across all corners.
    """

    value: float
    level_u: float
    level_v: float
    corners: tuple[LevelSample, ...]
    count: int


def _map_index(i, size, wrap):
    if wrap is WrapMode.REPEAT:
        return i % size
    return 0 if i < 0 else (size - 1 if i >= size else i)


def _sample_level(image, width, height, u, v, wrap):
    """Bilinear sample at (u, v) in level texture space."""
    x = u - 0.5
    y = v - 0.5
    i0 = math.floor(x)
    j0 = math.floor(y)
    fx = x - i0
    fy = y - j0
    value = 0.0
    texels = []
    weights = []
    for dj, wy in ((0, 1.0 - fy), (1, fy)):
        if wy == 0.0:
            continue
        tj = _map_index(j0 + dj, height, wrap)
        for di, wx in ((0, 1.0 - fx), (1, fx)):
            if wx == 0.0:
                continue
            ti = _map_index(i0 + di, width, wrap)
            w = wx * wy
            value += w * image[tj][ti]
            texels.append((ti, tj))
            weights.append(w)
    return value, tuple(texels), tuple(weights)


def ripmap_sample(ripmap, u, v, jacobian, *, wrap=WrapMode.REPEAT):
    """Sample ripmap at (u, v) with the footprint of jacobian.

    The footprint C = J * J^T + 0.25 * I selects a continuous level per
    axis; the four adjacent integer levels are bilinearly sampled (texel
    centers at integer + 0.5, wrap boundary addressing) and blended.
    Degenerate Jacobians (including all-zero) are legal and clamp to
    level 0.  An overflowing Jacobian selects the coarsest level along
    the affected axis.

    Raises:
        InvalidInputError: non-finite coordinates, malformed Jacobian,
            invalid wrap, or a non-Ripmap first argument.
    """
    if not isinstance(ripmap, Ripmap):
        raise InvalidInputError(
            f"ripmap: expected a Ripmap, got {type(ripmap).__name__}"
        )
    u = _as_finite_float("u", u)
    v = _as_finite_float("v", v)
    j00, j01, j10, j11 = _validate_jacobian(jacobian)
    wrap = _validate_wrap(wrap)

    c00 = j00 * j00 + j01 * j01 + 0.25
    c11 = j10 * j10 + j11 * j11 + 0.25

    def continuous_level(c, levels):
        if not math.isfinite(c):
            return float(levels - 1)
        scale = math.sqrt(c)
        lam = math.log2(scale) if scale > 0.0 else -math.inf
        return min(max(lam, 0.0), float(levels - 1))

    lam_u = continuous_level(c00, ripmap.levels_u)
    lam_v = continuous_level(c11, ripmap.levels_v)

    lu0 = min(int(math.floor(lam_u)), ripmap.levels_u - 1)
    lv0 = min(int(math.floor(lam_v)), ripmap.levels_v - 1)
    lu1 = min(lu0 + 1, ripmap.levels_u - 1)
    lv1 = min(lv0 + 1, ripmap.levels_v - 1)
    fu = lam_u - lu0
    fv = lam_v - lv0

    corners = []
    value = 0.0
    count = 0
    for lv, wy in ((lv0, 1.0 - fv), (lv1, fv)):
        if wy == 0.0:
            continue
        for lu, wx in ((lu0, 1.0 - fu), (lu1, fu)):
            if wx == 0.0:
                continue
            corner_weight = wx * wy
            width, height = ripmap.size(lu, lv)
            image = ripmap.image(lu, lv)
            level_value, texels, weights = _sample_level(
                image, width, height, u / (1 << lu), v / (1 << lv), wrap
            )
            value += corner_weight * level_value
            count += len(texels)
            corners.append(
                LevelSample(
                    level_u=lu,
                    level_v=lv,
                    size=(width, height),
                    corner_weight=corner_weight,
                    value=level_value,
                    texels=texels,
                    texel_weights=weights,
                )
            )

    return RipmapResult(
        value=value,
        level_u=lam_u,
        level_v=lam_v,
        corners=tuple(corners),
        count=count,
    )
