"""Ripmap construction and sampling for anisotropic texture minification.

Complements ``ellipse_filter.py``: instead of collapsing an elongated
footprint into a single scalar LOD (classic mipmapping), a ripmap keeps a
separate downsampled image for every (u-level, v-level) pair, so the two
axes may shrink at different rates.

Model
-----
Given the same 2x2 screen-to-texture Jacobian ``J`` used by
:func:`ellipse_filter.filter_sample`, form ``C = J * J^T + 0.25 * I``.  The
footprint semi-extents along the texture axes are

    radius_u = 2 * sqrt(C_00)      radius_v = 2 * sqrt(C_11)

(texels).  The per-axis level-of-detail is

    lod_u = log2(radius_u)  clamped to [0, max_level_u]
    lod_v = log2(radius_v)  clamped to [0, max_level_v]

so a footprint one texel wide along an axis selects level 0 on that axis.
The two real-valued LODs are interpolated bilinearly between the four
adjacent ripmap levels ``(a0/a1) x (b0/b1)``.

Levels
------
Level ``(a, b)`` has size ``ceil(W / 2^a) x ceil(H / 2^b)``; every level
size is recorded.  Downsampling halves one axis at a time by area
averaging: an output texel is the coverage-weighted mean of the 1 or 2
input texels it spans, so an odd input edge contributes its last
row/column with weight 1 instead of being dropped.

In-level sampling
-----------------
Within a level the texture coordinate is rescaled by the recorded level
size (``u_l = u * level_w / W``) and the nearest texel is taken, using the
same boundary addressing convention as the EWA entry point
(:class:`WrapMode.REPEAT` / :class:`WrapMode.CLAMP`).

Error semantics
---------------
The same input contract as ``ellipse_filter`` applies (re-using its
validators): NaN/Infinity, bools as numbers, malformed textures/Jacobians
and invalid ``wrap`` raise :class:`InvalidInputError`.  Degenerate
footprints (zero/singular ``J``) are *not* errors: the ``0.25 * I``
regularisation yields radius 1 on both axes, i.e. LOD (0, 0), reported
explicitly in the result.

Accuracy
--------
The result is a bilinear blend of four nearest-texel reads from
area-averaged levels.  It approximates the axis-aligned area average of
the footprint rectangle; it is **not** an exact integral of arbitrary
rotated ellipses (rotation is ignored, only the axis-aligned extents of
``C`` are used).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

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
    "RipmapLevel",
    "LevelSample",
    "RipmapResult",
    "build_ripmap",
    "sample_ripmap",
]


@dataclass(frozen=True)
class RipmapLevel:
    """One downsampled level of a ripmap.

    Attributes:
        lod_u: halving count applied along the u (column) axis.
        lod_v: halving count applied along the v (row) axis.
        width: level width in texels, ``ceil(W / 2**lod_u)``.
        height: level height in texels, ``ceil(H / 2**lod_v)``.
        data: ``height`` rows of ``width`` averaged texel values.
    """

    lod_u: int
    lod_v: int
    width: int
    height: int
    data: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class LevelSample:
    """Record of one in-level nearest-texel read.

    Attributes:
        level: ``(lod_u, lod_v)`` of the sampled level.
        size: ``(width, height)`` of that level.
        texel: ``(i, j)`` texel actually read after wrap/clamp.
        weight: bilinear blend weight of this level in the final value.
        value: texel value read.
    """

    level: tuple[int, int]
    size: tuple[int, int]
    texel: tuple[int, int]
    weight: float
    value: float


@dataclass(frozen=True)
class RipmapResult:
    """Result of one ripmap sample.

    Attributes:
        value: bilinear blend of the four level samples.
        lod_u, lod_v: clamped real-valued per-axis LODs actually used.
        radius_u, radius_v: footprint semi-extents in texels per axis.
        degenerate: True when the Jacobian was singular/zero and the
            footprint collapsed to the regularised 1-texel radius.
        samples: the four :class:`LevelSample` records (level selection
            and in-level sampling, in (a0,b0), (a0,b1), (a1,b0), (a1,b1)
            order).
    """

    value: float
    lod_u: float
    lod_v: float
    radius_u: float
    radius_v: float
    degenerate: bool
    samples: tuple[LevelSample, ...]


class Ripmap:
    """A full 2D pyramid of per-axis downsampled levels."""

    def __init__(self, texture: Sequence[Sequence[float]]) -> None:
        tex, width, height = _validate_texture(texture)
        self.width = width
        self.height = height
        self.max_lod_u = (width - 1).bit_length()
        self.max_lod_v = (height - 1).bit_length()
        # levels[a][b]: a halves u, b halves v.
        levels: list[list[RipmapLevel]] = []
        column: list[RipmapLevel] = []
        prev = RipmapLevel(0, 0, width, height, tex)
        column.append(prev)
        for b in range(1, self.max_lod_v + 1):
            prev = _halve_rows(prev, b)
            column.append(prev)
        levels.append(column)
        for a in range(1, self.max_lod_u + 1):
            column = []
            for b in range(self.max_lod_v + 1):
                column.append(_halve_columns(levels[a - 1][b], a))
            levels.append(column)
        self._levels = levels

    def level(self, lod_u: int, lod_v: int) -> RipmapLevel:
        """Return the recorded level ``(lod_u, lod_v)``."""
        return self._levels[lod_u][lod_v]

    def level_sizes(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        """``((w, h), ...)`` of every level, indexed ``[lod_u][lod_v]``."""
        return tuple(
            tuple((lvl.width, lvl.height) for lvl in column)
            for column in self._levels
        )


def _halve_rows(level: RipmapLevel, lod_v: int) -> RipmapLevel:
    """Halve the v axis of ``level`` with coverage-weighted averaging."""
    src, sw, sh = level.data, level.width, level.height
    dh = (sh + 1) // 2
    rows = []
    for dj in range(dh):
        j0 = 2 * dj
        if j0 + 1 < sh:
            rows.append(
                tuple(
                    (src[j0][i] + src[j0 + 1][i]) / 2.0 for i in range(sw)
                )
            )
        else:
            # Odd edge: the last output row covers a single input row.
            rows.append(tuple(src[j0][i] for i in range(sw)))
    return RipmapLevel(level.lod_u, lod_v, sw, dh, tuple(rows))


def _halve_columns(level: RipmapLevel, lod_u: int) -> RipmapLevel:
    """Halve the u axis of ``level`` with coverage-weighted averaging."""
    src, sw, sh = level.data, level.width, level.height
    dw = (sw + 1) // 2
    rows = []
    for j in range(sh):
        row = []
        for di in range(dw):
            i0 = 2 * di
            if i0 + 1 < sw:
                row.append((src[j][i0] + src[j][i0 + 1]) / 2.0)
            else:
                # Odd edge: keep the last input column with weight 1.
                row.append(src[j][i0])
        rows.append(tuple(row))
    return RipmapLevel(lod_u, level.lod_v, dw, sh, tuple(rows))


def build_ripmap(texture: Sequence[Sequence[float]]) -> Ripmap:
    """Build a ripmap from a rectangular scalar texture (at most 64x64).

    Raises:
        InvalidInputError: the texture violates the shared input contract
            (empty, non-rectangular, oversized, non-finite values, ...).
    """
    return Ripmap(texture)


def _sample_level(
    level: RipmapLevel,
    u: float,
    v: float,
    src_w: int,
    src_h: int,
    wrap: WrapMode,
) -> tuple[int, int, float]:
    """Nearest-texel read of ``level`` at texture-space ``(u, v)``."""
    i = math.floor(u * level.width / src_w)
    j = math.floor(v * level.height / src_h)
    if wrap is WrapMode.REPEAT:
        i %= level.width
        j %= level.height
    else:
        i = 0 if i < 0 else (level.width - 1 if i >= level.width else i)
        j = 0 if j < 0 else (level.height - 1 if j >= level.height else j)
    return i, j, level.data[j][i]


def sample_ripmap(
    ripmap: Ripmap,
    u: float,
    v: float,
    jacobian: Sequence[Sequence[float]],
    *,
    wrap: WrapMode = WrapMode.REPEAT,
) -> RipmapResult:
    """Sample ``ripmap`` at ``(u, v)`` with the anisotropic LODs of ``jacobian``.

    The footprint's per-axis semi-extents select two independent LODs; the
    value is interpolated bilinearly between the four adjacent ripmap
    levels.  Level selection and in-level sampling are recorded in the
    returned :class:`RipmapResult`.

    Raises:
        InvalidInputError: any argument violates the input contract.
            Degenerate (zero/singular) Jacobians are valid and yield
            LOD (0, 0) with ``degenerate=True`` in the result.
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
    if not (math.isfinite(c00) and math.isfinite(c11)):
        raise InvalidInputError(
            "jacobian: C = J*J^T + 0.25*I overflowed to non-finite values"
        )
    radius_u = 2.0 * math.sqrt(c00)
    radius_v = 2.0 * math.sqrt(c11)
    degenerate = radius_u <= 1.0 and radius_v <= 1.0

    lod_u = min(max(math.log2(radius_u), 0.0), float(ripmap.max_lod_u))
    lod_v = min(max(math.log2(radius_v), 0.0), float(ripmap.max_lod_v))

    a0 = min(int(math.floor(lod_u)), max(ripmap.max_lod_u - 1, 0))
    b0 = min(int(math.floor(lod_v)), max(ripmap.max_lod_v - 1, 0))
    a1 = min(a0 + 1, ripmap.max_lod_u)
    b1 = min(b0 + 1, ripmap.max_lod_v)
    fu = lod_u - a0
    fv = lod_v - b0

    plan = (
        (a0, b0, (1.0 - fu) * (1.0 - fv)),
        (a0, b1, (1.0 - fu) * fv),
        (a1, b0, fu * (1.0 - fv)),
        (a1, b1, fu * fv),
    )
    samples: list[LevelSample] = []
    value = 0.0
    for a, b, weight in plan:
        level = ripmap.level(a, b)
        i, j, texel_value = _sample_level(
            level, u, v, ripmap.width, ripmap.height, wrap
        )
        samples.append(
            LevelSample(
                level=(a, b),
                size=(level.width, level.height),
                texel=(i, j),
                weight=weight,
                value=texel_value,
            )
        )
        value += weight * texel_value

    return RipmapResult(
        value=value,
        lod_u=lod_u,
        lod_v=lod_v,
        radius_u=radius_u,
        radius_v=radius_v,
        degenerate=degenerate,
        samples=tuple(samples),
    )
