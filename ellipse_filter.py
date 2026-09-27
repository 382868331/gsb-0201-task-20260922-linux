"""Elliptic footprint filtering for anisotropic texture sampling.

Reusable, dependency-free module (Python standard library only) implementing an
EWA-style elliptical Gaussian footprint filter for scalar and RGBA textures.

Model
-----
Given a sampling coordinate ``(u, v)`` in texture space (texel centers at
integer + 0.5) and a 2x2 screen-to-texture Jacobian ``J``::

    C = J * J^T + 0.25 * I          (always symmetric positive definite)

The footprint is the ellipse ``d^T * C^-1 * d <= 4`` around ``(u, v)``.
Every texel center inside the bounding box of that ellipse is tested;
centers inside the ellipse contribute a Gaussian kernel weight
``exp(-2 * d^T * C^-1 * d)``.  The filtered value is the weighted average
of the contributing texels (weights normalized to sum 1).

Wrap modes ``repeat`` and ``clamp`` are supported.  When several candidate
texel centers map onto the same stored texel (tiling repeats, or clamping
to an edge), each copy keeps its own kernel weight and contributes
independently.

RGBA textures use rows of four-channel texels.  The caller must state whether
those texels are premultiplied or straight (unpremultiplied).  Straight colors
are converted to premultiplied before filtering, so fully transparent texels
contribute black color instead of leaking their irrelevant stored RGB values.
All four channels are filtered with the same spatial kernel weights; alpha is
not used as an additional color weight.

Error semantics
---------------
* Non-finite (NaN/Infinity) or non-numeric numeric inputs, bools passed as
  numbers, malformed textures/Jacobians and invalid enum/budget arguments
  raise :class:`InvalidInputError` with the offending parameter named.
* If the number of candidate texels inside the footprint bounding box
  exceeds ``budget`` (default 4096), :class:`BudgetExceededError` is raised
  before any accumulation.  The function never silently falls back to
  point sampling and never returns a partial result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

__all__ = [
    "MAX_TEXTURE_SIZE",
    "DEFAULT_BUDGET",
    "FOOTPRINT_RADIUS_SQUARED",
    "KERNEL_SCALE",
    "WrapMode",
    "AlphaMode",
    "EllipseFilterError",
    "InvalidInputError",
    "BudgetExceededError",
    "FilterResult",
    "filter_sample",
]

#: Maximum texture width/height (textures are at most 64x64).
MAX_TEXTURE_SIZE = 64

#: Default upper bound on candidate texels inside the footprint bounding box.
DEFAULT_BUDGET = 4096

#: Footprint boundary: d^T * C^-1 * d <= FOOTPRINT_RADIUS_SQUARED.
FOOTPRINT_RADIUS_SQUARED = 4.0

#: Kernel weight is exp(-KERNEL_SCALE * d^T * C^-1 * d).
KERNEL_SCALE = 2.0


class WrapMode(Enum):
    """How out-of-range texel coordinates are mapped back into the texture."""

    REPEAT = "repeat"
    CLAMP = "clamp"


class AlphaMode(Enum):
    """Alpha representation used by RGBA texture input and output."""

    PREMULTIPLIED = "premultiplied"
    STRAIGHT = "straight"
    UNPREMULTIPLIED = "straight"


class EllipseFilterError(Exception):
    """Base class for all errors raised by this module."""


class InvalidInputError(EllipseFilterError, ValueError):
    """A supplied argument violates the input contract."""


class BudgetExceededError(EllipseFilterError, RuntimeError):
    """The footprint bounding box contains more candidates than allowed.

    Attributes:
        candidates: number of candidate texels in the bounding box
            (``math.inf`` when the footprint matrix itself overflowed).
        budget: the budget that was in effect.
    """

    def __init__(self, message: str, *, candidates: float, budget: int) -> None:
        super().__init__(message)
        self.candidates = candidates
        self.budget = budget


@dataclass(frozen=True)
class FilterResult:
    """Result of one footprint-filtered sample.

    Attributes:
        value: normalized filtered scalar, or an ``(r, g, b, a)`` tuple for
            RGBA textures.  RGBA output uses the same alpha representation
            requested by the caller.
        weights: per-contribution kernel weights, normalized to sum 1,
            in row-major scan order over candidate texel centers.
        texels: ``(i, j)`` texel indices (column, row) actually sampled,
            aligned with ``weights``; the same texel may appear multiple
            times when repeat/clamp maps several centers onto it.
        count: number of actual contributions (``len(weights)``).
        weight_sum: raw (unnormalized) kernel weight sum.
    """

    value: float | tuple[float, float, float, float]
    weights: tuple[float, ...]
    texels: tuple[tuple[int, int], ...]
    count: int
    weight_sum: float


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _as_finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInputError(
            f"{name}: expected a real number, got {type(value).__name__}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise InvalidInputError(f"{name}: must be finite, got {value!r}")
    return result


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_texture(
    texture: object,
) -> tuple[tuple[tuple[float, ...] | tuple[float, float, float, float], ...], int, int, int]:
    if isinstance(texture, (str, bytes)) or not isinstance(texture, Sequence):
        raise InvalidInputError(
            f"texture: expected a sequence of rows, got {type(texture).__name__}"
        )
    if len(texture) == 0:
        raise InvalidInputError("texture: must contain at least one row")
    rows: list[tuple[float, ...] | tuple[float, float, float, float]] = []
    width: int | None = None
    channels = 1
    for r, row in enumerate(texture):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise InvalidInputError(
                f"texture[{r}]: expected a sequence of numbers or RGBA texels, "
                f"got {type(row).__name__}"
            )
        if width is None:
            width = len(row)
            if width == 0:
                raise InvalidInputError("texture: rows must not be empty")
        elif len(row) != width:
            raise InvalidInputError(
                f"texture[{r}]: row has {len(row)} columns, expected {width} "
                "(texture must be rectangular)"
            )
        row_channels = 1 if len(row) > 0 and _is_number(row[0]) else 4
        if r == 0:
            channels = row_channels
        elif row_channels != channels:
            raise InvalidInputError(
                f"texture[{r}]: mixes scalar and RGBA texels; all rows must "
                "use the same texture format"
            )

        if channels == 1:
            converted = []
            for c, value in enumerate(row):
                converted.append(_as_finite_float(f"texture[{r}][{c}]", value))
            rows.append(tuple(converted))
        else:
            converted_row = []
            for c, texel in enumerate(row):
                if isinstance(texel, (str, bytes)) or not isinstance(texel, Sequence):
                    raise InvalidInputError(
                        f"texture[{r}][{c}]: expected an RGBA sequence, "
                        f"got {type(texel).__name__}"
                    )
                if len(texel) != 4:
                    raise InvalidInputError(
                        f"texture[{r}][{c}]: expected 4 RGBA channels, "
                        f"got {len(texel)}"
                    )
                converted = tuple(
                    _as_finite_float(
                        f"texture[{r}][{c}][{channel}]", value
                    )
                    for channel, value in enumerate(texel)
                )
                alpha = converted[3]
                if not 0.0 <= alpha <= 1.0:
                    raise InvalidInputError(
                        f"texture[{r}][{c}][3]: alpha must be in [0, 1], "
                        f"got {alpha}"
                    )
                converted_row.append(converted)
            rows.append(tuple(converted_row))
    height = len(rows)
    assert width is not None
    if width > MAX_TEXTURE_SIZE or height > MAX_TEXTURE_SIZE:
        raise InvalidInputError(
            f"texture: size {width}x{height} exceeds the "
            f"{MAX_TEXTURE_SIZE}x{MAX_TEXTURE_SIZE} limit"
        )
    return tuple(rows), width, height, channels


def _validate_alpha_mode(alpha_mode: object, channels: int) -> AlphaMode | None:
    if channels == 1:
        if alpha_mode is not None and not isinstance(alpha_mode, AlphaMode):
            raise InvalidInputError(
                "alpha_mode: expected an AlphaMode member or None for scalar "
                f"textures, got {alpha_mode!r}"
            )
        return None
    if not isinstance(alpha_mode, AlphaMode):
        raise InvalidInputError(
            "alpha_mode: RGBA textures require an explicit AlphaMode member, "
            f"got {alpha_mode!r}"
        )
    return alpha_mode


def _to_premultiplied(
    texel: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    red, green, blue, alpha = texel
    if alpha == 0.0:
        return 0.0, 0.0, 0.0, alpha
    return red * alpha, green * alpha, blue * alpha, alpha


def _validate_jacobian(jacobian: object) -> tuple[float, float, float, float]:
    if isinstance(jacobian, (str, bytes)) or not isinstance(jacobian, Sequence):
        raise InvalidInputError(
            f"jacobian: expected a 2x2 sequence, got {type(jacobian).__name__}"
        )
    if len(jacobian) != 2:
        raise InvalidInputError(
            f"jacobian: expected 2 rows, got {len(jacobian)}"
        )
    out = []
    for r in range(2):
        row = jacobian[r]
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise InvalidInputError(
                f"jacobian[{r}]: expected a sequence of 2 numbers, "
                f"got {type(row).__name__}"
            )
        if len(row) != 2:
            raise InvalidInputError(
                f"jacobian[{r}]: expected 2 columns, got {len(row)}"
            )
        for c in range(2):
            out.append(_as_finite_float(f"jacobian[{r}][{c}]", row[c]))
    return out[0], out[1], out[2], out[3]


def _validate_wrap(wrap: object) -> WrapMode:
    if not isinstance(wrap, WrapMode):
        raise InvalidInputError(
            f"wrap: expected a WrapMode member, got {wrap!r}"
        )
    return wrap


def _validate_budget(budget: object) -> int:
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise InvalidInputError(
            f"budget: expected an int, got {type(budget).__name__}"
        )
    if budget < 1:
        raise InvalidInputError(f"budget: must be >= 1, got {budget}")
    return budget


# ---------------------------------------------------------------------------
# Core filter
# ---------------------------------------------------------------------------


def filter_sample(
    texture: Sequence[Sequence[float | Sequence[float]]],
    u: float,
    v: float,
    jacobian: Sequence[Sequence[float]],
    *,
    wrap: WrapMode = WrapMode.REPEAT,
    budget: int = DEFAULT_BUDGET,
    alpha_mode: AlphaMode | None = None,
) -> FilterResult:
    """Filter ``texture`` at ``(u, v)`` with the elliptical footprint of ``jacobian``.

    Args:
        texture: rectangular scalar texture or RGBA texture, at most 64x64.
            ``texture[j][i]`` is the texel whose center is at
            ``(i + 0.5, j + 0.5)`` in texture space.  RGBA texels are
            four-element sequences.
        u, v: sampling coordinate in texture space (finite).
        jacobian: 2x2 screen-to-texture Jacobian ``J``; the footprint uses
            ``C = J * J^T + 0.25 * I``, so any finite ``J`` (including
            singular or zero) yields a positive definite ``C``.
        wrap: :class:`WrapMode` for out-of-range texel coordinates.
        budget: maximum number of candidate texel centers inside the
            footprint bounding box (default 4096).
        alpha_mode: required for RGBA textures and ignored for scalar
            textures.  Selects premultiplied or straight alpha for both input
            and output; straight input is converted to premultiplied only for
            filtering.

    Returns:
        A :class:`FilterResult` with the filtered value, the normalized
        per-contribution weights and the actual contribution count.

    Raises:
        InvalidInputError: any argument violates the input contract.
        BudgetExceededError: the footprint bounding box holds more than
            ``budget`` candidate texels.  No partial result is produced
            and no point-sampling fallback is performed.
    """
    tex, width, height, channels = _validate_texture(texture)
    u = _as_finite_float("u", u)
    v = _as_finite_float("v", v)
    j00, j01, j10, j11 = _validate_jacobian(jacobian)
    wrap = _validate_wrap(wrap)
    budget = _validate_budget(budget)
    alpha_mode = _validate_alpha_mode(alpha_mode, channels)

    # C = J * J^T + 0.25 * I (symmetric positive definite for finite J).
    c00 = j00 * j00 + j01 * j01 + 0.25
    c01 = j00 * j10 + j01 * j11
    c11 = j10 * j10 + j11 * j11 + 0.25
    det = c00 * c11 - c01 * c01
    if not (math.isfinite(c00) and math.isfinite(c11) and math.isfinite(det)):
        raise BudgetExceededError(
            "jacobian: C = J*J^T + 0.25*I overflowed to non-finite values; "
            "the footprint is far larger than any supported budget",
            candidates=math.inf,
            budget=budget,
        )

    # C^-1 for a symmetric 2x2.
    inv00 = c11 / det
    inv01 = -c01 / det
    inv11 = c00 / det

    # Tight axis-aligned bounding box of d^T * C^-1 * d <= 4:
    # half-extents are 2*sqrt(C_00) and 2*sqrt(C_11).
    half_u = 2.0 * math.sqrt(c00)
    half_v = 2.0 * math.sqrt(c11)

    # Texel centers are at integer + 0.5.
    lo_i = math.ceil(u - half_u - 0.5)
    hi_i = math.floor(u + half_u - 0.5)
    lo_j = math.ceil(v - half_v - 0.5)
    hi_j = math.floor(v + half_v - 0.5)
    count_i = hi_i - lo_i + 1
    count_j = hi_j - lo_j + 1
    candidates = count_i * count_j
    if candidates > budget:
        raise BudgetExceededError(
            f"footprint bounding box holds {candidates} candidate texels, "
            f"exceeding the budget of {budget}",
            candidates=candidates,
            budget=budget,
        )

    total_weight = 0.0
    weighted_value = 0.0 if channels == 1 else [0.0, 0.0, 0.0, 0.0]
    weights: list[float] = []
    texels: list[tuple[int, int]] = []
    for jy in range(lo_j, hi_j + 1):
        dy = (jy + 0.5) - v
        for ix in range(lo_i, hi_i + 1):
            dx = (ix + 0.5) - u
            q = inv00 * dx * dx + 2.0 * inv01 * dx * dy + inv11 * dy * dy
            if q <= FOOTPRINT_RADIUS_SQUARED:
                w = math.exp(-KERNEL_SCALE * q)
                if wrap is WrapMode.REPEAT:
                    ti = ix % width
                    tj = jy % height
                else:
                    ti = 0 if ix < 0 else (width - 1 if ix >= width else ix)
                    tj = 0 if jy < 0 else (height - 1 if jy >= height else jy)
                texel = tex[tj][ti]
                total_weight += w
                if channels == 1:
                    weighted_value += w * texel
                else:
                    if alpha_mode is AlphaMode.STRAIGHT:
                        texel = _to_premultiplied(texel)
                    elif texel[3] == 0.0:
                        texel = (0.0, 0.0, 0.0, 0.0)
                    for channel in range(4):
                        weighted_value[channel] += w * texel[channel]
                weights.append(w)
                texels.append((ti, tj))

    if not weights:
        # Unreachable: the texel center nearest to (u, v) is at squared
        # distance <= 0.5, and C^-1's largest eigenvalue is <= 1/0.25 = 4,
        # so its q is <= 2 < FOOTPRINT_RADIUS_SQUARED.
        raise EllipseFilterError("internal error: footprint contains no texel")

    if channels == 1:
        value = weighted_value / total_weight
    else:
        filtered = tuple(accum / total_weight for accum in weighted_value)
        if alpha_mode is AlphaMode.STRAIGHT:
            alpha = filtered[3]
            if alpha == 0.0:
                value = (0.0, 0.0, 0.0, 0.0)
            else:
                value = (
                    filtered[0] / alpha,
                    filtered[1] / alpha,
                    filtered[2] / alpha,
                    alpha,
                )
        else:
            value = filtered
    normalized = tuple(w / total_weight for w in weights)
    return FilterResult(
        value=value,
        weights=normalized,
        texels=tuple(texels),
        count=len(weights),
        weight_sum=total_weight,
    )
