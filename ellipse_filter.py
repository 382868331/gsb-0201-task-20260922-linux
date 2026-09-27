"""Elliptic footprint filtering for anisotropic texture sampling.

Reusable, dependency-free module (Python standard library only) implementing
an EWA-style elliptical Gaussian footprint filter for scalar and RGBA
textures.

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

RGBA filtering always uses one shared spatial weight for color and alpha.
Premultiplied texels are filtered directly.  Unpremultiplied texels are
converted to premultiplied form first and converted back after filtering;
the two representations are never mixed in one accumulation.  Fully
transparent texels therefore have zero premultiplied color and cannot
introduce a colored fringe, while opaque black remains
``(0, 0, 0, 1)`` rather than becoming transparent black.

Wrap modes ``repeat`` and ``clamp`` are supported.  When several candidate
texel centers map onto the same stored texel (tiling repeats, or clamping
to an edge), each copy keeps its own kernel weight and contributes
independently.

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
    """How RGBA texels store color and alpha."""

    PREMULTIPLIED = "premultiplied"
    UNPREMULTIPLIED = "unpremultiplied"


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
        value: normalized scalar value or RGBA tuple.  Scalar results use
            ``sum(w_k * t_k) / sum(w_k)``; RGBA results are filtered using
            one shared spatial kernel for all four channels.
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


def _validate_texture(
    texture: object, alpha_mode: AlphaMode | None
) -> tuple[tuple[tuple[float, ...], ...], int, int, int]:
    if isinstance(texture, (str, bytes)) or not isinstance(texture, Sequence):
        raise InvalidInputError(
            f"texture: expected a sequence of rows, got {type(texture).__name__}"
        )
    if len(texture) == 0:
        raise InvalidInputError("texture: must contain at least one row")
    rows: list[tuple[float, ...]] = []
    width: int | None = None
    channels: int | None = None
    for r, row in enumerate(texture):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise InvalidInputError(
                f"texture[{r}]: expected a sequence of numbers, "
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
        converted = []
        for c, value in enumerate(row):
            name = f"texture[{r}][{c}]"
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                scalar = _as_finite_float(name, value)
                row_channels = 1
                converted.append(scalar)
            else:
                if len(value) != 4:
                    raise InvalidInputError(
                        f"{name}: color texels must contain exactly 4 values "
                        f"(r, g, b, a), got {len(value)}"
                    )
                if alpha_mode is None:
                    raise InvalidInputError(
                        "alpha_mode: is required for RGBA textures; use "
                        "AlphaMode.PREMULTIPLIED or AlphaMode.UNPREMULTIPLIED"
                    )
                row_channels = 4
                texel = tuple(
                    _as_finite_float(f"{name}[{channel}]", component)
                    for channel, component in enumerate(value)
                )
                red, green, blue, alpha = texel
                if not 0.0 <= alpha <= 1.0:
                    raise InvalidInputError(
                        f"{name}[3]: alpha must be in [0, 1], got {alpha}"
                    )
                if alpha_mode is AlphaMode.PREMULTIPLIED:
                    for channel_name, component in (("r", red), ("g", green), ("b", blue)):
                        if not 0.0 <= component <= alpha:
                            raise InvalidInputError(
                                f"{name}: premultiplied {channel_name} must be "
                                f"in [0, alpha], got {component} with alpha {alpha}"
                            )
                else:
                    for channel_name, component in (("r", red), ("g", green), ("b", blue)):
                        if not 0.0 <= component <= 1.0:
                            raise InvalidInputError(
                                f"{name}: unpremultiplied {channel_name} must be "
                                f"in [0, 1], got {component}"
                            )
                    if alpha == 0.0:
                        texel = (0.0, 0.0, 0.0, 0.0)
                    else:
                        texel = (red * alpha, green * alpha, blue * alpha, alpha)
                converted.extend(texel)

            if channels is None:
                channels = row_channels
            elif row_channels != channels:
                raise InvalidInputError(
                    f"texture[{r}][{c}]: cannot mix scalar and RGBA texels"
                )
        rows.append(tuple(converted))
    height = len(rows)
    assert width is not None
    if width > MAX_TEXTURE_SIZE or height > MAX_TEXTURE_SIZE:
        raise InvalidInputError(
            f"texture: size {width}x{height} exceeds the "
            f"{MAX_TEXTURE_SIZE}x{MAX_TEXTURE_SIZE} limit"
        )
    if channels == 4 and alpha_mode is None:
        raise InvalidInputError(
            "alpha_mode: is required for RGBA textures; use "
            "AlphaMode.PREMULTIPLIED or AlphaMode.UNPREMULTIPLIED"
        )
    if channels == 1 and alpha_mode is not None:
        raise InvalidInputError(
            "alpha_mode: must be None for scalar textures, got "
            f"{alpha_mode.value!r}"
        )
    return tuple(rows), width, height, channels


def _validate_alpha_mode(alpha_mode: object) -> AlphaMode | None:
    if alpha_mode is None:
        return None
    if not isinstance(alpha_mode, AlphaMode):
        raise InvalidInputError(
            "alpha_mode: expected AlphaMode.PREMULTIPLIED, "
            f"AlphaMode.UNPREMULTIPLIED, or None, got {alpha_mode!r}"
        )
    return alpha_mode


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
    texture: Sequence[Sequence[float]],
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
        texture: rectangular sequence of rows of scalar texel values,
            or RGBA tuples, at most 64x64.  ``texture[j][i]`` is the texel
            whose center is at ``(i + 0.5, j + 0.5)`` in texture space.
        u, v: sampling coordinate in texture space (finite).
        jacobian: 2x2 screen-to-texture Jacobian ``J``; the footprint uses
            ``C = J * J^T + 0.25 * I``, so any finite ``J`` (including
            singular or zero) yields a positive definite ``C``.
        wrap: :class:`WrapMode` for out-of-range texel coordinates.
        budget: maximum number of candidate texel centers inside the
            footprint bounding box (default 4096).
        alpha_mode: representation of RGBA texels.  Scalar textures leave
            this as ``None``.  Unpremultiplied RGBA is converted to
            premultiplied RGBA before filtering; premultiplied RGBA is
            filtered directly.  Results use the requested representation.

    Returns:
        A :class:`FilterResult` with the filtered value, the normalized
        per-contribution weights and the actual contribution count.

    Raises:
        InvalidInputError: any argument violates the input contract.
        BudgetExceededError: the footprint bounding box holds more than
            ``budget`` candidate texels.  No partial result is produced
            and no point-sampling fallback is performed.
    """
    alpha_mode = _validate_alpha_mode(alpha_mode)
    tex, width, height, channels = _validate_texture(texture, alpha_mode)
    u = _as_finite_float("u", u)
    v = _as_finite_float("v", v)
    j00, j01, j10, j11 = _validate_jacobian(jacobian)
    wrap = _validate_wrap(wrap)
    budget = _validate_budget(budget)

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
    weighted_channels = [0.0] * channels
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
                total_weight += w
                texel = tex[tj]
                offset = ti * channels
                for channel in range(channels):
                    weighted_channels[channel] += w * texel[offset + channel]
                weights.append(w)
                texels.append((ti, tj))

    if not weights:
        # Unreachable: the texel center nearest to (u, v) is at squared
        # distance <= 0.5, and C^-1's largest eigenvalue is <= 1/0.25 = 4,
        # so its q is <= 2 < FOOTPRINT_RADIUS_SQUARED.
        raise EllipseFilterError("internal error: footprint contains no texel")

    filtered_channels = [accum / total_weight for accum in weighted_channels]
    if channels == 1:
        value: float | tuple[float, float, float, float] = filtered_channels[0]
    else:
        red, green, blue, alpha = filtered_channels
        if alpha_mode is AlphaMode.UNPREMULTIPLIED and alpha > 0.0:
            value = (red / alpha, green / alpha, blue / alpha, alpha)
        elif alpha <= 0.0:
            value = (0.0, 0.0, 0.0, alpha)
        else:
            value = (red, green, blue, alpha)
    normalized = tuple(w / total_weight for w in weights)
    return FilterResult(
        value=value,
        weights=normalized,
        texels=tuple(texels),
        count=len(weights),
        weight_sum=total_weight,
    )
