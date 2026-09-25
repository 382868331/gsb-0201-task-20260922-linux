"""Elliptic footprint filtering for anisotropic texture sampling.

Reusable, dependency-free module (Python standard library only) implementing
an EWA-style elliptical Gaussian footprint filter for scalar textures.

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

Derivatives
-----------
:func:`filter_sample_with_derivatives` additionally returns the analytic
partial derivatives of the filtered value with respect to the sampling
coordinates ``u`` and ``v`` for a fixed Jacobian.  Both the kernel weights
and the normalization denominator are differentiated (quotient rule)::

    value = N / D,  N = sum(w_k * t_k),  D = sum(w_k)
    dw_k/du = -2 * w_k * dq_k/du
    dvalue/du = (dN/du * D - N * dD/du) / D**2     (same for v)

Every copy of a repeated/clamped texel contributes its own weight
derivative to the sums before normalization; the derivative is *not* a
finite difference of final pixel values.

The value is a smooth function of ``(u, v)`` except where the discrete
candidate set changes, i.e. where some texel center sits exactly on the
footprint boundary ``q = 4`` (a candidate appears/disappears with weight
e^-8 != 0).  When such a boundary texel lies outside the stored texture,
the repeat/clamp mapping of the candidate switches at the same time.
Positions within ``boundary_tol`` of those events are reported via the
``nondifferentiable`` flag and human-readable ``reasons`` instead of
silently returning a derivative that does not exist there.

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
    "DEFAULT_BOUNDARY_TOL",
    "WrapMode",
    "EllipseFilterError",
    "InvalidInputError",
    "BudgetExceededError",
    "FilterResult",
    "GradientResult",
    "filter_sample",
    "filter_sample_with_derivatives",
]

#: Maximum texture width/height (textures are at most 64x64).
MAX_TEXTURE_SIZE = 64

#: Default upper bound on candidate texels inside the footprint bounding box.
DEFAULT_BUDGET = 4096

#: Footprint boundary: d^T * C^-1 * d <= FOOTPRINT_RADIUS_SQUARED.
FOOTPRINT_RADIUS_SQUARED = 4.0

#: Kernel weight is exp(-KERNEL_SCALE * d^T * C^-1 * d).
KERNEL_SCALE = 2.0

#: Default tolerance for the nondifferentiability probe: texel centers with
#: ``|q - FOOTPRINT_RADIUS_SQUARED|`` below this sit "on" the boundary.
DEFAULT_BOUNDARY_TOL = 1e-9


class WrapMode(Enum):
    """How out-of-range texel coordinates are mapped back into the texture."""

    REPEAT = "repeat"
    CLAMP = "clamp"


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
        value: normalized filtered value, ``sum(w_k * t_k) / sum(w_k)``.
        weights: per-contribution kernel weights, normalized to sum 1,
            in row-major scan order over candidate texel centers.
        texels: ``(i, j)`` texel indices (column, row) actually sampled,
            aligned with ``weights``; the same texel may appear multiple
            times when repeat/clamp maps several centers onto it.
        count: number of actual contributions (``len(weights)``).
        weight_sum: raw (unnormalized) kernel weight sum.
    """

    value: float
    weights: tuple[float, ...]
    texels: tuple[tuple[int, int], ...]
    count: int
    weight_sum: float


@dataclass(frozen=True)
class GradientResult:
    """Result of one footprint-filtered sample with analytic derivatives.

    Attributes:
        sample: the plain FilterResult; numerically identical to what
            filter_sample returns for the same arguments.
        dvalue_du: analytic partial derivative of sample.value with
            respect to u (Jacobian held fixed).
        dvalue_dv: analytic partial derivative with respect to v.
        nondifferentiable: True when the sampling position lies within
            boundary_tol of a locus where the value is not differentiable
            (a texel center crossing the footprint boundary, possibly
            switching the repeat/clamp mapping at the same time).  The
            returned derivatives are then the one-sided analytic values
            of the current candidate configuration and should be treated
            as advisory.
        reasons: human-readable descriptions of each nearby
            nondifferentiable locus; empty when nondifferentiable is
            False.
    """

    sample: FilterResult
    dvalue_du: float
    dvalue_dv: float
    nondifferentiable: bool
    reasons: tuple[str, ...]


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


def _validate_texture(texture: object) -> tuple[tuple[tuple[float, ...], ...], int, int]:
    if isinstance(texture, (str, bytes)) or not isinstance(texture, Sequence):
        raise InvalidInputError(
            f"texture: expected a sequence of rows, got {type(texture).__name__}"
        )
    if len(texture) == 0:
        raise InvalidInputError("texture: must contain at least one row")
    rows: list[tuple[float, ...]] = []
    width: int | None = None
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
            converted.append(_as_finite_float(f"texture[{r}][{c}]", value))
        rows.append(tuple(converted))
    height = len(rows)
    assert width is not None
    if width > MAX_TEXTURE_SIZE or height > MAX_TEXTURE_SIZE:
        raise InvalidInputError(
            f"texture: size {width}x{height} exceeds the "
            f"{MAX_TEXTURE_SIZE}x{MAX_TEXTURE_SIZE} limit"
        )
    return tuple(rows), width, height


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


def _validate_boundary_tol(boundary_tol: object) -> float:
    tol = _as_finite_float("boundary_tol", boundary_tol)
    if tol < 0.0:
        raise InvalidInputError(f"boundary_tol: must be >= 0, got {tol}")
    return tol


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
) -> FilterResult:
    """Filter ``texture`` at ``(u, v)`` with the elliptical footprint of ``jacobian``.

    Args:
        texture: rectangular sequence of rows of scalar texel values,
            at most 64x64.  ``texture[j][i]`` is the texel whose center is
            at ``(i + 0.5, j + 0.5)`` in texture space.
        u, v: sampling coordinate in texture space (finite).
        jacobian: 2x2 screen-to-texture Jacobian ``J``; the footprint uses
            ``C = J * J^T + 0.25 * I``, so any finite ``J`` (including
            singular or zero) yields a positive definite ``C``.
        wrap: :class:`WrapMode` for out-of-range texel coordinates.
        budget: maximum number of candidate texel centers inside the
            footprint bounding box (default 4096).

    Returns:
        A :class:`FilterResult` with the filtered value, the normalized
        per-contribution weights and the actual contribution count.

    Raises:
        InvalidInputError: any argument violates the input contract.
        BudgetExceededError: the footprint bounding box holds more than
            ``budget`` candidate texels.  No partial result is produced
            and no point-sampling fallback is performed.
    """
    tex, width, height = _validate_texture(texture)
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
    weighted_value = 0.0
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
                weighted_value += w * tex[tj][ti]
                weights.append(w)
                texels.append((ti, tj))

    if not weights:
        # Unreachable: the texel center nearest to (u, v) is at squared
        # distance <= 0.5, and C^-1's largest eigenvalue is <= 1/0.25 = 4,
        # so its q is <= 2 < FOOTPRINT_RADIUS_SQUARED.
        raise EllipseFilterError("internal error: footprint contains no texel")

    value = weighted_value / total_weight
    normalized = tuple(w / total_weight for w in weights)
    return FilterResult(
        value=value,
        weights=normalized,
        texels=tuple(texels),
        count=len(weights),
        weight_sum=total_weight,
    )


def filter_sample_with_derivatives(
    texture: Sequence[Sequence[float]],
    u: float,
    v: float,
    jacobian: Sequence[Sequence[float]],
    *,
    wrap: WrapMode = WrapMode.REPEAT,
    budget: int = DEFAULT_BUDGET,
    boundary_tol: float = DEFAULT_BOUNDARY_TOL,
) -> GradientResult:
    """Like :func:`filter_sample`, but also differentiate w.r.t. ``u``/``v``.

    The Jacobian is held fixed.  Both the kernel weights and the
    normalization denominator are differentiated analytically
    (quotient rule); every repeat/clamp copy of a stored texel
    contributes its own weight derivative before the sums are
    normalized.  This is not a finite difference of final values.

    Args:
        texture, u, v, jacobian, wrap, budget: identical contract to
            :func:`filter_sample`.  The budget covers every candidate
            that participates in the differentiation; derivative mode
            cannot bypass the 4096 (or custom) candidate limit.
        boundary_tol: non-negative tolerance of the nondifferentiability
            probe.  Texel centers whose ``q`` lies within this distance
            of the footprint boundary (q = 4) mark the position as
            nondifferentiable.

    Returns:
        A :class:`GradientResult`.  ``result.sample`` is numerically
        identical to ``filter_sample(...)`` for the same arguments.

    Raises:
        InvalidInputError: any argument violates the input contract.
        BudgetExceededError: the footprint bounding box holds more than
            ``budget`` candidate texels (checked before any work, exactly
            as in :func:`filter_sample`).
    """
    tex, width, height = _validate_texture(texture)
    u = _as_finite_float("u", u)
    v = _as_finite_float("v", v)
    j00, j01, j10, j11 = _validate_jacobian(jacobian)
    wrap = _validate_wrap(wrap)
    budget = _validate_budget(budget)
    boundary_tol = _validate_boundary_tol(boundary_tol)

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

    inv00 = c11 / det
    inv01 = -c01 / det
    inv11 = c00 / det

    half_u = 2.0 * math.sqrt(c00)
    half_v = 2.0 * math.sqrt(c11)

    lo_i = math.ceil(u - half_u - 0.5)
    hi_i = math.floor(u + half_u - 0.5)
    lo_j = math.ceil(v - half_v - 0.5)
    hi_j = math.floor(v + half_v - 0.5)
    candidates = (hi_i - lo_i + 1) * (hi_j - lo_j + 1)
    if candidates > budget:
        raise BudgetExceededError(
            f"footprint bounding box holds {candidates} candidate texels, "
            f"exceeding the budget of {budget}",
            candidates=candidates,
            budget=budget,
        )

    total_weight = 0.0
    weighted_value = 0.0
    # d(sum w)/d(u,v) and d(sum w*t)/d(u,v): the normalization
    # denominator participates in the differentiation.
    dtotal_du = 0.0
    dtotal_dv = 0.0
    dweighted_du = 0.0
    dweighted_dv = 0.0
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
                t = tex[tj][ti]
                total_weight += w
                weighted_value += w * t
                # dq/du, dq/dv of q = d^T C^-1 d with d = (dx, dy).
                dq_du = -2.0 * (inv00 * dx + inv01 * dy)
                dq_dv = -2.0 * (inv01 * dx + inv11 * dy)
                dw_du = -KERNEL_SCALE * w * dq_du
                dw_dv = -KERNEL_SCALE * w * dq_dv
                dtotal_du += dw_du
                dtotal_dv += dw_dv
                dweighted_du += dw_du * t
                dweighted_dv += dw_dv * t
                weights.append(w)
                texels.append((ti, tj))

    if not weights:
        # Unreachable: see filter_sample.
        raise EllipseFilterError("internal error: footprint contains no texel")

    value = weighted_value / total_weight
    normalized = tuple(w / total_weight for w in weights)
    sample = FilterResult(
        value=value,
        weights=normalized,
        texels=tuple(texels),
        count=len(weights),
        weight_sum=total_weight,
    )

    # Quotient rule on value = weighted_value / total_weight.
    denom = total_weight * total_weight
    dvalue_du = (dweighted_du * total_weight - weighted_value * dtotal_du) / denom
    dvalue_dv = (dweighted_dv * total_weight - weighted_value * dtotal_dv) / denom

    reasons = _nondifferentiability_reasons(
        width, height, u, v, inv00, inv01, inv11,
        lo_i, hi_i, lo_j, hi_j, wrap, boundary_tol,
    )
    return GradientResult(
        sample=sample,
        dvalue_du=dvalue_du,
        dvalue_dv=dvalue_dv,
        nondifferentiable=bool(reasons),
        reasons=tuple(reasons),
    )


def _nondifferentiability_reasons(
    width: int,
    height: int,
    u: float,
    v: float,
    inv00: float,
    inv01: float,
    inv11: float,
    lo_i: int,
    hi_i: int,
    lo_j: int,
    hi_j: int,
    wrap: WrapMode,
    boundary_tol: float,
) -> list[str]:
    """Probe texel centers near the footprint boundary q == 4.

    The value is smooth in (u, v) while the contributing candidate set is
    fixed.  It changes (with a jump of weight e^-8) whenever a texel
    center crosses the footprint boundary; for out-of-range centers the
    repeat/clamp mapping of that candidate switches at the same locus.
    The probe scans the bounding box expanded by one texel on each side,
    which also covers candidates about to enter the scan range.
    """
    reasons: list[str] = []
    for jy in range(lo_j - 1, hi_j + 2):
        dy = (jy + 0.5) - v
        for ix in range(lo_i - 1, hi_i + 2):
            dx = (ix + 0.5) - u
            q = inv00 * dx * dx + 2.0 * inv01 * dx * dy + inv11 * dy * dy
            if abs(q - FOOTPRINT_RADIUS_SQUARED) > boundary_tol:
                continue
            state = "inside" if q <= FOOTPRINT_RADIUS_SQUARED else "outside"
            reason = (
                f"texel center ({ix}, {jy}) has q={q:.6g} ({state}, within "
                f"boundary_tol={boundary_tol} of the footprint boundary "
                f"q={FOOTPRINT_RADIUS_SQUARED}): the candidate set and its "
                f"weight sum change non-smoothly here"
            )
            if not (0 <= ix < width and 0 <= jy < height):
                if wrap is WrapMode.REPEAT:
                    reason += (
                        f"; the repeat mapping of this candidate switches "
                        f"at the same locus (currently wraps to stored "
                        f"texel ({ix % width}, {jy % height}))"
                    )
                else:
                    ci = 0 if ix < 0 else (width - 1 if ix >= width else ix)
                    cj = 0 if jy < 0 else (height - 1 if jy >= height else jy)
                    reason += (
                        f"; the clamp mapping of this candidate switches "
                        f"at the same locus (currently pins to stored "
                        f"texel ({ci}, {cj}))"
                    )
            reasons.append(reason)
    return reasons
