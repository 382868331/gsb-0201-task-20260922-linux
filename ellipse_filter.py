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

:func:`filter_sample_grad` additionally returns the analytic derivatives
of the filtered value with respect to the sampling center ``(u, v)``
(Jacobian held fixed), differentiating both the kernel weights and the
normalization denominator, and flags positions where the value is not
differentiable (footprint boundary, candidate-set / wrap-copy switches).

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
    "EllipseFilterError",
    "InvalidInputError",
    "BudgetExceededError",
    "FilterResult",
    "GradientResult",
    "filter_sample",
    "filter_sample_grad",
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
    """Result of one footprint-filtered sample with center derivatives.

    Attributes:
        value: normalized filtered value, identical to
            FilterResult.value for the same inputs.
        grad_u, grad_v: analytic derivatives of value with respect to the
            sampling center u / v (Jacobian held fixed).  Both the kernel
            weights and the normalization denominator are differentiated
            (quotient rule); every repeat/clamp copy of a stored texel
            contributes the derivative of its own kernel weight.
        differentiable: False when (u, v) sits on or within tolerance of
            a nonsmooth point; the gradient fields are still filled in but
            must not be trusted there.
        nondifferentiable_reason: human-readable cause when
            differentiable is False, else None.
        weights, texels, count, weight_sum: same contract as FilterResult.
    """

    value: float
    grad_u: float
    grad_v: float
    differentiable: bool
    nondifferentiable_reason: str | None
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


# ---------------------------------------------------------------------------
# Core filter
# ---------------------------------------------------------------------------

def _prepare_footprint(texture, u, v, jacobian, wrap, budget):
    """Validate inputs and compute the footprint scan setup.

    Returns (tex, width, height, u, v, wrap, inv00, inv01, inv11,
    lo_i, hi_i, lo_j, hi_j): the validated texture, sampling center, wrap
    mode, the entries of C^-1 and the integer bounding-box range of
    candidate texel indices.  Raises InvalidInputError or
    BudgetExceededError on contract violations.
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
    return (tex, width, height, u, v, wrap, inv00, inv01, inv11,
            lo_i, hi_i, lo_j, hi_j, half_u, half_v)


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
    (tex, width, height, u, v, wrap, inv00, inv01, inv11,
     lo_i, hi_i, lo_j, hi_j, _half_u, _half_v) = _prepare_footprint(
        texture, u, v, jacobian, wrap, budget
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


def _distance_to_integer(x: float) -> float:
    return abs(x - round(x))


def filter_sample_grad(
    texture: Sequence[Sequence[float]],
    u: float,
    v: float,
    jacobian: Sequence[Sequence[float]],
    *,
    wrap: WrapMode = WrapMode.REPEAT,
    budget: int = DEFAULT_BUDGET,
    boundary_tol: float = 1e-9,
    edge_tol: float = 1e-9,
) -> GradientResult:
    """Like filter_sample, but also differentiates the value w.r.t. (u, v).

    The Jacobian is held fixed.  With value = A / B, A = sum(w_k * t_k),
    B = sum(w_k) and w_k = exp(-2 * q_k), the analytic derivatives are

        dq_k/du = -2 * (inv00 * dx + inv01 * dy)
        dq_k/dv = -2 * (inv01 * dx + inv11 * dy)
        dw_k/d* = -2 * w_k * dq_k/d*
        dvalue/d* = (dA/d* - value * dB/d*) / B

    so both the kernel weights and the normalization denominator are
    differentiated.  Repeat/clamp copies of the same stored texel each
    carry their own kernel weight and therefore their own weight
    derivative; nothing is obtained by finite-differencing the final
    pixel value.

    The result is marked nondifferentiable (with a reason) when (u, v)
    lies on or within tolerance of a nonsmooth configuration:

    * a bounding-box candidate whose q is within boundary_tol of the
      footprint boundary 4.0: the kernel has a hard cutoff there, so the
      weight jumps by exp(-8) as the candidate enters/leaves;
    * a bounding-box edge within edge_tol of an integer texel index:
      the candidate set (and thus the repeat/clamp copy multiplicity of
      stored texels) changes discretely.

    The candidate budget is enforced exactly as in filter_sample;
    derivative mode cannot bypass the 4096 default limit.

    Raises:
        InvalidInputError: any argument violates the input contract
            (including non-finite or negative tolerances).
        BudgetExceededError: the footprint bounding box holds more than
            budget candidate texels.
    """
    boundary_tol = _as_finite_float("boundary_tol", boundary_tol)
    edge_tol = _as_finite_float("edge_tol", edge_tol)
    if boundary_tol < 0.0:
        raise InvalidInputError(
            f"boundary_tol: must be >= 0, got {boundary_tol}")
    if edge_tol < 0.0:
        raise InvalidInputError(
            f"edge_tol: must be >= 0, got {edge_tol}")
    (tex, width, height, u, v, wrap, inv00, inv01, inv11,
     lo_i, hi_i, lo_j, hi_j, half_u, half_v) = _prepare_footprint(
        texture, u, v, jacobian, wrap, budget
    )

    total_weight = 0.0
    weighted_value = 0.0
    dweight_du_total = 0.0
    dweight_dv_total = 0.0
    dvalue_du_total = 0.0
    dvalue_dv_total = 0.0
    weights: list[float] = []
    texels: list[tuple[int, int]] = []
    boundary_hit: tuple[int, int, float] | None = None
    for jy in range(lo_j, hi_j + 1):
        dy = (jy + 0.5) - v
        for ix in range(lo_i, hi_i + 1):
            dx = (ix + 0.5) - u
            q = inv00 * dx * dx + 2.0 * inv01 * dx * dy + inv11 * dy * dy
            if abs(q - FOOTPRINT_RADIUS_SQUARED) <= boundary_tol:
                if boundary_hit is None or \
                        abs(q - FOOTPRINT_RADIUS_SQUARED) < abs(
                            boundary_hit[2] - FOOTPRINT_RADIUS_SQUARED):
                    boundary_hit = (ix, jy, q)
            if q <= FOOTPRINT_RADIUS_SQUARED:
                w = math.exp(-KERNEL_SCALE * q)
                if wrap is WrapMode.REPEAT:
                    ti = ix % width
                    tj = jy % height
                else:
                    ti = 0 if ix < 0 else (width - 1 if ix >= width else ix)
                    tj = 0 if jy < 0 else (height - 1 if jy >= height else jy)
                t = tex[tj][ti]
                # dw/du = 4*w*(inv00*dx + inv01*dy),
                # dw/dv = 4*w*(inv01*dx + inv11*dy).
                dw_du = 2.0 * KERNEL_SCALE * w * (inv00 * dx + inv01 * dy)
                dw_dv = 2.0 * KERNEL_SCALE * w * (inv01 * dx + inv11 * dy)
                total_weight += w
                weighted_value += w * t
                dweight_du_total += dw_du
                dweight_dv_total += dw_dv
                dvalue_du_total += dw_du * t
                dvalue_dv_total += dw_dv * t
                weights.append(w)
                texels.append((ti, tj))

    if not weights:
        # Unreachable: see filter_sample.
        raise EllipseFilterError("internal error: footprint contains no texel")

    value = weighted_value / total_weight
    grad_u = (dvalue_du_total - value * dweight_du_total) / total_weight
    grad_v = (dvalue_dv_total - value * dweight_dv_total) / total_weight
    normalized = tuple(w / total_weight for w in weights)

    reason: str | None = None
    if boundary_hit is not None:
        bx, by, bq = boundary_hit
        reason = (
            f"candidate texel ({bx}, {by}) has q={bq!r}, within "
            f"boundary_tol={boundary_tol!r} of the footprint boundary "
            f"{FOOTPRINT_RADIUS_SQUARED}: the kernel cutoff makes the "
            "weight jump by exp(-8) as the candidate enters or leaves"
        )
    else:
        # A bounding-box edge sitting exactly on a texel center means the
        # candidate set (and the repeat/clamp copy multiplicity of stored
        # texels) changes discretely under an infinitesimal move.
        edge_dist = min(
            _distance_to_integer(u - half_u - 0.5),
            _distance_to_integer(u + half_u - 0.5),
            _distance_to_integer(v - half_v - 0.5),
            _distance_to_integer(v + half_v - 0.5),
        )
        if edge_dist <= edge_tol:
            reason = (
                f"footprint bounding-box edge is within edge_tol="
                f"{edge_tol!r} of a texel center (distance {edge_dist!r}): "
                "the candidate set and the repeat/clamp copy multiplicity "
                "change discretely here"
            )
    return GradientResult(
        value=value,
        grad_u=grad_u,
        grad_v=grad_v,
        differentiable=reason is None,
        nondifferentiable_reason=reason,
        weights=normalized,
        texels=tuple(texels),
        count=len(weights),
        weight_sum=total_weight,
    )
