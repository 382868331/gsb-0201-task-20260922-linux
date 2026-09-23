"""Elliptical (EWA-style) footprint filtering for scalar textures.

Reusable, dependency-free module (Python standard library only).

Model
-----
Given a scalar texture and a 2x2 screen-to-texture Jacobian ``J`` at the
sample point ``(u, v)``, the filter covariance is

    C = J @ J.T + 0.25 * I

The reconstruction footprint is the ellipse

    q(d) = d^T @ C^-1 @ d <= 4

evaluated at texel centers ``d = (i + 0.5 - u, j + 0.5 - v)``.  Each texel
center inside the footprint contributes its texel value with the Gaussian
weight ``exp(-2 * q)``; the filtered value is the weighted average
``sum(w_k * t_k) / sum(w_k)``.

Coordinate convention
---------------------
Texture coordinates are in *texel units*: texel ``(i, j)`` covers the square
``[i, i+1) x [j, j+1)`` and its center is ``(i + 0.5, j + 0.5)``.  ``u`` maps
to the horizontal (column) axis, ``v`` to the vertical (row) axis.  The
Jacobian maps screen pixels to these texel units.

Candidate enumeration walks the axis-aligned bounding box of the ellipse
(half extents ``2*sqrt(C11)`` and ``2*sqrt(C22)``).  If the bounding box
holds more than ``max_candidates`` texels (default 4096), a
:class:`BudgetExceededError` is raised *before* any sampling — the function
never silently falls back to point sampling.

Wrap modes
----------
``"repeat"`` maps texel indices modulo the texture size; ``"clamp"`` clamps
them to the valid range.  In both modes every *candidate position* keeps its
own kernel weight: if several candidate positions map to the same texel,
that texel appears once per position in the contributions (its copies are
NOT merged).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

__all__ = [
    "MAX_TEXTURE_SIZE",
    "DEFAULT_MAX_CANDIDATES",
    "WRAP_REPEAT",
    "WRAP_CLAMP",
    "FootprintError",
    "InvalidInputError",
    "BudgetExceededError",
    "Contribution",
    "FilterResult",
    "filter_footprint",
]

#: Maximum texture width/height accepted by :func:`filter_footprint`.
MAX_TEXTURE_SIZE = 64

#: Default upper bound on bounding-box candidate texels per sample.
DEFAULT_MAX_CANDIDATES = 4096

#: Wrap mode: texel indices wrap modulo the texture size.
WRAP_REPEAT = "repeat"
#: Wrap mode: texel indices clamp to the texture edge.
WRAP_CLAMP = "clamp"
_WRAP_MODES = (WRAP_REPEAT, WRAP_CLAMP)

# Footprint definition: q(d) = d^T C^-1 d <= _FOOTPRINT_RADIUS_SQ.
_FOOTPRINT_RADIUS_SQ = 4.0
# Kernel: weight = exp(-_GAUSSIAN_SCALE * q).
_GAUSSIAN_SCALE = 2.0
# Regularization: C = J J^T + _REGULARIZATION * I  (a 0.5-texel base filter).
_REGULARIZATION = 0.25


class FootprintError(Exception):
    """Base class for all errors raised by this module."""


class InvalidInputError(FootprintError, ValueError):
    """A supplied argument violates the module contract.

    The message names the offending parameter so the caller can locate it.
    Nothing is silently corrected (no NaN clamping, no wrap-mode guessing).
    """


class BudgetExceededError(FootprintError):
    """The footprint bounding box exceeds the candidate budget.

    Distinct from input-validation failures: the inputs were well formed,
    but the search budget ran out.  Attributes carry the numbers involved.
    """

    def __init__(self, candidates: int, limit: int) -> None:
        self.candidates = candidates
        self.limit = limit
        super().__init__(
            f"footprint bounding box holds {candidates} candidate texels, "
            f"exceeding the budget of {limit}"
        )


@dataclass(frozen=True)
class Contribution:
    """One weighted texel contribution to a filtered sample.

    ``texel_x``/``texel_y`` are the texture indices *after* wrap mapping.
    ``weight`` is the raw (unnormalized) kernel weight ``exp(-2*q)``; divide
    by :attr:`FilterResult.weight_sum` to get the normalized weight.
    """

    texel_x: int
    texel_y: int
    weight: float


@dataclass(frozen=True)
class FilterResult:
    """Result of :func:`filter_footprint`.

    ``value`` is the normalized filtered value, ``weight_sum`` the sum of
    raw kernel weights, ``count`` the number of contributing texel
    positions, and ``contributions`` the per-position breakdown (a texel
    may appear multiple times when several candidate positions map to it).
    """

    value: float
    weight_sum: float
    count: int
    contributions: Tuple[Contribution, ...]


def _is_finite_real(x: object) -> bool:
    # bool is a subclass of int; reject it everywhere a real number is due.
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(x)
    )


def _coerce_texture(texture: object) -> Tuple[Tuple[Tuple[float, ...], ...], int, int]:
    if isinstance(texture, (str, bytes)):
        raise InvalidInputError("texture must be a sequence of rows, not a string")
    try:
        rows = list(texture)  # type: ignore[arg-type]
    except TypeError:
        raise InvalidInputError("texture must be a sequence of rows") from None
    if not rows:
        raise InvalidInputError("texture must have at least one row")
    grid = []
    width = -1
    for r, row in enumerate(rows):
        if isinstance(row, (str, bytes)):
            raise InvalidInputError(f"texture row {r} must be a sequence, not a string")
        try:
            vals = list(row)
        except TypeError:
            raise InvalidInputError(f"texture row {r} is not a sequence") from None
        if r == 0:
            width = len(vals)
            if width == 0:
                raise InvalidInputError("texture rows must be non-empty")
            if width > MAX_TEXTURE_SIZE:
                raise InvalidInputError(
                    f"texture width {width} exceeds the limit of {MAX_TEXTURE_SIZE}"
                )
        elif len(vals) != width:
            raise InvalidInputError(
                f"texture row {r} has length {len(vals)}, expected {width} (ragged texture)"
            )
        out = []
        for c, val in enumerate(vals):
            if not _is_finite_real(val):
                raise InvalidInputError(
                    f"texture[{r}][{c}]={val!r} is not a finite real number"
                )
            out.append(float(val))
        grid.append(tuple(out))
    if len(grid) > MAX_TEXTURE_SIZE:
        raise InvalidInputError(
            f"texture height {len(grid)} exceeds the limit of {MAX_TEXTURE_SIZE}"
        )
    return tuple(grid), width, len(grid)


def _coerce_jacobian(jacobian: object) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    try:
        rows = [list(r) for r in jacobian]  # type: ignore[union-attr]
    except TypeError:
        raise InvalidInputError("jacobian must be a 2x2 sequence of rows") from None
    if len(rows) != 2 or any(len(r) != 2 for r in rows):
        raise InvalidInputError(
            f"jacobian must be 2x2, got row lengths {[len(r) for r in rows]}"
        )
    out = []
    for r in range(2):
        for c in range(2):
            val = rows[r][c]
            if not _is_finite_real(val):
                raise InvalidInputError(
                    f"jacobian[{r}][{c}]={val!r} is not a finite real number"
                )
            out.append(float(val))
    return ((out[0], out[1]), (out[2], out[3]))


def _coerce_max_candidates(max_candidates: object) -> int:
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int):
        raise InvalidInputError(
            f"max_candidates must be an int, got {type(max_candidates).__name__}"
        )
    if max_candidates < 1:
        raise InvalidInputError(
            f"max_candidates must be >= 1, got {max_candidates}"
        )
    return max_candidates


def filter_footprint(
    texture: Sequence[Sequence[float]],
    u: float,
    v: float,
    jacobian: Sequence[Sequence[float]],
    *,
    wrap: str = WRAP_REPEAT,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> FilterResult:
    """Filter ``texture`` at ``(u, v)`` with the elliptical footprint of ``jacobian``.

    Parameters
    ----------
    texture:
        Rectangular sequence of rows of finite reals, width and height in
        ``[1, MAX_TEXTURE_SIZE]``.  ``texture[y][x]`` is texel ``(x, y)``.
    u, v:
        Sample position in texel units (finite reals).
    jacobian:
        2x2 screen-to-texture Jacobian ``[[du/dx, du/dy], [dv/dx, dv/dy]]``
        of finite reals.
    wrap:
        ``WRAP_REPEAT`` or ``WRAP_CLAMP``.
    max_candidates:
        Budget on bounding-box candidate texels (int >= 1).  Exceeding it
        raises :class:`BudgetExceededError` before any sampling.

    Returns
    -------
    FilterResult
        ``value`` (normalized), ``weight_sum``, ``count`` and the raw
        per-position ``contributions``.

    Raises
    ------
    InvalidInputError
        Any argument violates the contract (NaN/Infinity, bools where
        numbers are due, ragged/oversized texture, unknown wrap mode, ...).
    BudgetExceededError
        The footprint bounding box holds more than ``max_candidates``
        texels.  No partial result is produced.
    """
    grid, width, height = _coerce_texture(texture)
    if not _is_finite_real(u):
        raise InvalidInputError(f"u={u!r} is not a finite real number")
    if not _is_finite_real(v):
        raise InvalidInputError(f"v={v!r} is not a finite real number")
    u = float(u)
    v = float(v)
    (j00, j01), (j10, j11) = _coerce_jacobian(jacobian)
    if wrap not in _WRAP_MODES:
        raise InvalidInputError(f"wrap must be one of {_WRAP_MODES}, got {wrap!r}")
    limit = _coerce_max_candidates(max_candidates)

    # C = J J^T + 0.25 I  (symmetric positive definite, det >= 0.0625 > 0).
    c11 = j00 * j00 + j01 * j01 + _REGULARIZATION
    c12 = j00 * j10 + j01 * j11
    c22 = j10 * j10 + j11 * j11 + _REGULARIZATION
    det = c11 * c22 - c12 * c12
    i11 = c22 / det
    i12 = -c12 / det
    i22 = c11 / det

    # Axis-aligned bounding box of {d : d^T C^-1 d <= 4}.
    rx = 2.0 * math.sqrt(c11)
    ry = 2.0 * math.sqrt(c22)
    i0 = math.ceil(u - rx - 0.5)
    i1 = math.floor(u + rx - 0.5)
    j0 = math.ceil(v - ry - 0.5)
    j1 = math.floor(v + ry - 0.5)
    candidates = (i1 - i0 + 1) * (j1 - j0 + 1)
    if candidates > limit:
        raise BudgetExceededError(candidates, limit)

    contributions = []
    weighted = 0.0
    weight_sum = 0.0
    for tj in range(j0, j1 + 1):
        dy = (tj + 0.5) - v
        for ti in range(i0, i1 + 1):
            dx = (ti + 0.5) - u
            q = (i11 * dx + i12 * dy) * dx + (i12 * dx + i22 * dy) * dy
            if q > _FOOTPRINT_RADIUS_SQ:
                continue
            w = math.exp(-_GAUSSIAN_SCALE * q)
            if wrap == WRAP_REPEAT:
                mi = ti % width
                mj = tj % height
            else:  # WRAP_CLAMP
                mi = 0 if ti < 0 else (width - 1 if ti >= width else ti)
                mj = 0 if tj < 0 else (height - 1 if tj >= height else tj)
            contributions.append(Contribution(mi, mj, w))
            weighted += w * grid[mj][mi]
            weight_sum += w

    # weight_sum is always > 0: C^-1 <= 4 I, and the nearest texel center is
    # within |d|^2 <= 0.5, so q <= 2 < 4 — at least one texel contributes.
    return FilterResult(
        value=weighted / weight_sum,
        weight_sum=weight_sum,
        count=len(contributions),
        contributions=tuple(contributions),
    )
