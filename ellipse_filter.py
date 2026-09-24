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

Tiled atlases
-------------
A :class:`TileAtlas` holds several independently sized scalar tiles, each
addressed by an explicit tile id.  :func:`filter_tile` filters one tile:
the footprint ellipse is laid out in that tile's own texture space and
out-of-range candidate centers are wrapped/clamped against *that tile's*
width and height only, so a footprint can never bleed into an unrelated
tile.  A wrap mode may be attached per tile (at atlas construction time)
and/or selected per call; the explicit call argument wins.

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
import types
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
    "TileNotFoundError",
    "FilterResult",
    "TileAtlas",
    "filter_sample",
    "filter_tile",
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


class TileNotFoundError(InvalidInputError):
    """A tile id does not exist in the atlas.

    Subclasses :class:`InvalidInputError` (hence ``ValueError``) so that an
    unknown id is reported through the same invalid-argument channel.

    Attributes:
        tile_id: the id that could not be resolved.
        available: tuple of ids the atlas does contain.
    """

    def __init__(self, message: str, *, tile_id: object,
                 available: tuple[object, ...]) -> None:
        super().__init__(message)
        self.tile_id = tile_id
        self.available = available


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
        tile_id: id of the sampled tile, or ``None`` for the legacy
            single-texture entry point :func:`filter_sample`.
    """

    value: float
    weights: tuple[float, ...]
    texels: tuple[tuple[int, int], ...]
    count: int
    weight_sum: float
    tile_id: object = None


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


def _validate_tile_id(name: str, tile_id: object) -> object:
    if tile_id is None:
        raise InvalidInputError(f"{name}: tile id must not be None")
    if isinstance(tile_id, bool):
        # Reject bools: True/False would silently collide with ids 1/0.
        raise InvalidInputError(
            f"{name}: tile id must not be a bool, got {tile_id!r}"
        )
    try:
        hash(tile_id)
    except TypeError:
        raise InvalidInputError(
            f"{name}: tile id must be hashable, got {type(tile_id).__name__}"
        ) from None
    return tile_id


# ---------------------------------------------------------------------------
# Tile atlas
# ---------------------------------------------------------------------------


class TileAtlas:
    """An immutable collection of independently sized, explicitly id-ed tiles.

    Build from a mapping ``{id: texture}`` / ``{id: (texture, wrap)}`` or a
    sequence of ``(id, texture[, wrap])`` triples.  Every tile is validated
    eagerly; if any tile is invalid the constructor raises before the atlas
    exists, so a malformed specification can never leave a half-populated
    atlas behind.

    Each tile may carry its own default :class:`WrapMode`; ``default_wrap``
    sets the fallback for tiles that do not name one.  Tiles never share
    storage and filtering never leaves the selected tile's own grid.
    """

    def __init__(
        self,
        tiles: object,
        *,
        default_wrap: WrapMode | None = None,
    ) -> None:
        if default_wrap is not None:
            default_wrap = _validate_wrap(default_wrap)
        is_mapping = callable(getattr(tiles, "items", None))
        if (
            not is_mapping
            and (isinstance(tiles, (str, bytes)) or not isinstance(tiles, Sequence))
        ):
            raise InvalidInputError(
                "tiles: expected a mapping or a sequence of (id, texture) "
                f"entries, got {type(tiles).__name__}"
            )
        entries = self._entries(tiles)

        resolved: dict[object, tuple[tuple[tuple[float, ...], ...], int, int,
                                     WrapMode | None]] = {}
        order: list[object] = []
        for position, entry in enumerate(entries):
            tile_id, texture, tile_wrap = self._parse_entry(entry, position)
            _validate_tile_id("tile id", tile_id)
            if tile_id in resolved:
                raise InvalidInputError(
                    f"tiles: duplicate tile id {tile_id!r}"
                )
            if tile_wrap is not None:
                tile_wrap = _validate_wrap(tile_wrap)
            tex, width, height = _validate_texture(texture)
            resolved[tile_id] = (tex, width, height, tile_wrap)
            order.append(tile_id)

        object.__setattr__(self, "_tiles", types.MappingProxyType(resolved))
        object.__setattr__(self, "_order", tuple(order))
        object.__setattr__(self, "_default_wrap", default_wrap)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("TileAtlas is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("TileAtlas is immutable")

    @staticmethod
    def _entries(tiles: Sequence) -> Sequence:
        """Normalize the constructor input to a sequence of entries.

        Mappings (``{id: texture}`` / ``{id: (texture, wrap)}``) become a
        sequence of ``(id, value)`` items; sequences pass through unchanged.
        """
        if callable(getattr(tiles, "items", None)):
            return list(tiles.items())
        return tiles

    @staticmethod
    def _parse_entry(entry: object, position: int):
        if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes)):
            raise InvalidInputError(
                f"tiles[{position}]: expected an (id, texture) or "
                f"(id, texture, wrap) entry, got {type(entry).__name__}"
            )
        if len(entry) == 2:
            tile_id, texture = entry
            tile_wrap = None
        elif len(entry) == 3:
            tile_id, texture, tile_wrap = entry
        else:
            raise InvalidInputError(
                f"tiles[{position}]: expected 2 or 3 fields "
                f"(id, texture[, wrap]), got {len(entry)}"
            )
        # Mapping values may be (texture, wrap) pairs.
        if (
            tile_wrap is None
            and isinstance(texture, tuple)
            and len(texture) == 2
            and isinstance(texture[1], WrapMode)
        ):
            texture, tile_wrap = texture
        return tile_id, texture, tile_wrap

    def __len__(self) -> int:
        return len(self._order)

    def __contains__(self, tile_id: object) -> bool:
        try:
            return tile_id in self._tiles
        except TypeError:
            return False

    @property
    def tile_ids(self) -> tuple[object, ...]:
        """All tile ids in construction order."""
        return self._order

    @property
    def default_wrap(self) -> WrapMode | None:
        """Atlas-wide fallback wrap mode, if one was given."""
        return self._default_wrap

    def size(self, tile_id: object) -> tuple[int, int]:
        """Return ``(width, height)`` of ``tile_id``.

        Raises:
            TileNotFoundError: the id is not in the atlas.
        """
        _, width, height, _ = self._resolve(tile_id)
        return width, height

    def wrap_for(self, tile_id: object) -> WrapMode | None:
        """Effective configured wrap of a tile (its own, else the atlas
        default, else ``None`` meaning the global call default)."""
        _, _, _, tile_wrap = self._resolve(tile_id)
        return tile_wrap if tile_wrap is not None else self._default_wrap

    def _resolve(
        self, tile_id: object
    ) -> tuple[tuple[tuple[float, ...], ...], int, int, WrapMode | None]:
        try:
            return self._tiles[tile_id]
        except TypeError:
            raise InvalidInputError(
                f"tile_id: expected a hashable id, got "
                f"{type(tile_id).__name__}"
            ) from None
        except KeyError:
            raise TileNotFoundError(
                f"tile_id: no tile with id {tile_id!r}; "
                f"available ids: {list(self._order)}",
                tile_id=tile_id,
                available=self._order,
            ) from None


# ---------------------------------------------------------------------------
# Core filter
# ---------------------------------------------------------------------------


def _resolve_coord(ix: int, jy: int, width: int, height: int,
                   wrap: WrapMode) -> tuple[int, int]:
    """Map a candidate texel center into one tile's stored grid.

    Repeat takes the modulus against this tile's own dimensions; clamp pins
    to this tile's own edges.  No other tile is ever consulted.
    """
    if wrap is WrapMode.REPEAT:
        return ix % width, jy % height
    ti = 0 if ix < 0 else (width - 1 if ix >= width else ix)
    tj = 0 if jy < 0 else (height - 1 if jy >= height else jy)
    return ti, tj


def _run_filter(
    tex: Sequence[Sequence[float]],
    width: int,
    height: int,
    u: float,
    v: float,
    j00: float,
    j01: float,
    j10: float,
    j11: float,
    wrap: WrapMode,
    budget: int,
    tile_id: object = None,
) -> FilterResult:
    """Shared math for :func:`filter_sample` and :func:`filter_tile`.

    All inputs are already validated; ``tex`` belongs to exactly one tile.
    """
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
                ti, tj = _resolve_coord(ix, jy, width, height, wrap)
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
        tile_id=tile_id,
    )


def filter_sample(
    texture: Sequence[Sequence[float]],
    u: float,
    v: float,
    jacobian: Sequence[Sequence[float]],
    *,
    wrap: WrapMode = WrapMode.REPEAT,
    budget: int = DEFAULT_BUDGET,
) -> FilterResult:
    """Filter a single ``texture`` at ``(u, v)`` with the elliptical footprint.

    Legacy single-texture entry point; behavior is unchanged.  The result's
    ``tile_id`` is always ``None``.

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
    return _run_filter(
        tex, width, height, u, v, j00, j01, j10, j11, wrap, budget
    )


def filter_tile(
    atlas: TileAtlas,
    tile_id: object,
    u: float,
    v: float,
    jacobian: Sequence[Sequence[float]],
    *,
    wrap: WrapMode | None = None,
    budget: int = DEFAULT_BUDGET,
) -> FilterResult:
    """Filter one tile of a :class:`TileAtlas` at ``(u, v)``.

    The footprint is resolved entirely inside the tile named by
    ``tile_id``: candidate centers outside that tile's own grid are wrapped
    or clamped against *its* width and height only.  No other tile's texels
    can enter the result, regardless of ``u``/``v`` or footprint size.

    Args:
        atlas: the tile atlas to sample from.
        tile_id: explicit id of the tile to filter.
        u, v: sampling coordinate in the tile's own texture space (finite).
        jacobian: 2x2 screen-to-texture Jacobian ``J``.
        wrap: :class:`WrapMode` for this call.  ``None`` (the default) uses
            the tile's own configured wrap, else the atlas ``default_wrap``,
            else :attr:`WrapMode.REPEAT`.  An explicit value always wins.
        budget: maximum number of candidate texel centers inside the
            footprint bounding box (default 4096).

    Returns:
        A :class:`FilterResult` whose ``tile_id`` echoes ``tile_id`` and
        whose ``texels`` are indices within that tile.

    Raises:
        InvalidInputError: an argument violates the input contract.
        TileNotFoundError: ``tile_id`` is absent from ``atlas``.
        BudgetExceededError: the footprint bounding box holds more than
            ``budget`` candidate texels; no partial result is produced.
    """
    if not isinstance(atlas, TileAtlas):
        raise InvalidInputError(
            f"atlas: expected a TileAtlas, got {type(atlas).__name__}"
        )
    tile_id = _validate_tile_id("tile_id", tile_id)
    u = _as_finite_float("u", u)
    v = _as_finite_float("v", v)
    j00, j01, j10, j11 = _validate_jacobian(jacobian)
    if wrap is None:
        wrap = atlas.wrap_for(tile_id)
        if wrap is None:
            wrap = WrapMode.REPEAT
    else:
        wrap = _validate_wrap(wrap)
    budget = _validate_budget(budget)
    tex, width, height, _ = atlas._resolve(tile_id)
    return _run_filter(
        tex, width, height, u, v, j00, j01, j10, j11, wrap, budget,
        tile_id=tile_id,
    )
