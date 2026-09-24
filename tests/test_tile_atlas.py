"""Tests for atlas/tile sampling: filter_sample_atlas and Tile.

The brute-force reference in this file (``reference_atlas``) is an
independent re-implementation: it derives C and C^-1 in closed form,
scans a deliberately oversized ("wide") box in tile-local coordinates,
remaps into the tile rectangle with plain arithmetic, and reads the
atlas directly.  It never imports or calls the module under test for
expected values.  Hand-computed cases use closed-form kernel weights
exp(-8) derived from the zero-Jacobian footprint (C = 0.25 I).
"""

import math
import unittest

from ellipse_filter import (
    BudgetExceededError,
    InvalidInputError,
    Tile,
    WrapMode,
    filter_sample,
    filter_sample_atlas,
)


def make_texture(width, height, fn):
    return [[fn(i, j) for i in range(width)] for j in range(height)]


def pseudo_random_texture(width, height):
    """Deterministic non-constant texture in [0, 1)."""
    def value(i, j):
        x = math.sin(i * 12.9898 + j * 78.233) * 43758.5453
        return x - math.floor(x)
    return make_texture(width, height, value)


def reference_atlas(atlas, rect, wrap, u, v, jacobian, box_radius=12):
    """Independent wide-box reference in tile-local coordinates.

    ``rect`` is (x, y, width, height) of the tile inside ``atlas``.
    Same mathematical contract as the module, derived here from scratch:
    C = J*J^T + 0.25I, kernel exp(-2 q) for q = d^T C^-1 d <= 4, with
    out-of-tile candidates remapped into the tile rectangle.
    """
    rx, ry, rw, rh = rect
    (j00, j01), (j10, j11) = jacobian
    c00 = j00 * j00 + j01 * j01 + 0.25
    c01 = j00 * j10 + j01 * j11
    c11 = j10 * j10 + j11 * j11 + 0.25
    det = c00 * c11 - c01 * c01
    m00, m01, m11 = c11 / det, -c01 / det, c00 / det

    total = 0.0
    accum = 0.0
    count = 0
    ci = math.floor(u)
    cj = math.floor(v)
    for jy in range(cj - box_radius, cj + box_radius + 1):
        for ix in range(ci - box_radius, ci + box_radius + 1):
            dx = ix + 0.5 - u
            dy = jy + 0.5 - v
            q = m00 * dx * dx + 2.0 * m01 * dx * dy + m11 * dy * dy
            if q <= 4.0:
                w = math.exp(-2.0 * q)
                if wrap is WrapMode.REPEAT:
                    ti, tj = ix % rw, jy % rh
                else:
                    ti = min(max(ix, 0), rw - 1)
                    tj = min(max(jy, 0), rh - 1)
                total += w
                accum += w * atlas[ry + tj][rx + ti]
                count += 1
    return accum / total, total, count


def assert_confined(testcase, result, rect):
    """Every sampled atlas texel must lie inside the tile rectangle."""
    rx, ry, rw, rh = rect
    testcase.assertEqual(result.count, len(result.texels))
    testcase.assertGreater(result.count, 0)
    for (i, j) in result.texels:
        testcase.assertTrue(rx <= i < rx + rw,
                            f"texel ({i}, {j}) escapes tile rect {rect}")
        testcase.assertTrue(ry <= j < ry + rh,
                            f"texel ({i}, {j}) escapes tile rect {rect}")


ATLAS16 = pseudo_random_texture(16, 16)
TILES16 = {
    "alpha": Tile(0, 0, 4, 4, WrapMode.CLAMP),
    "beta": Tile(4, 0, 8, 2, WrapMode.REPEAT),
    "gamma": Tile(0, 4, 3, 5, WrapMode.CLAMP),
    7: Tile(12, 12, 1, 1, WrapMode.REPEAT),  # non-string id, 1x1 tile
}


class TestNoBleedIntoNeighbourTiles(unittest.TestCase):
    """Footprints near a tile edge must never read neighbouring tiles."""

    def setUp(self):
        # Two constant 8x8 tiles side by side in a 16x8 atlas.
        self.atlas = make_texture(16, 8,
                                  lambda i, j: 1.0 if i < 8 else 100.0)
        self.tiles = {
            "left": Tile(0, 0, 8, 8, WrapMode.CLAMP),
            "right": Tile(8, 0, 8, 8, WrapMode.REPEAT),
        }
        # Footprint reaches ~4.4 texels past the sample point: sampling
        # near the shared seam at local u = 7.5 crosses into the other
        # tile unless the sampler confines it.
        self.jacobian = ((1.5, 0.0), (0.0, 1.5))

    def test_clamp_tile_keeps_its_own_constant(self):
        result = filter_sample_atlas(
            self.atlas, self.tiles, "left", 7.5, 4.0, self.jacobian
        )
        self.assertAlmostEqual(result.value, 1.0, delta=1e-12)
        assert_confined(self, result, (0, 0, 8, 8))

    def test_repeat_tile_wraps_within_itself(self):
        result = filter_sample_atlas(
            self.atlas, self.tiles, "right", 0.5, 4.0, self.jacobian
        )
        self.assertAlmostEqual(result.value, 100.0, delta=1e-12)
        assert_confined(self, result, (8, 0, 8, 8))

    def test_whole_atlas_sample_does_bleed(self):
        # Contrast: the legacy single-texture call over the whole atlas
        # at the same atlas coordinate mixes both constants.
        result = filter_sample(self.atlas, 7.5, 4.0, self.jacobian,
                               wrap=WrapMode.CLAMP)
        self.assertGreater(result.value, 1.0)
        self.assertLess(result.value, 100.0)


class TestCornerHandComputed(unittest.TestCase):
    """Closed-form checks at a tile corner with a zero Jacobian.

    Zero Jacobian -> C = 0.25 I, C^-1 = 4 I, footprint radius 1: the
    contributing candidate centers are exactly (0,0), (+-1,0), (0,+-1)
    around the sampled center, with weights 1 and e^-8.
    """

    def setUp(self):
        # Distinct value per atlas texel: atlas[j][i] = i + 10 j.
        self.atlas = make_texture(5, 5, lambda i, j: float(i + 10 * j))
        self.rect = (1, 1, 3, 3)  # tile covers atlas (1..3, 1..3)
        self.zero_j = ((0.0, 0.0), (0.0, 0.0))
        self.edge = math.exp(-8.0)

    def local_value(self, li, lj):
        rx, ry, _, _ = self.rect
        return float((rx + li) + 10 * (ry + lj))

    def test_clamp_at_top_left_corner(self):
        tiles = {"t": Tile(*self.rect, WrapMode.CLAMP)}
        result = filter_sample_atlas(
            self.atlas, tiles, "t", 0.5, 0.5, self.zero_j
        )
        e = self.edge
        # Candidates (-1,0) and (0,-1) clamp onto (0,0) but keep their
        # own kernel weights: (0,0) contributes 1 + 2e.
        expected = (self.local_value(0, 0) * (1.0 + 2.0 * e)
                    + self.local_value(1, 0) * e
                    + self.local_value(0, 1) * e) / (1.0 + 4.0 * e)
        self.assertAlmostEqual(result.value, expected, delta=1e-12)
        self.assertEqual(result.count, 5)
        rx, ry, _, _ = self.rect
        self.assertEqual(
            sorted(result.texels),
            [(rx, ry), (rx, ry), (rx, ry), (rx, ry + 1), (rx + 1, ry)],
        )

    def test_repeat_at_top_left_corner(self):
        tiles = {"t": Tile(*self.rect, WrapMode.REPEAT)}
        result = filter_sample_atlas(
            self.atlas, tiles, "t", 0.5, 0.5, self.zero_j
        )
        e = self.edge
        # Candidate (-1,0) wraps to local column 2, (0,-1) to local row 2.
        expected = (self.local_value(0, 0)
                    + e * (self.local_value(2, 0) + self.local_value(1, 0)
                           + self.local_value(0, 2) + self.local_value(0, 1))
                    ) / (1.0 + 4.0 * e)
        self.assertAlmostEqual(result.value, expected, delta=1e-12)
        self.assertEqual(result.count, 5)
        assert_confined(self, result, self.rect)

    def test_clamp_at_bottom_right_corner(self):
        tiles = {"t": Tile(*self.rect, WrapMode.CLAMP)}
        result = filter_sample_atlas(
            self.atlas, tiles, "t", 2.5, 2.5, self.zero_j
        )
        e = self.edge
        # Candidates (3,2) and (2,3) clamp onto the corner texel (2,2).
        expected = (self.local_value(2, 2) * (1.0 + 2.0 * e)
                    + self.local_value(1, 2) * e
                    + self.local_value(2, 1) * e) / (1.0 + 4.0 * e)
        self.assertAlmostEqual(result.value, expected, delta=1e-12)
        assert_confined(self, result, self.rect)


class TestNegativeUV(unittest.TestCase):
    """Negative tile-local UVs are legal and follow the tile's wrap mode."""

    def setUp(self):
        self.atlas = ATLAS16
        self.rect = (0, 4, 3, 5)  # tile "gamma"
        self.zero_j = ((0.0, 0.0), (0.0, 0.0))

    def test_repeat_negative_uv_wraps_into_same_tile(self):
        tiles = {"gamma": Tile(*self.rect, WrapMode.REPEAT)}
        result = filter_sample_atlas(
            self.atlas, tiles, "gamma", -0.5, -1.5, self.zero_j
        )
        # u = -0.5: center column ix = -1, neighbours ix in {-2, 0};
        # mod 3 these are local columns {2, 1, 0} — all inside the tile.
        local_cols = {i - self.rect[0] for (i, j) in result.texels}
        self.assertEqual(local_cols, {0, 1, 2})
        expected, expected_sum, expected_count = reference_atlas(
            self.atlas, self.rect, WrapMode.REPEAT, -0.5, -1.5, self.zero_j
        )
        self.assertEqual(result.count, expected_count)
        self.assertAlmostEqual(result.value, expected, delta=1e-9)
        self.assertAlmostEqual(result.weight_sum, expected_sum, delta=1e-9)
        assert_confined(self, result, self.rect)

    def test_clamp_negative_uv_sticks_to_tile_edge(self):
        tiles = {"gamma": Tile(*self.rect, WrapMode.CLAMP)}
        result = filter_sample_atlas(
            self.atlas, tiles, "gamma", -0.5, 2.5, self.zero_j
        )
        # All candidate columns ix in {-2, -1, 0} clamp to local column 0.
        local_cols = {i - self.rect[0] for (i, j) in result.texels}
        self.assertEqual(local_cols, {0})
        expected, _, expected_count = reference_atlas(
            self.atlas, self.rect, WrapMode.CLAMP, -0.5, 2.5, self.zero_j
        )
        self.assertEqual(result.count, expected_count)
        self.assertAlmostEqual(result.value, expected, delta=1e-9)
        assert_confined(self, result, self.rect)

    def test_uv_beyond_tile_size(self):
        # u = 7.5 on a width-3 tile: repeat wraps (7.5 -> local 1.5 mod 3),
        # clamp sticks to the far edge.  Both stay inside the tile.
        for wrap in (WrapMode.REPEAT, WrapMode.CLAMP):
            tiles = {"gamma": Tile(*self.rect, wrap)}
            result = filter_sample_atlas(
                self.atlas, tiles, "gamma", 7.5, 9.5, self.zero_j
            )
            expected, _, expected_count = reference_atlas(
                self.atlas, self.rect, wrap, 7.5, 9.5, self.zero_j
            )
            self.assertEqual(result.count, expected_count)
            self.assertAlmostEqual(result.value, expected, delta=1e-9)
            assert_confined(self, result, self.rect)


class TestDifferentSizedTiles(unittest.TestCase):
    def test_matches_independent_reference_per_tile(self):
        jacobian = ((1.3, 0.6), (-0.4, 1.1))
        samples = {
            "alpha": [(0.4, 0.5), (3.9, 3.1), (-1.2, 5.7)],
            "beta": [(7.9, 0.5), (-3.3, 1.8)],
            "gamma": [(1.5, 2.5), (2.9, -0.4)],
            7: [(0.5, 0.5), (-2.0, 3.0)],
        }
        for tile_id, points in samples.items():
            tile = TILES16[tile_id]
            rect = (tile.x, tile.y, tile.width, tile.height)
            for (u, v) in points:
                with self.subTest(tile=tile_id, u=u, v=v):
                    expected, expected_sum, expected_count = reference_atlas(
                        ATLAS16, rect, tile.wrap, u, v, jacobian
                    )
                    result = filter_sample_atlas(
                        ATLAS16, TILES16, tile_id, u, v, jacobian
                    )
                    self.assertEqual(result.count, expected_count)
                    self.assertAlmostEqual(result.value, expected, delta=1e-9)
                    self.assertAlmostEqual(result.weight_sum, expected_sum,
                                           delta=1e-9)
                    assert_confined(self, result, rect)

    def test_one_by_one_tile_is_constant_for_any_uv(self):
        # A 1x1 tile has exactly one texel; repeat/clamp both land on it.
        expected_value = ATLAS16[12][12]
        for u, v in ((0.5, 0.5), (-4.2, 9.9), (100.5, -100.5)):
            result = filter_sample_atlas(
                ATLAS16, TILES16, 7, u, v, ((1.0, 0.2), (0.0, 1.0))
            )
            self.assertAlmostEqual(result.value, expected_value, delta=1e-12)
            self.assertEqual(set(result.texels), {(12, 12)})

    def test_per_tile_wrap_modes_are_independent(self):
        # Same atlas, same local coordinate: the boundary behaviour is
        # decided by the selected tile's own wrap mode.
        atlas = make_texture(4, 1, lambda i, j: float(i))
        tiles = {
            "rep": Tile(0, 0, 2, 1, WrapMode.REPEAT),
            "cla": Tile(2, 0, 2, 1, WrapMode.CLAMP),
        }
        zero_j = ((0.0, 0.0), (0.0, 0.0))
        e = math.exp(-8.0)
        # Local u = 0.5, radius-1 footprint: cross-shaped candidates
        # (0,0), (+-1,0), (0,+-1); the tile is 2x1, so rows +-1 also map
        # back onto row 0 with their own kernel weights.
        rep = filter_sample_atlas(atlas, tiles, "rep", 0.5, 0.5, zero_j)
        # repeat: col -1 -> local 1 (value 1); rows +-1 -> row 0 (value 0).
        # value = (0*1 + 1*e + 1*e + 0*e + 0*e) / (1 + 4e)
        self.assertAlmostEqual(
            rep.value, (2.0 * e) / (1.0 + 4.0 * e), delta=1e-12)
        cla = filter_sample_atlas(atlas, tiles, "cla", 0.5, 0.5, zero_j)
        # clamp: col -1 -> local 0 (value 2); rows +-1 -> row 0 (value 2);
        # col 1 -> local 1 (value 3).
        # value = (2*1 + 2*e + 3*e + 2*e + 2*e) / (1 + 4e)
        self.assertAlmostEqual(
            cla.value, (2.0 + 9.0 * e) / (1.0 + 4.0 * e), delta=1e-12)
        self.assertNotAlmostEqual(rep.value, cla.value, delta=0.1)


class TestWholeAtlasEquivalence(unittest.TestCase):
    """A tile covering the whole atlas must reproduce filter_sample."""

    def test_equivalent_to_legacy_call(self):
        atlas = pseudo_random_texture(12, 10)
        jacobian = ((0.8, -0.6), (0.6, 0.8))
        for wrap in (WrapMode.REPEAT, WrapMode.CLAMP):
            tiles = {"all": Tile(0, 0, 12, 10, wrap)}
            for u, v in ((0.2, 0.3), (11.8, 9.6), (5.5, 4.5), (-1.5, 2.5)):
                legacy = filter_sample(atlas, u, v, jacobian, wrap=wrap)
                tiled = filter_sample_atlas(atlas, tiles, "all", u, v, jacobian)
                self.assertEqual(tiled.count, legacy.count)
                self.assertEqual(tiled.texels, legacy.texels)
                self.assertEqual(tiled.value, legacy.value)
                self.assertEqual(tiled.weight_sum, legacy.weight_sum)


class TestAtlasBudget(unittest.TestCase):
    def test_budget_exceeded_on_atlas(self):
        tiles = {"t": Tile(0, 0, 16, 16, WrapMode.CLAMP)}
        with self.assertRaises(BudgetExceededError) as ctx:
            filter_sample_atlas(ATLAS16, tiles, "t", 8.0, 8.0,
                                ((32.0, 0.0), (0.0, 32.0)))
        self.assertGreater(ctx.exception.candidates, 4096)

    def test_custom_budget_on_atlas(self):
        tiles = {"t": Tile(0, 0, 16, 16, WrapMode.REPEAT)}
        jacobian = ((2.0, 0.0), (0.0, 2.0))
        result = filter_sample_atlas(ATLAS16, tiles, "t", 8.0, 8.0, jacobian)
        self.assertGreater(result.count, 0)
        with self.assertRaises(BudgetExceededError):
            filter_sample_atlas(ATLAS16, tiles, "t", 8.0, 8.0, jacobian,
                                budget=10)


class TestTileAndAtlasValidation(unittest.TestCase):
    def setUp(self):
        self.atlas = pseudo_random_texture(8, 8)
        self.tiles = {"t": Tile(0, 0, 4, 4, WrapMode.CLAMP)}
        self.jacobian = ((1.0, 0.0), (0.0, 1.0))

    def test_tile_field_validation(self):
        with self.assertRaises(InvalidInputError):
            Tile(-1, 0, 4, 4)
        with self.assertRaises(InvalidInputError):
            Tile(0, -1, 4, 4)
        with self.assertRaises(InvalidInputError):
            Tile(0, 0, 0, 4)
        with self.assertRaises(InvalidInputError):
            Tile(0, 0, 4, -2)
        with self.assertRaises(InvalidInputError):
            Tile(0.5, 0, 4, 4)
        with self.assertRaises(InvalidInputError):
            Tile(0, 0, 4, 4, "clamp")
        with self.assertRaises(InvalidInputError):
            Tile(True, 0, 4, 4)
        # Control: a valid Tile constructs fine and defaults to CLAMP.
        self.assertEqual(Tile(0, 0, 4, 4).wrap, WrapMode.CLAMP)

    def test_unknown_tile_id_rejected(self):
        with self.assertRaises(InvalidInputError) as ctx:
            filter_sample_atlas(self.atlas, self.tiles, "nope", 1.0, 1.0,
                                self.jacobian)
        self.assertIn("nope", str(ctx.exception))

    def test_unhashable_tile_id_rejected(self):
        with self.assertRaises(InvalidInputError):
            filter_sample_atlas(self.atlas, self.tiles, ["t"], 1.0, 1.0,
                                self.jacobian)

    def test_malformed_tiles_mapping_rejected(self):
        for bad in ({}, "tiles", [Tile(0, 0, 4, 4)],
                    {"t": (0, 0, 4, 4)}, {"t": None}):
            with self.assertRaises(InvalidInputError):
                filter_sample_atlas(self.atlas, bad, "t", 1.0, 1.0,
                                    self.jacobian)

    def test_tile_rect_outside_atlas_rejected(self):
        for bad_tile in (Tile(6, 0, 4, 4),   # 6+4 > 8 in x
                         Tile(0, 6, 4, 4),   # 6+4 > 8 in y
                         Tile(0, 0, 9, 1)):  # wider than the atlas
            with self.assertRaises(InvalidInputError):
                filter_sample_atlas(self.atlas, {"t": bad_tile}, "t",
                                    1.0, 1.0, self.jacobian)

    def test_non_finite_uv_rejected(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(InvalidInputError):
                filter_sample_atlas(self.atlas, self.tiles, "t", bad, 1.0,
                                    self.jacobian)
            with self.assertRaises(InvalidInputError):
                filter_sample_atlas(self.atlas, self.tiles, "t", 1.0, bad,
                                    self.jacobian)

    def test_malformed_atlas_and_jacobian_rejected(self):
        with self.assertRaises(InvalidInputError):
            filter_sample_atlas([[1.0, 2.0], [3.0]], self.tiles, "t",
                                0.5, 0.5, self.jacobian)
        with self.assertRaises(InvalidInputError):
            filter_sample_atlas(self.atlas, self.tiles, "t", 1.0, 1.0,
                                ((1.0,), (0.0, 1.0)))
        with self.assertRaises(InvalidInputError):
            filter_sample_atlas(self.atlas, self.tiles, "t", 1.0, 1.0,
                                self.jacobian, budget=0)


class TestAtlasResultContract(unittest.TestCase):
    def test_result_contract_holds_for_tiles(self):
        jacobian = ((1.2, 0.4), (-0.3, 0.9))
        for tile_id, tile in TILES16.items():
            result = filter_sample_atlas(ATLAS16, TILES16, tile_id,
                                         1.31, 0.87, jacobian)
            values = [ATLAS16[j][i] for (i, j) in result.texels]
            self.assertGreaterEqual(result.value, min(values) - 1e-15)
            self.assertLessEqual(result.value, max(values) + 1e-15)
            self.assertAlmostEqual(sum(result.weights), 1.0, delta=1e-12)
            self.assertEqual(result.count, len(result.weights))


if __name__ == "__main__":
    unittest.main()
