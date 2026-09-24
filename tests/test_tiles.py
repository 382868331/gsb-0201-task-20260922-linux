"""Tests for the tiled-atlas entry point (TileAtlas / filter_tile).

The reference here (``reference_tile``) is an independent re-implementation
of the footprint math written from scratch for these tests: it builds
``C``/``C^-1`` itself, scans a deliberately oversized box, and maps each
candidate center with modulus/clamp against the *selected tile's own*
dimensions.  It never imports or calls the module under test to produce an
expected value.

Small footprints (zero Jacobian -> radius-1 cross) are also checked against
fully hand-computed texel sets and weighted values.
"""

import math
import unittest

from ellipse_filter import (
    BudgetExceededError,
    InvalidInputError,
    TileAtlas,
    TileNotFoundError,
    WrapMode,
    filter_sample,
    filter_tile,
)


def make_texture(width, height, fn):
    return [[fn(i, j) for i in range(width)] for j in range(height)]


def reference_tile(tile, u, v, jacobian, wrap, box_radius=12):
    """Independent brute-force reference over a wide box for ONE tile.

    ``tile`` is a plain row-major list.  Out-of-range centers resolve
    against this tile's own width/height and nothing else.
    """
    height = len(tile)
    width = len(tile[0])
    (j00, j01), (j10, j11) = jacobian
    c00 = j00 * j00 + j01 * j01 + 0.25
    c01 = j00 * j10 + j01 * j11
    c11 = j10 * j10 + j11 * j11 + 0.25
    det = c00 * c11 - c01 * c01
    m00, m01, m11 = c11 / det, -c01 / det, c00 / det

    total = 0.0
    accum = 0.0
    count = 0
    texels = []
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
                    ti, tj = ix % width, jy % height
                else:
                    ti = min(max(ix, 0), width - 1)
                    tj = min(max(jy, 0), height - 1)
                total += w
                accum += w * tile[tj][ti]
                count += 1
                texels.append((ti, tj))
    return accum / total, total, count, texels


def zero_jac():
    return ((0.0, 0.0), (0.0, 0.0))


EDGE = math.exp(-8.0)


class TestTileAtlasConstruction(unittest.TestCase):
    def test_build_from_mapping_and_sequence(self):
        a = make_texture(3, 2, lambda i, j: float(i))
        b = make_texture(5, 4, lambda i, j: float(i + j))
        atlas_map = TileAtlas({"a": a, "b": b})
        atlas_seq = TileAtlas([("a", a), ("b", b)])
        self.assertEqual(atlas_map.tile_ids, ("a", "b"))
        self.assertEqual(atlas_seq.tile_ids, ("a", "b"))
        self.assertEqual(len(atlas_map), 2)
        self.assertIn("a", atlas_map)
        self.assertNotIn("zzz", atlas_map)
        self.assertEqual(atlas_map.size("a"), (3, 2))
        self.assertEqual(atlas_map.size("b"), (5, 4))

    def test_integer_and_enum_like_ids(self):
        t = make_texture(2, 2, lambda i, j: 0.0)
        atlas = TileAtlas({0: t, 7: t, "roof": t})
        self.assertEqual(atlas.size(0), (2, 2))
        self.assertEqual(atlas.size(7), (2, 2))
        self.assertEqual(atlas.size("roof"), (2, 2))

    def test_invalid_tile_aborts_whole_atlas(self):
        good = make_texture(2, 2, lambda i, j: 0.0)
        bad = [[1.0, 2.0], [3.0]]  # non-rectangular
        with self.assertRaises(InvalidInputError):
            TileAtlas({"good": good, "bad": bad})
        with self.assertRaises(InvalidInputError):
            TileAtlas([("bad", bad), ("good", good)])

    def test_duplicate_id_rejected(self):
        t = make_texture(2, 2, lambda i, j: 0.0)
        with self.assertRaises(InvalidInputError):
            TileAtlas([("x", t), ("x", t)])

    def test_bad_wrap_in_spec_rejected(self):
        t = make_texture(2, 2, lambda i, j: 0.0)
        with self.assertRaises(InvalidInputError):
            TileAtlas({"x": (t, "repeat")})
        with self.assertRaises(InvalidInputError):
            TileAtlas({"x": t}, default_wrap="clamp")

    def test_bool_and_unhashable_ids_rejected(self):
        t = make_texture(2, 2, lambda i, j: 0.0)
        with self.assertRaises(InvalidInputError):
            TileAtlas({True: t})
        with self.assertRaises(InvalidInputError):
            TileAtlas({None: t})
        with self.assertRaises(InvalidInputError):
            TileAtlas([([1, 2], t)])

    def test_atlas_is_immutable(self):
        atlas = TileAtlas({"a": make_texture(2, 2, lambda i, j: 0.0)})
        with self.assertRaises(AttributeError):
            atlas._tiles = {}
        with self.assertRaises(AttributeError):
            atlas.new_attr = 1

    def test_malformed_spec_rejected(self):
        t = make_texture(2, 2, lambda i, j: 0.0)
        with self.assertRaises(InvalidInputError):
            TileAtlas("not-an-atlas")
        with self.assertRaises(InvalidInputError):
            TileAtlas([("a", t, WrapMode.REPEAT, "extra")])
        with self.assertRaises(InvalidInputError):
            TileAtlas([("only-id",)])


class TestTileNotFound(unittest.TestCase):
    def setUp(self):
        self.atlas = TileAtlas({"a": make_texture(2, 2, lambda i, j: 0.0)})

    def test_unknown_id_raises_tile_not_found(self):
        with self.assertRaises(TileNotFoundError) as ctx:
            filter_tile(self.atlas, "ghost", 0.5, 0.5, zero_jac())
        err = ctx.exception
        self.assertEqual(err.tile_id, "ghost")
        self.assertEqual(err.available, ("a",))
        # It is still an invalid-input / ValueError, but distinct enough to
        # catch specifically.
        self.assertIsInstance(err, ValueError)
        self.assertIsInstance(err, InvalidInputError)

    def test_unknown_id_direct_resolve(self):
        with self.assertRaises(TileNotFoundError):
            self.atlas.size(42)
        self.assertNotIn(42, self.atlas)


class TestCornerAndNegativeUV(unittest.TestCase):
    def test_repeat_negative_uv_wraps_to_far_columns(self):
        # 4x3 tile, texel value = column index.  Sample at (0.5, 1.5) with
        # radius-1 cross.  Column -1 wraps to column 3; rows -1/2 wrap to
        # rows 2 and 0 respectively (height 3).
        tile = make_texture(4, 3, lambda i, j: float(i))
        atlas = TileAtlas({"t": tile})
        res = filter_tile(atlas, "t", 0.5, 1.5, zero_jac(),
                          wrap=WrapMode.REPEAT)
        # Cross centers (-1,1),(0,0),(0,1),(0,2),(1,1) map to:
        # (3,1),(0,0),(0,1),(0,2),(1,1).
        self.assertEqual(set(res.texels),
                         {(3, 1), (0, 0), (0, 1), (0, 2), (1, 1)})
        # Values: 3 (e), 0,0,0 (center+vertical), 1 (e).
        expected = (3.0 * EDGE + 1.0 * EDGE) / (1.0 + 4.0 * EDGE)
        self.assertAlmostEqual(res.value, expected, delta=1e-12)

    def test_clamp_negative_uv_pins_to_edges(self):
        tile = make_texture(4, 3, lambda i, j: float(i) + 10.0 * j)
        atlas = TileAtlas({"t": tile})
        res = filter_tile(atlas, "t", 0.5, 1.5, zero_jac(),
                          wrap=WrapMode.CLAMP)
        # (-1,1)->(0,1), (0,0)->(0,0), center (0,1), (0,2)->(0,2),
        # (1,1)->(1,1).  (0,1) appears twice (center + clamped column -1).
        self.assertEqual(sorted(res.texels),
                         [(0, 0), (0, 1), (0, 1), (0, 2), (1, 1)])
        # Texel values: (0,0)=0,(0,1)=10 x2,(0,2)=20,(1,1)=11.
        expected = (0.0 * EDGE + 20.0 * EDGE + 11.0 * EDGE
                    + 10.0 * (1.0 + EDGE)) / (1.0 + 4.0 * EDGE)
        self.assertAlmostEqual(res.value, expected, delta=1e-12)

    def test_corner_repeat_four_corners_join(self):
        # Sample at the (0.5, 0.5) corner of a 4x3 tile under repeat:
        # the cross wraps around all four edges.
        tile = make_texture(4, 3, lambda i, j: 10.0 * i + j)
        atlas = TileAtlas({"t": tile})
        res = filter_tile(atlas, "t", 0.5, 0.5, zero_jac(),
                          wrap=WrapMode.REPEAT)
        # centers (-1,0)->(3,0)=30 ; (0,-1)->(0,2)=2 ; center (0,0)=0 ;
        # (1,0)->(1,0)=10 ; (0,1)->(0,1)=1
        self.assertEqual(set(res.texels),
                         {(3, 0), (0, 2), (0, 0), (1, 0), (0, 1)})
        expected = (30.0 + 2.0 + 10.0 + 1.0) * EDGE / (1.0 + 4.0 * EDGE)
        self.assertAlmostEqual(res.value, expected, delta=1e-12)

    def test_corner_clamp_stacks_corner_texel(self):
        tile = make_texture(4, 3, lambda i, j: 10.0 * i + j)
        atlas = TileAtlas({"t": tile})
        res = filter_tile(atlas, "t", 0.5, 0.5, zero_jac(),
                          wrap=WrapMode.CLAMP)
        # (-1,-1) is not in the radius-1 cross (diagonal). Cross:
        # (-1,0)->(0,0); (0,-1)->(0,0); center (0,0); (1,0)->(1,0);
        # (0,1)->(0,1).  (0,0) thus appears 3 times.
        self.assertEqual(sorted(res.texels),
                         [(0, 0), (0, 0), (0, 0), (0, 1), (1, 0)])
        expected = (0.0 * (1.0 + 2.0 * EDGE) + 1.0 * EDGE + 10.0 * EDGE) \
            / (1.0 + 4.0 * EDGE)
        self.assertAlmostEqual(res.value, expected, delta=1e-12)

    def test_far_negative_and_large_uv_repeat(self):
        # UV far outside [0,w)x[0,h) still maps modulo the tile size.
        tile = make_texture(4, 4, lambda i, j: float(i))
        atlas = TileAtlas({"t": tile})
        res = filter_tile(atlas, "t", -7.5, 101.5, zero_jac(),
                          wrap=WrapMode.REPEAT)
        # -7.5 is a texel center (-8 + 0.5); column -8 % 4 = 0;
        # 101.5 -> row 101 % 4 = 1.  Center texel (0,1), value 0.
        self.assertEqual(res.texels[res.weights.index(max(res.weights))],
                         (0, 1))
        # Repeating a periodic value field column i: result must equal the
        # in-range equivalent at (0.5, 1.5).
        ref = filter_tile(atlas, "t", 0.5, 1.5, zero_jac(),
                          wrap=WrapMode.REPEAT)
        self.assertAlmostEqual(res.value, ref.value, delta=1e-12)
        self.assertEqual(res.texels, ref.texels)

    def test_far_uv_clamp_pins_to_nearest_edge(self):
        tile = make_texture(4, 4, lambda i, j: float(i) + 10.0 * j)
        atlas = TileAtlas({"t": tile})
        res = filter_tile(atlas, "t", 100.5, -100.5, zero_jac(),
                          wrap=WrapMode.CLAMP)
        # The whole radius-1 cross clamps onto the (3,0) corner texel.
        self.assertTrue(all(t == (3, 0) for t in res.texels))
        self.assertAlmostEqual(res.value, tile[0][3], delta=1e-12)


class TestDifferentSizedTiles(unittest.TestCase):
    def test_tiles_of_different_sizes(self):
        sizes = {"small": (3, 2), "wide": (7, 1), "tall": (1, 5),
                 "big": (16, 9)}
        tiles = {}
        for name, (w, h) in sizes.items():
            tiles[name] = make_texture(w, h, lambda i, j, n=name: i + j)
        atlas = TileAtlas(tiles)
        for name, (w, h) in sizes.items():
            self.assertEqual(atlas.size(name), (w, h))
            res = filter_tile(atlas, name, w / 2, h / 2,
                              ((0.7, 0.2), (-0.3, 0.6)),
                              wrap=WrapMode.REPEAT)
            self.assertEqual(res.tile_id, name)
            for (i, j) in res.texels:
                self.assertTrue(0 <= i < w and 0 <= j < h)

    def test_one_by_one_tile(self):
        atlas = TileAtlas({"dot": [[7.0]]})
        res = filter_tile(atlas, "dot", 0.5, 0.5, zero_jac(),
                          wrap=WrapMode.REPEAT)
        self.assertTrue(all(t == (0, 0) for t in res.texels))
        self.assertAlmostEqual(res.value, 7.0, delta=1e-12)
        # Same under clamp and for wild UV.
        res2 = filter_tile(atlas, "dot", -50.25, 1e3, zero_jac(),
                           wrap=WrapMode.CLAMP)
        self.assertAlmostEqual(res2.value, 7.0, delta=1e-12)


class TestTileIsolation(unittest.TestCase):
    def test_footprint_never_leaks_other_tiles(self):
        # Two tiles with disjoint value sets; a footprint centered on a tile
        # edge with repeat/clamp must only ever return that tile's indices.
        a = make_texture(4, 4, lambda i, j: 0.0)     # all zero
        b = make_texture(4, 4, lambda i, j: 100.0)  # all 100
        atlas = TileAtlas({"a": a, "b": b})
        jac = ((1.4, 0.0), (0.0, 1.4))
        for wrap in (WrapMode.REPEAT, WrapMode.CLAMP):
            for u, v in ((0.5, 0.5), (3.5, 3.5), (-2.5, 6.5)):
                res = filter_tile(atlas, "a", u, v, jac, wrap=wrap)
                self.assertEqual(res.tile_id, "a")
                self.assertAlmostEqual(res.value, 0.0, delta=1e-12)
                res_b = filter_tile(atlas, "b", u, v, jac, wrap=wrap)
                self.assertAlmostEqual(res_b.value, 100.0, delta=1e-12)

    def test_different_sizes_same_uv_resolve_independently(self):
        # Same UV, zero footprint at a texel center: column -1 must wrap to
        # width-1 of EACH tile (3 for the 4-wide tile, 6 for the 7-wide).
        a = make_texture(4, 4, lambda i, j: float(i))
        b = make_texture(7, 2, lambda i, j: float(i))
        atlas = TileAtlas({"a": a, "b": b})
        ra = filter_tile(atlas, "a", 0.5, 1.5, zero_jac(),
                         wrap=WrapMode.REPEAT)
        rb = filter_tile(atlas, "b", 0.5, 0.5, zero_jac(),
                         wrap=WrapMode.REPEAT)
        self.assertIn((3, 1), ra.texels)
        self.assertNotIn((6, 1), ra.texels)
        self.assertIn((6, 0), rb.texels)
        self.assertNotIn((3, 0),
                         [t for t in rb.texels if t != (0, 0)])

    def test_tile_result_matches_independent_reference(self):
        tiles = {
            "a": make_texture(8, 5, lambda i, j: math.sin(i * 3.1 + j)),
            "b": make_texture(3, 6, lambda i, j: (i * 7 + j * 13) % 5),
        }
        atlas = TileAtlas(tiles)
        jac = ((1.1, 0.5), (-0.4, 0.9))
        cases = [("a", 0.2, 0.3, WrapMode.REPEAT),
                 ("a", 7.8, 4.7, WrapMode.CLAMP),
                 ("b", -2.3, 9.1, WrapMode.REPEAT),
                 ("b", 1.4, 2.6, WrapMode.CLAMP)]
        for name, u, v, wrap in cases:
            w, h = atlas.size(name)
            exp_val, exp_sum, exp_count, exp_tex = reference_tile(
                tiles[name], u, v, jac, wrap
            )
            res = filter_tile(atlas, name, u, v, jac, wrap=wrap)
            self.assertEqual(res.count, exp_count, (name, u, v))
            self.assertAlmostEqual(res.value, exp_val, delta=1e-9)
            self.assertAlmostEqual(res.weight_sum, exp_sum, delta=1e-9)
            self.assertEqual(res.texels, tuple(exp_tex))
            for (i, j) in res.texels:
                self.assertTrue(0 <= i < w and 0 <= j < h)


class TestPerTileWrapDefault(unittest.TestCase):
    def setUp(self):
        # 4x4 tile sampled at (0.5, 1.5) with the radius-1 cross:
        # column -1 wraps to 3 under repeat, pins to 0 under clamp; the
        # vertical arms (rows 0,2) stay in-range either way.
        self.tile = make_texture(4, 4, lambda i, j: float(i))
        self.u, self.v = 0.5, 1.5

    def test_tile_own_wrap_used_by_default(self):
        atlas = TileAtlas({"r": (self.tile, WrapMode.REPEAT),
                           "c": (self.tile, WrapMode.CLAMP)})
        self.assertIs(atlas.wrap_for("r"), WrapMode.REPEAT)
        self.assertIs(atlas.wrap_for("c"), WrapMode.CLAMP)
        rr = filter_tile(atlas, "r", self.u, self.v, zero_jac())
        rc = filter_tile(atlas, "c", self.u, self.v, zero_jac())
        self.assertIn((3, 1), rr.texels)
        self.assertNotIn((3, 1), rc.texels)
        # Clamp: center column 0 receives both the center and clamped -1.
        self.assertEqual(rc.texels.count((0, 1)), 2)
        # Repeat: exactly one center-column texel at row 1.
        self.assertEqual(rr.texels.count((0, 1)), 1)

    def test_atlas_default_wrap(self):
        atlas = TileAtlas({"t": self.tile},
                          default_wrap=WrapMode.CLAMP)
        self.assertIs(atlas.wrap_for("t"), WrapMode.CLAMP)
        res = filter_tile(atlas, "t", self.u, self.v, zero_jac())
        self.assertEqual(res.texels.count((0, 1)), 2)
        self.assertNotIn((3, 1), res.texels)

    def test_explicit_call_wrap_overrides(self):
        atlas = TileAtlas({"t": (self.tile, WrapMode.CLAMP)},
                          default_wrap=WrapMode.CLAMP)
        res = filter_tile(atlas, "t", self.u, self.v, zero_jac(),
                          wrap=WrapMode.REPEAT)
        self.assertIn((3, 1), res.texels)
        self.assertEqual(res.texels.count((0, 1)), 1)

    def test_global_default_is_repeat(self):
        atlas = TileAtlas({"t": self.tile})
        self.assertIsNone(atlas.wrap_for("t"))
        res = filter_tile(atlas, "t", self.u, self.v, zero_jac())
        self.assertIn((3, 1), res.texels)


class TestTileResultContract(unittest.TestCase):
    def test_result_carries_tile_id_and_is_consistent(self):
        atlas = TileAtlas({"k": make_texture(6, 6,
                         lambda i, j: (i * j) % 3)})
        res = filter_tile(atlas, "k", 2.3, 4.1,
                          ((1.0, 0.3), (-0.2, 0.8)))
        self.assertEqual(res.tile_id, "k")
        self.assertEqual(res.count, len(res.weights))
        self.assertEqual(res.count, len(res.texels))
        self.assertAlmostEqual(sum(res.weights), 1.0, delta=1e-12)

    def test_tile_matches_legacy_entry_on_same_grid(self):
        # filter_tile over a one-tile atlas must reproduce filter_sample.
        tile = make_texture(9, 7, lambda i, j: math.sin(i * 2.3 + j * 0.7))
        atlas = TileAtlas({"only": tile})
        jac = ((1.2, -0.4), (0.5, 0.9))
        for wrap in (WrapMode.REPEAT, WrapMode.CLAMP):
            old = filter_sample(tile, 3.7, 2.2, jac, wrap=wrap)
            new = filter_tile(atlas, "only", 3.7, 2.2, jac, wrap=wrap)
            self.assertEqual(new.value, old.value)
            self.assertEqual(new.weights, old.weights)
            self.assertEqual(new.texels, old.texels)
            self.assertEqual(new.count, old.count)
            self.assertIsNone(old.tile_id)
            self.assertEqual(new.tile_id, "only")


class TestTileBudgetAndValidation(unittest.TestCase):
    def test_budget_exceeded_in_tile(self):
        atlas = TileAtlas({"big": make_texture(64, 64, lambda i, j: 0.0)})
        with self.assertRaises(BudgetExceededError) as ctx:
            filter_tile(atlas, "big", 32.0, 32.0,
                        ((32.0, 0.0), (0.0, 32.0)))
        self.assertEqual(ctx.exception.budget, 4096)

    def test_custom_budget_tile(self):
        atlas = TileAtlas({"t": make_texture(16, 16, lambda i, j: 0.0)})
        with self.assertRaises(BudgetExceededError):
            filter_tile(atlas, "t", 8.0, 8.0,
                        ((2.0, 0.0), (0.0, 2.0)), budget=10)

    def test_bad_arguments_to_filter_tile(self):
        good_atlas = TileAtlas({"t": make_texture(4, 4, lambda i, j: 0.0)})
        jac = zero_jac()
        # atlas wrong type
        with self.assertRaises(InvalidInputError):
            filter_tile([[1.0]], "t", 0.5, 0.5, jac)
        # non-finite uv / jacobian
        with self.assertRaises(InvalidInputError):
            filter_tile(good_atlas, "t", math.nan, 0.5, jac)
        with self.assertRaises(InvalidInputError):
            filter_tile(good_atlas, "t", 0.5, math.inf, jac)
        with self.assertRaises(InvalidInputError):
            filter_tile(good_atlas, "t", 0.5, 0.5,
                        ((1.0, 0.0), (0.0, True)))
        # bad wrap / budget / bool id
        with self.assertRaises(InvalidInputError):
            filter_tile(good_atlas, "t", 0.5, 0.5, jac, wrap="repeat")
        with self.assertRaises(InvalidInputError):
            filter_tile(good_atlas, "t", 0.5, 0.5, jac, budget=0)
        with self.assertRaises(InvalidInputError):
            filter_tile(good_atlas, "t", 0.5, 0.5, jac, budget=True)
        with self.assertRaises(InvalidInputError):
            filter_tile(good_atlas, True, 0.5, 0.5, jac)


if __name__ == "__main__":
    unittest.main()
