"""Tests for ripmap.

The reference here (``reference_box_average``) is an independent
re-implementation: it integrates the original texture over the
axis-aligned footprint rectangle with exact per-texel coverage weights
(repeat wrapping), never importing the module under test for expected
values.  The ripmap result is only an *approximation* of that integral
(bilinear blend of nearest reads from area-averaged levels), so the
comparison uses a documented, loose tolerance -- see README.
"""

import math
import unittest

from ellipse_filter import InvalidInputError, WrapMode
from ripmap import build_ripmap, sample_ripmap


def make_texture(width, height, fn):
    return [[fn(i, j) for i in range(width)] for j in range(height)]


def pseudo_random_texture(width, height):
    def value(i, j):
        x = math.sin(i * 12.9898 + j * 78.233) * 43758.5453
        return x - math.floor(x)
    return make_texture(width, height, value)


def reference_box_average(texture, u, v, radius_u, radius_v):
    """Independent area integral of the footprint box on the source image.

    Averages texel values over [u-radius_u, u+radius_u] x
    [v-radius_v, v+radius_v], weighting each texel cell by its exact
    overlap area with the box (repeat addressing).
    """
    height = len(texture)
    width = len(texture[0])
    u0, u1 = u - radius_u, u + radius_u
    v0, v1 = v - radius_v, v + radius_v
    total = 0.0
    accum = 0.0
    for jy in range(math.floor(v0), math.ceil(v1)):
        oy = max(0.0, min(v1, jy + 1.0) - max(v0, float(jy)))
        if oy <= 0.0:
            continue
        for ix in range(math.floor(u0), math.ceil(u1)):
            ox = max(0.0, min(u1, ix + 1.0) - max(u0, float(ix)))
            if ox <= 0.0:
                continue
            coverage = ox * oy
            accum += coverage * texture[jy % height][ix % width]
            total += coverage
    return accum / total


class TestRipmapConstruction(unittest.TestCase):
    def test_level_sizes_recorded_power_of_two(self):
        rip = build_ripmap(make_texture(8, 4, lambda i, j: 0.0))
        self.assertEqual(
            rip.level_sizes(),
            (
                ((8, 4), (8, 2), (8, 1)),
                ((4, 4), (4, 2), (4, 1)),
                ((2, 4), (2, 2), (2, 1)),
                ((1, 4), (1, 2), (1, 1)),
            ),
        )

    def test_odd_edges_use_coverage_weights_not_dropped(self):
        # 5x3: last column/row must survive with weight 1, not be dropped.
        texture = [
            [1.0, 2.0, 3.0, 4.0, 100.0],
            [5.0, 6.0, 7.0, 8.0, 200.0],
            [9.0, 10.0, 11.0, 12.0, 300.0],
        ]
        rip = build_ripmap(texture)
        level10 = rip.level(1, 0)
        self.assertEqual((level10.width, level10.height), (3, 3))
        self.assertEqual(level10.data[0], (1.5, 3.5, 100.0))
        self.assertEqual(level10.data[2], (9.5, 11.5, 300.0))
        level01 = rip.level(0, 1)
        self.assertEqual((level01.width, level01.height), (5, 2))
        self.assertEqual(level01.data[0], (3.0, 4.0, 5.0, 6.0, 150.0))
        self.assertEqual(level01.data[1], (9.0, 10.0, 11.0, 12.0, 300.0))
        level11 = rip.level(1, 1)
        self.assertEqual((level11.width, level11.height), (3, 2))
        self.assertEqual(level11.data[0], (3.5, 5.5, 150.0))
        self.assertEqual(level11.data[1], (9.5, 11.5, 300.0))

    def test_single_pixel_dimensions(self):
        for width, height in ((1, 1), (1, 7), (9, 1)):
            rip = build_ripmap(make_texture(width, height, lambda i, j: i + j))
            sizes = rip.level_sizes()
            self.assertEqual(len(sizes), max(1, (width - 1).bit_length() + 1))
            self.assertEqual(len(sizes[0]), max(1, (height - 1).bit_length() + 1))
            result = sample_ripmap(
                rip, 0.5 * width, 0.5 * height, ((4.0, 0.0), (0.0, 4.0))
            )
            self.assertTrue(math.isfinite(result.value))
            self.assertEqual(len(result.samples), 4)


class TestConstantConservation(unittest.TestCase):
    def test_constant_texture_any_footprint(self):
        texture = make_texture(13, 9, lambda i, j: -2.5)
        rip = build_ripmap(texture)
        for jacobian in (
            ((0.0, 0.0), (0.0, 0.0)),
            ((1.0, 0.0), (0.0, 1.0)),
            ((6.0, 0.0), (0.0, 0.5)),
            ((3.0, 2.0), (-1.0, 4.0)),
        ):
            for wrap in (WrapMode.REPEAT, WrapMode.CLAMP):
                result = sample_ripmap(rip, 5.3, 4.1, jacobian, wrap=wrap)
                self.assertAlmostEqual(result.value, -2.5, delta=1e-12)
                self.assertAlmostEqual(
                    sum(s.weight for s in result.samples), 1.0, delta=1e-12
                )


class TestLevelSelection(unittest.TestCase):
    def test_anisotropic_lods_select_different_axis_levels(self):
        # Long horizontal footprint: radius_u = 2*sqrt(16+0.25) ~ 8.06,
        # radius_v = 1 -> lod_u ~ 3.01, lod_v = 0.
        rip = build_ripmap(make_texture(16, 16, lambda i, j: 0.0))
        result = sample_ripmap(rip, 8.0, 8.0, ((4.0, 0.0), (0.0, 0.0)))
        self.assertAlmostEqual(result.lod_v, 0.0, delta=1e-12)
        self.assertGreater(result.lod_u, 2.9)
        self.assertLess(result.lod_u, 3.1)
        levels = sorted(set(s.level for s in result.samples))
        self.assertEqual(levels, [(3, 0), (3, 1), (4, 0), (4, 1)])
        for sample in result.samples:
            self.assertEqual(
                sample.size,
                rip.level_sizes()[sample.level[0]][sample.level[1]],
            )

    def test_degenerate_jacobian_is_explicit_not_an_error(self):
        texture = make_texture(8, 8, lambda i, j: float(i * 8 + j))
        rip = build_ripmap(texture)
        result = sample_ripmap(rip, 3.2, 5.7, ((0.0, 0.0), (0.0, 0.0)))
        self.assertTrue(result.degenerate)
        self.assertEqual((result.lod_u, result.lod_v), (0.0, 0.0))
        # LOD (0,0): plain nearest-texel read of the base level.
        self.assertEqual(result.value, texture[5][3])
        self.assertTrue(
            all(s.weight == 0.0 or s.level == (0, 0) for s in result.samples)
        )

    def test_wrap_modes_record_in_level_texels(self):
        texture = make_texture(8, 8, lambda i, j: float(i + 10 * j))
        rip = build_ripmap(texture)
        repeat = sample_ripmap(rip, 9.5, -1.5, ((0.0, 0.0), (0.0, 0.0)),
                               wrap=WrapMode.REPEAT)
        self.assertEqual(repeat.samples[0].texel, (1, 6))
        clamp = sample_ripmap(rip, 9.5, -1.5, ((0.0, 0.0), (0.0, 0.0)),
                              wrap=WrapMode.CLAMP)
        self.assertEqual(clamp.samples[0].texel, (7, 0))


class TestAgainstIndependentAreaIntegral(unittest.TestCase):
    """Ripmap output vs. exact box area average of the original image.

    The ripmap is an approximation (nearest-in-level sampling plus a
    bilinear LOD blend), so we assert a documented loose bound, not
    exact equality.
    """

    def test_horizontal_highfreq_vertical_smooth(self):
        # Alternating columns (high frequency in u), smooth ramp in v.
        texture = make_texture(
            32, 32, lambda i, j: (i % 2) + 0.3 * (j / 31.0)
        )
        rip = build_ripmap(texture)
        # Wide in u, narrow in v: lod_u ~ 4, lod_v = 0.
        jacobian = ((8.0, 0.0), (0.0, 0.0))
        u, v = 10.3, 20.6
        result = sample_ripmap(rip, u, v, jacobian)
        reference = reference_box_average(
            texture, u, v, result.radius_u, result.radius_v
        )
        # The alternating column pattern averages to ~0.5 at level >= 1;
        # the v ramp is resolved at level 0, so the approximation is tight.
        self.assertAlmostEqual(result.value, reference, delta=0.05)
        self.assertAlmostEqual(
            result.value, 0.5 + 0.3 * (20 / 31.0), delta=0.05
        )

    def test_pseudo_random_axis_aligned(self):
        texture = pseudo_random_texture(32, 32)
        rip = build_ripmap(texture)
        jacobian = ((6.0, 0.0), (0.0, 2.0))
        u, v = 15.25, 12.75
        result = sample_ripmap(rip, u, v, jacobian)
        reference = reference_box_average(
            texture, u, v, result.radius_u, result.radius_v
        )
        # Approximation bound: nearest-in-level sampling of area-averaged
        # levels vs. the exact box average.  Not exact; see README.
        self.assertAlmostEqual(result.value, reference, delta=0.15)

    def test_rotated_footprint_is_only_approximate(self):
        # A rotated ellipse is NOT integrated exactly: the ripmap only
        # sees the axis-aligned extents of C.  We only require the result
        # to stay within the texture's value range.
        texture = pseudo_random_texture(16, 16)
        rip = build_ripmap(texture)
        result = sample_ripmap(rip, 8.0, 8.0, ((3.0, 2.0), (-2.0, 3.0)))
        self.assertGreaterEqual(result.value, 0.0)
        self.assertLessEqual(result.value, 1.0)


class TestInvalidInput(unittest.TestCase):
    def setUp(self):
        self.rip = build_ripmap(make_texture(4, 4, lambda i, j: 0.0))

    def test_non_ripmap_rejected(self):
        with self.assertRaises(InvalidInputError):
            sample_ripmap([[0.0]], 0.5, 0.5, ((0.0, 0.0), (0.0, 0.0)))

    def test_nan_and_inf_rejected(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(InvalidInputError):
                sample_ripmap(self.rip, bad, 0.5, ((0.0, 0.0), (0.0, 0.0)))
            with self.assertRaises(InvalidInputError):
                sample_ripmap(self.rip, 0.5, bad, ((0.0, 0.0), (0.0, 0.0)))

    def test_bool_rejected(self):
        with self.assertRaises(InvalidInputError):
            sample_ripmap(self.rip, True, 0.5, ((0.0, 0.0), (0.0, 0.0)))

    def test_bad_jacobian_rejected(self):
        with self.assertRaises(InvalidInputError):
            sample_ripmap(self.rip, 0.5, 0.5, ((0.0, 0.0),))
        with self.assertRaises(InvalidInputError):
            sample_ripmap(self.rip, 0.5, 0.5,
                          ((math.nan, 0.0), (0.0, 0.0)))

    def test_bad_wrap_rejected(self):
        with self.assertRaises(InvalidInputError):
            sample_ripmap(self.rip, 0.5, 0.5,
                          ((0.0, 0.0), (0.0, 0.0)), wrap="repeat")

    def test_build_rejects_bad_texture(self):
        with self.assertRaises(InvalidInputError):
            build_ripmap([])
        with self.assertRaises(InvalidInputError):
            build_ripmap([[1.0], [1.0, 2.0]])
        with self.assertRaises(InvalidInputError):
            build_ripmap([[math.inf]])


if __name__ == "__main__":
    unittest.main()
