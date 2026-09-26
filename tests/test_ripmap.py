"""Tests for ripmap.py.

The reference here (reference_box_average) is an independent exact area
integration of the original texture over the axis-aligned footprint box
[u-su, u+su] x [v-sv, v+sv] with repeat addressing, computed by
per-texel coverage clipping.  It never imports or calls the module under
test for expected values.
"""

import math
import unittest

from ellipse_filter import InvalidInputError, WrapMode
from ripmap import build_ripmap, ripmap_sample


def reference_box_average(texture, u, v, su, sv):
    """Exact area average of texture over the box around (u, v).

    Repeat addressing: texel (i, j) covers the unit square
    [i, i+1] x [j, j+1] and maps to stored texel (i mod w, j mod h).
    The box is clipped against every texel cell it overlaps and the
    covered area is used as the weight.
    """
    height = len(texture)
    width = len(texture[0])
    x0, x1 = u - su, u + su
    y0, y1 = v - sv, v + sv
    total_area = 0.0
    accum = 0.0
    for j in range(math.floor(y0), math.ceil(y1)):
        oy = min(y1, j + 1.0) - max(y0, float(j))
        if oy <= 0.0:
            continue
        tj = j % height
        for i in range(math.floor(x0), math.ceil(x1)):
            ox = min(x1, i + 1.0) - max(x0, float(i))
            if ox <= 0.0:
                continue
            area = ox * oy
            total_area += area
            accum += area * texture[tj][i % width]
    return accum / total_area


def constant_texture(width, height, value):
    return [[value] * width for _ in range(height)]


def horizontal_stripes_texture(width, height):
    """High frequency along u (alternating columns), smooth along v."""
    return [
        [(i % 2) * (j / (height - 1)) for i in range(width)]
        for j in range(height)
    ]


class TestBuild(unittest.TestCase):
    def test_level_sizes_recorded(self):
        rm = build_ripmap(constant_texture(7, 5, 1.0))
        self.assertEqual(rm.levels_u, 4)
        self.assertEqual(rm.levels_v, 4)
        self.assertEqual(
            rm.all_sizes(),
            (
                ((7, 5), (4, 5), (2, 5), (1, 5)),
                ((7, 3), (4, 3), (2, 3), (1, 3)),
                ((7, 2), (4, 2), (2, 2), (1, 2)),
                ((7, 1), (4, 1), (2, 1), (1, 1)),
            ),
        )

    def test_odd_width_coverage_weights(self):
        # Row [1, 2, 3, 4, 5]: halving must give [(1+2)/2, (3+4)/2, 5],
        # i.e. the last output texel keeps the uncovered odd column with
        # weight 1 instead of dropping it.
        rm = build_ripmap([[1.0, 2.0, 3.0, 4.0, 5.0]])
        self.assertEqual(rm.size(1, 0), (3, 1))
        self.assertEqual(rm.image(1, 0)[0], (1.5, 3.5, 5.0))
        self.assertEqual(rm.image(2, 0)[0], (2.5, 5.0))
        self.assertEqual(rm.image(3, 0)[0], (3.75,))

    def test_odd_height_coverage_weights(self):
        rm = build_ripmap([[1.0], [2.0], [3.0], [4.0], [5.0]])
        self.assertEqual(rm.size(0, 1), (1, 3))
        self.assertEqual([r[0] for r in rm.image(0, 1)], [1.5, 3.5, 5.0])

    def test_constant_levels_stay_constant(self):
        rm = build_ripmap(constant_texture(9, 6, 2.5))
        for lv in range(rm.levels_v):
            for lu in range(rm.levels_u):
                for row in rm.image(lu, lv):
                    for value in row:
                        self.assertAlmostEqual(value, 2.5, places=12)

    def test_invalid_texture_rejected(self):
        with self.assertRaises(InvalidInputError):
            build_ripmap([[1.0, 2.0], [3.0]])
        with self.assertRaises(InvalidInputError):
            build_ripmap([])
        with self.assertRaises(InvalidInputError):
            build_ripmap([[math.nan]])
        with self.assertRaises(InvalidInputError):
            build_ripmap([[True]])


class TestSample(unittest.TestCase):
    def test_constant_texture_conserved(self):
        rm = build_ripmap(constant_texture(16, 16, 0.75))
        for jacobian in (
            ((4.0, 0.0), (0.0, 2.0)),
            ((0.0, 0.0), (0.0, 0.0)),
            ((8.0, 1.0), (0.5, 8.0)),
        ):
            res = ripmap_sample(rm, 5.3, 7.7, jacobian)
            self.assertAlmostEqual(res.value, 0.75, places=12)

    def test_corner_weights_sum_to_one(self):
        rm = build_ripmap(constant_texture(16, 16, 1.0))
        res = ripmap_sample(rm, 5.3, 7.7, ((3.2, 0.0), (0.0, 1.4)))
        self.assertAlmostEqual(
            sum(c.corner_weight for c in res.corners), 1.0, places=12
        )
        for corner in res.corners:
            self.assertAlmostEqual(
                sum(corner.texel_weights), 1.0, places=12
            )
        self.assertEqual(res.count, sum(len(c.texels) for c in res.corners))

    def test_anisotropic_level_selection_recorded(self):
        rm = build_ripmap(constant_texture(32, 32, 1.0))
        # Footprint ~8 texels wide along u, ~1 along v.  The 0.25 * I
        # regularization shifts the continuous level slightly upward.
        # v scale 0.5 keeps lambda_v below zero, so it clamps to level 0.
        res = ripmap_sample(rm, 16.0, 16.0, ((8.0, 0.0), (0.0, 0.5)))
        self.assertAlmostEqual(
            res.level_u, math.log2(math.sqrt(8.0 ** 2 + 0.25)), places=12
        )
        self.assertEqual(res.level_v, 0.0)
        levels = {(c.level_u, c.level_v) for c in res.corners}
        self.assertEqual(levels, {(3, 0), (4, 0)})

    def test_horizontal_stripes_match_box_reference(self):
        # High frequency along u, smooth along v: a wide-u footprint must
        # average the stripes out, close to the exact box integral.
        width, height = 32, 16
        texture = horizontal_stripes_texture(width, height)
        rm = build_ripmap(texture)
        su, sv = 8.0, 1.0
        jacobian = ((su, 0.0), (0.0, sv))
        worst = 0.0
        for k in range(6):
            u = 3.0 + 4.1 * k
            v = 2.0 + 2.3 * k
            res = ripmap_sample(rm, u, v, jacobian)
            ref = reference_box_average(texture, u, v, su, sv)
            worst = max(worst, abs(res.value - ref))
        self.assertLess(worst, 0.06)

    def test_isotropic_moderate_matches_box_reference(self):
        width, height = 16, 16
        texture = [
            [math.sin(i * 0.7) * math.cos(j * 0.5) for i in range(width)]
            for j in range(height)
        ]
        rm = build_ripmap(texture)
        su = sv = 2.0
        jacobian = ((su, 0.0), (0.0, sv))
        # The ripmap tent-shaped prefilter is only an approximation of the
        # exact box integral, hence the looser tolerance here.
        for u, v in ((4.5, 4.5), (9.25, 6.75), (13.0, 12.5)):
            res = ripmap_sample(rm, u, v, jacobian)
            ref = reference_box_average(texture, u, v, su, sv)
            self.assertAlmostEqual(res.value, ref, delta=0.15)

    def test_single_pixel_width(self):
        texture = [[float(j)] for j in range(8)]
        rm = build_ripmap(texture)
        self.assertEqual(rm.levels_u, 1)
        res = ripmap_sample(rm, 0.5, 4.0, ((3.0, 0.0), (0.0, 2.0)))
        self.assertEqual(res.level_u, 0.0)
        ref = reference_box_average(texture, 0.5, 4.0, 3.0, 2.0)
        self.assertAlmostEqual(res.value, ref, delta=0.05)

    def test_single_pixel_height(self):
        texture = [[float(i) for i in range(8)]]
        rm = build_ripmap(texture)
        self.assertEqual(rm.levels_v, 1)
        res = ripmap_sample(rm, 4.0, 0.5, ((2.0, 0.0), (0.0, 3.0)))
        self.assertEqual(res.level_v, 0.0)
        ref = reference_box_average(texture, 4.0, 0.5, 2.0, 3.0)
        self.assertAlmostEqual(res.value, ref, delta=0.05)

    def test_single_texel_texture(self):
        rm = build_ripmap([[4.25]])
        res = ripmap_sample(rm, 0.5, 0.5, ((10.0, 0.0), (0.0, 10.0)))
        self.assertAlmostEqual(res.value, 4.25, places=12)

    def test_degenerate_jacobian_clamps_to_level_zero(self):
        texture = [[float(i + 8 * j) for i in range(8)] for j in range(8)]
        rm = build_ripmap(texture)
        res = ripmap_sample(rm, 3.5, 2.5, ((0.0, 0.0), (0.0, 0.0)))
        self.assertEqual(res.level_u, 0.0)
        self.assertEqual(res.level_v, 0.0)
        # Level 0 bilinear at an exact texel center is that texel.
        self.assertAlmostEqual(res.value, 3.0 + 8 * 2.0, places=12)

    def test_clamp_wrap_mode(self):
        texture = [[float(i) for i in range(4)] for _ in range(4)]
        rm = build_ripmap(texture)
        res = ripmap_sample(
            rm, -10.0, 1.5, ((0.0, 0.0), (0.0, 0.0)), wrap=WrapMode.CLAMP
        )
        self.assertAlmostEqual(res.value, 0.0, places=12)
        res = ripmap_sample(
            rm, 99.0, 1.5, ((0.0, 0.0), (0.0, 0.0)), wrap=WrapMode.CLAMP
        )
        self.assertAlmostEqual(res.value, 3.0, places=12)

    def test_repeat_wraps_across_seam(self):
        texture = [[1.0, 0.0]]
        rm = build_ripmap(texture)
        # u = 1.0 sits exactly on the boundary between texels 0 and 1.
        res = ripmap_sample(rm, 1.0, 0.5, ((0.0, 0.0), (0.0, 0.0)))
        self.assertAlmostEqual(res.value, 0.5, places=12)
        # u = 1.9 blends texel 1 with texel 0 across the wrap seam.
        res = ripmap_sample(rm, 1.9, 0.5, ((0.0, 0.0), (0.0, 0.0)))
        self.assertAlmostEqual(res.value, 0.4, places=12)
        corner = res.corners[0]
        self.assertEqual(set(corner.texels), {(1, 0), (0, 0)})

    def test_invalid_inputs_rejected(self):
        rm = build_ripmap(constant_texture(4, 4, 1.0))
        with self.assertRaises(InvalidInputError):
            ripmap_sample(rm, math.nan, 1.0, ((1.0, 0.0), (0.0, 1.0)))
        with self.assertRaises(InvalidInputError):
            ripmap_sample(rm, math.inf, 1.0, ((1.0, 0.0), (0.0, 1.0)))
        with self.assertRaises(InvalidInputError):
            ripmap_sample(rm, True, 1.0, ((1.0, 0.0), (0.0, 1.0)))
        with self.assertRaises(InvalidInputError):
            ripmap_sample(rm, 1.0, 1.0, ((1.0, 0.0),))
        with self.assertRaises(InvalidInputError):
            ripmap_sample(
                rm, 1.0, 1.0, ((1.0, 0.0), (0.0, 1.0)), wrap="repeat"
            )
        with self.assertRaises(InvalidInputError):
            ripmap_sample([[1.0]], 1.0, 1.0, ((1.0, 0.0), (0.0, 1.0)))

    def test_overflowing_jacobian_selects_coarsest_level(self):
        rm = build_ripmap(constant_texture(8, 8, 1.0))
        res = ripmap_sample(rm, 4.0, 4.0, ((1e300, 0.0), (0.0, 1.0)))
        self.assertEqual(res.level_u, float(rm.levels_u - 1))
        self.assertAlmostEqual(res.value, 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
