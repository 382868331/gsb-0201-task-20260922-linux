"""Tests for ellipse_filter.

The brute-force reference in this file (``reference_filter``) is an
independent re-implementation: it computes C and C^-1 in closed form,
scans a deliberately oversized ("wide") box instead of the tight bounding
box, and accumulates in its own order.  It never imports or calls the
module under test for expected values.
"""

import math
import unittest

from ellipse_filter import (
    AlphaMode,
    BudgetExceededError,
    DEFAULT_BUDGET,
    FilterResult,
    InvalidInputError,
    WrapMode,
    filter_sample,
)


def make_texture(width, height, fn):
    return [[fn(i, j) for i in range(width)] for j in range(height)]


def make_rgba_texture(width, height, fn):
    return [[tuple(fn(i, j)) for i in range(width)] for j in range(height)]


def pseudo_random_texture(width, height):
    """Deterministic non-constant texture in [0, 1)."""
    def value(i, j):
        x = math.sin(i * 12.9898 + j * 78.233) * 43758.5453
        return x - math.floor(x)
    return make_texture(width, height, value)


def reference_filter(texture, u, v, jacobian, wrap, box_radius=12):
    """Independent brute-force reference over a wide box.

    Same mathematical contract as the module, derived here from scratch:
    C = J*J^T + 0.25I, kernel exp(-2 q) for q = d^T C^-1 d <= 4.
    """
    height = len(texture)
    width = len(texture[0])
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
                    ti, tj = ix % width, jy % height
                else:
                    ti = min(max(ix, 0), width - 1)
                    tj = min(max(jy, 0), height - 1)
                total += w
                accum += w * texture[tj][ti]
                count += 1
    return accum / total, total, count


class TestConstantConservation(unittest.TestCase):
    def test_constant_texture_repeat_and_clamp(self):
        texture = make_texture(16, 16, lambda i, j: 2.5)
        jacobian = ((1.3, 0.7), (-0.4, 2.1))
        for wrap in (WrapMode.REPEAT, WrapMode.CLAMP):
            result = filter_sample(texture, 5.3, 9.7, jacobian, wrap=wrap)
            self.assertAlmostEqual(result.value, 2.5, delta=1e-12)
            self.assertAlmostEqual(sum(result.weights), 1.0, delta=1e-12)
            self.assertEqual(result.count, len(result.weights))
            self.assertEqual(result.count, len(result.texels))
            self.assertGreater(result.count, 1)

    def test_constant_texture_degenerate_jacobian(self):
        texture = make_texture(8, 8, lambda i, j: -1.25)
        result = filter_sample(texture, 3.5, 4.5, ((0.0, 0.0), (0.0, 0.0)))
        self.assertAlmostEqual(result.value, -1.25, delta=1e-12)


class TestImpulseAnalytic(unittest.TestCase):
    def test_impulse_weights_match_closed_form(self):
        # Zero Jacobian -> C = 0.25 I, C^-1 = 4 I, footprint radius 1.
        # Texel centers at integer offsets: q = 4*(di^2+dj^2) <= 4 keeps
        # exactly (0,0), (+-1,0), (0,+-1); weights exp(-8 r^2).
        texture = make_texture(9, 9, lambda i, j: 1.0 if (i, j) == (4, 4) else 0.0)
        result = filter_sample(
            texture, 4.5, 4.5, ((0.0, 0.0), (0.0, 0.0)), wrap=WrapMode.CLAMP
        )
        edge = math.exp(-8.0)
        expected_sum = 1.0 + 4.0 * edge
        self.assertAlmostEqual(result.weight_sum, expected_sum, delta=1e-15)
        self.assertAlmostEqual(result.value, 1.0 / expected_sum, delta=1e-15)
        self.assertEqual(result.count, 5)
        expected_weights = sorted([1.0 / expected_sum] + [edge / expected_sum] * 4)
        self.assertEqual(len(result.weights), len(expected_weights))
        for got, want in zip(sorted(result.weights), expected_weights):
            self.assertAlmostEqual(got, want, delta=1e-15)
        self.assertEqual(set(result.texels),
                         {(4, 4), (3, 4), (5, 4), (4, 3), (4, 5)})


class TestWideBoxReference(unittest.TestCase):
    def test_matches_independent_reference_repeat(self):
        texture = pseudo_random_texture(16, 16)
        jacobian = ((1.6, 0.9), (-0.5, 1.2))
        for u, v in ((3.7, 11.2), (8.5, 8.5), (0.4, 15.9)):
            expected_value, expected_sum, expected_count = reference_filter(
                texture, u, v, jacobian, WrapMode.REPEAT
            )
            result = filter_sample(texture, u, v, jacobian, wrap=WrapMode.REPEAT)
            self.assertEqual(result.count, expected_count)
            self.assertAlmostEqual(result.value, expected_value, delta=1e-9)
            self.assertAlmostEqual(result.weight_sum, expected_sum, delta=1e-9)


class TestRGBAFiltering(unittest.TestCase):
    def test_opaque_constant_conserved_in_both_alpha_modes(self):
        straight = make_rgba_texture(
            8, 8, lambda i, j: (0.25, 0.5, 0.75, 1.0)
        )
        premultiplied = [
            [(0.25, 0.5, 0.75, 1.0) for _ in range(8)] for _ in range(8)
        ]
        jacobian = ((1.1, -0.3), (0.4, 0.9))

        straight_result = filter_sample(
            straight, 3.5, 4.5, jacobian, alpha_mode=AlphaMode.STRAIGHT
        )
        premultiplied_result = filter_sample(
            premultiplied,
            3.5,
            4.5,
            jacobian,
            alpha_mode=AlphaMode.PREMULTIPLIED,
        )

        self.assertEqual(len(straight_result.value), 4)
        for got, want in zip(straight_result.value, (0.25, 0.5, 0.75, 1.0)):
            self.assertAlmostEqual(got, want, delta=1e-12)
        for got, want in zip(
            premultiplied_result.value, (0.25, 0.5, 0.75, 1.0)
        ):
            self.assertAlmostEqual(got, want, delta=1e-12)
        self.assertEqual(
            straight_result.weights, premultiplied_result.weights
        )

    def test_transparent_color_does_not_bleed_in_straight_texture(self):
        def texel(i, j):
            if (i, j) == (4, 4):
                return (1.0, 0.0, 0.0, 1.0)
            return (0.0, 1.0, 0.0, 0.0)

        texture = make_rgba_texture(9, 9, texel)
        result = filter_sample(
            texture,
            4.5,
            4.5,
            ((0.0, 0.0), (0.0, 0.0)),
            wrap=WrapMode.CLAMP,
            alpha_mode=AlphaMode.STRAIGHT,
        )

        edge = math.exp(-8.0)
        expected_alpha = 1.0 / (1.0 + 4.0 * edge)
        red, green, blue, alpha = result.value
        self.assertAlmostEqual(red, 1.0, delta=1e-12)
        self.assertAlmostEqual(green, 0.0, delta=1e-12)
        self.assertAlmostEqual(blue, 0.0, delta=1e-12)
        self.assertAlmostEqual(alpha, expected_alpha, delta=1e-12)

    def test_transparent_color_does_not_bleed_in_premultiplied_texture(self):
        def texel(i, j):
            if (i, j) == (4, 4):
                return (1.0, 0.0, 0.0, 1.0)
            return (0.8, 0.9, 0.1, 0.0)

        texture = make_rgba_texture(9, 9, texel)
        result = filter_sample(
            texture,
            4.5,
            4.5,
            ((0.0, 0.0), (0.0, 0.0)),
            wrap=WrapMode.CLAMP,
            alpha_mode=AlphaMode.PREMULTIPLIED,
        )

        edge = math.exp(-8.0)
        expected_alpha = 1.0 / (1.0 + 4.0 * edge)
        self.assertAlmostEqual(result.value[0], expected_alpha, delta=1e-12)
        self.assertAlmostEqual(result.value[1], 0.0, delta=1e-12)
        self.assertAlmostEqual(result.value[2], 0.0, delta=1e-12)
        self.assertAlmostEqual(result.value[3], expected_alpha, delta=1e-12)

    def test_output_distinguishes_transparent_from_opaque_black(self):
        transparent = make_rgba_texture(
            4, 4, lambda i, j: (0.0, 0.0, 0.0, 0.0)
        )
        opaque_black = make_rgba_texture(
            4, 4, lambda i, j: (0.0, 0.0, 0.0, 1.0)
        )
        jacobian = ((0.0, 0.0), (0.0, 0.0))

        transparent_result = filter_sample(
            transparent,
            1.5,
            1.5,
            jacobian,
            alpha_mode=AlphaMode.UNPREMULTIPLIED,
        )
        opaque_result = filter_sample(
            opaque_black,
            1.5,
            1.5,
            jacobian,
            alpha_mode=AlphaMode.STRAIGHT,
        )

        self.assertEqual(transparent_result.value, (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(opaque_result.value, (0.0, 0.0, 0.0, 1.0))

    def test_rgba_alpha_uses_same_spatial_weights_as_scalar(self):
        scalar = make_texture(9, 9, lambda i, j: 1.0 if (i, j) == (4, 4) else 0.0)
        rgba = make_rgba_texture(
            9, 9,
            lambda i, j: (0.0, 0.0, 0.0, 1.0)
            if (i, j) == (4, 4)
            else (0.0, 0.0, 0.0, 0.0),
        )
        jacobian = ((1.2, -0.4), (0.3, 0.8))

        scalar_result = filter_sample(scalar, 4.5, 4.5, jacobian)
        rgba_result = filter_sample(
            rgba,
            4.5,
            4.5,
            jacobian,
            alpha_mode=AlphaMode.PREMULTIPLIED,
        )

        self.assertEqual(rgba_result.weights, scalar_result.weights)
        self.assertEqual(rgba_result.texels, scalar_result.texels)
        self.assertAlmostEqual(
            rgba_result.value[3], scalar_result.value, delta=1e-12
        )

    def test_straight_partial_alpha_round_trip(self):
        texture = make_rgba_texture(
            5, 5, lambda i, j: (0.4, 0.6, 0.8, 0.25)
        )
        result = filter_sample(
            texture,
            2.5,
            2.5,
            ((0.7, 0.2), (-0.1, 0.5)),
            alpha_mode=AlphaMode.STRAIGHT,
        )
        for got, want in zip(result.value, (0.4, 0.6, 0.8, 0.25)):
            self.assertAlmostEqual(got, want, delta=1e-12)

    def test_rgba_validation(self):
        good = make_rgba_texture(2, 2, lambda i, j: (0.0, 0.0, 0.0, 1.0))
        jacobian = ((0.0, 0.0), (0.0, 0.0))

        with self.assertRaises(InvalidInputError):
            filter_sample(good, 1.5, 1.5, jacobian)
        with self.assertRaises(InvalidInputError):
            filter_sample(
                good, 1.5, 1.5, jacobian, alpha_mode="straight"
            )
        with self.assertRaises(InvalidInputError):
            bad = [[(0.0, 0.0, 1.0) for _ in range(2)] for _ in range(2)]
            filter_sample(
                bad,
                1.5,
                1.5,
                jacobian,
                alpha_mode=AlphaMode.STRAIGHT,
            )
        with self.assertRaises(InvalidInputError):
            bad = [[(0.0, 0.0, 0.0, 1.1) for _ in range(2)] for _ in range(2)]
            filter_sample(
                bad,
                1.5,
                1.5,
                jacobian,
                alpha_mode=AlphaMode.STRAIGHT,
            )
        with self.assertRaises(InvalidInputError):
            mixed = [
                [1.0, (0.0, 0.0, 0.0, 1.0)],
                [1.0, (0.0, 0.0, 0.0, 1.0)],
            ]
            filter_sample(
                mixed,
                1.0,
                1.0,
                jacobian,
                alpha_mode=AlphaMode.STRAIGHT,
            )

    def test_matches_independent_reference_clamp(self):
        texture = pseudo_random_texture(12, 10)
        jacobian = ((0.8, -0.6), (0.6, 0.8))
        for u, v in ((0.2, 0.3), (11.8, 9.6), (5.5, 4.5)):
            expected_value, expected_sum, expected_count = reference_filter(
                texture, u, v, jacobian, WrapMode.CLAMP
            )
            result = filter_sample(texture, u, v, jacobian, wrap=WrapMode.CLAMP)
            self.assertEqual(result.count, expected_count)
            self.assertAlmostEqual(result.value, expected_value, delta=1e-9)
            self.assertAlmostEqual(result.weight_sum, expected_sum, delta=1e-9)


class TestSlantedFootprint(unittest.TestCase):
    def test_diagonal_anisotropy(self):
        # J = [[2, 0], [2, 0.5]] -> C = [[4.25, 4], [4, 4.5]],
        # C^-1 = [[4.5, -4], [-4, 4.25]] / 3.125.
        # Offset (1, 1): q = 0.75/3.125 = 0.24  -> inside.
        # Offset (1,-1): q = 16.75/3.125 = 5.36 -> outside.
        texture = make_texture(16, 16, lambda i, j: 0.0)
        jacobian = ((2.0, 0.0), (2.0, 0.5))
        result = filter_sample(texture, 8.5, 8.5, jacobian, wrap=WrapMode.CLAMP)
        texels = set(result.texels)
        self.assertIn((9, 9), texels)
        self.assertNotIn((9, 7), texels)
        self.assertIn((7, 7), texels)
        self.assertNotIn((7, 9), texels)
        # The ellipse is symmetric under 180-degree rotation about (u, v).
        for (i, j) in texels:
            self.assertIn((16 - i, 16 - j), texels)


class TestDegenerateJacobian(unittest.TestCase):
    def test_zero_jacobian_isotropic_radius_one(self):
        texture = pseudo_random_texture(8, 8)
        result = filter_sample(
            texture, 4.5, 4.5, ((0.0, 0.0), (0.0, 0.0)), wrap=WrapMode.CLAMP
        )
        self.assertEqual(result.count, 5)  # cross shape, diagonals excluded

    def test_rank_one_jacobian(self):
        # J = [[1.1, 0], [0, 0]] -> C = diag(1.46, 0.25): elongated along u.
        texture = pseudo_random_texture(16, 16)
        jacobian = ((1.1, 0.0), (0.0, 0.0))
        result = filter_sample(texture, 8.5, 8.5, jacobian, wrap=WrapMode.CLAMP)
        texels = set(result.texels)
        self.assertIn((10, 8), texels)   # 2 texels along u: q = 4/1.46 < 4
        self.assertNotIn((11, 8), texels)  # 3 texels along u: q = 9/1.46 > 4
        self.assertNotIn((8, 10), texels)  # 2 texels along v: q = 16 > 4
        # Reference agreement for the degenerate case as well.
        expected_value, _, expected_count = reference_filter(
            texture, 8.5, 8.5, jacobian, WrapMode.CLAMP
        )
        self.assertEqual(result.count, expected_count)
        self.assertAlmostEqual(result.value, expected_value, delta=1e-9)


class TestWrapModes(unittest.TestCase):
    def test_repeat_crosses_period_seam(self):
        # Sample at the center of texel column 0 with radius-1 footprint:
        # candidate column -1 must wrap to column 3.
        texture = make_texture(4, 4, lambda i, j: float(i))
        result = filter_sample(
            texture, 0.5, 2.5, ((0.0, 0.0), (0.0, 0.0)), wrap=WrapMode.REPEAT
        )
        self.assertIn((3, 2), result.texels)
        self.assertIn((0, 2), result.texels)
        self.assertIn((1, 2), result.texels)
        # Cross-shaped footprint: (3,2), (0,2), (1,2), (0,1), (0,3) with
        # weights e^-8, 1, e^-8, e^-8, e^-8 and texel values 3, 0, 1, 0, 0.
        edge = math.exp(-8.0)
        expected = (3.0 * edge + 1.0 * edge) / (1.0 + 4.0 * edge)
        self.assertAlmostEqual(result.value, expected, delta=1e-12)

    def test_repeat_keeps_duplicate_copies(self):
        # 2x2 texture, footprint wider than one period: the same stored
        # texel is hit by several candidate centers, each with its own
        # kernel weight.
        texture = [[1.0, 2.0], [3.0, 4.0]]
        jacobian = ((1.5, 0.0), (0.0, 0.0))
        result = filter_sample(texture, 0.5, 0.5, jacobian, wrap=WrapMode.REPEAT)
        self.assertGreater(result.count, 4)
        self.assertLess(len(set(result.texels)), result.count)
        # Independent recomputation of the duplicate-copy weighted sum.
        c00 = 1.5 * 1.5 + 0.25
        inv00 = 1.0 / c00  # C is diagonal: diag(c00, 0.25)
        inv11 = 1.0 / 0.25
        total = 0.0
        accum = 0.0
        for jy in range(-6, 7):
            for ix in range(-6, 7):
                dx, dy = ix + 0.5 - 0.5, jy + 0.5 - 0.5
                q = inv00 * dx * dx + inv11 * dy * dy
                if q <= 4.0:
                    w = math.exp(-2.0 * q)
                    total += w
                    accum += w * texture[jy % 2][ix % 2]
        self.assertEqual(result.count, sum(
            1 for jy in range(-6, 7) for ix in range(-6, 7)
            if inv00 * ix * ix + inv11 * jy * jy <= 4.0))
        self.assertAlmostEqual(result.value, accum / total, delta=1e-12)

    def test_clamp_sticks_to_edge_with_duplicate_weights(self):
        texture = make_texture(4, 4, lambda i, j: float(i))
        result = filter_sample(
            texture, 0.5, 2.5, ((0.0, 0.0), (0.0, 0.0)), wrap=WrapMode.CLAMP
        )
        # Column -1 clamps onto column 0 but keeps its own kernel weight.
        self.assertEqual(sorted(result.texels),
                         [(0, 1), (0, 2), (0, 2), (0, 3), (1, 2)])
        edge = math.exp(-8.0)
        # Weights 1 (center) + 4 * e^-8; only texel (1, 2) has value != 0.
        expected = edge / (1.0 + 4.0 * edge)
        self.assertAlmostEqual(result.value, expected, delta=1e-12)


class TestBudget(unittest.TestCase):
    def test_budget_exceeded_raises_and_carries_counts(self):
        texture = pseudo_random_texture(64, 64)
        jacobian = ((32.0, 0.0), (0.0, 32.0))
        with self.assertRaises(BudgetExceededError) as ctx:
            filter_sample(texture, 32.0, 32.0, jacobian)
        err = ctx.exception
        self.assertGreater(err.candidates, DEFAULT_BUDGET)
        self.assertEqual(err.budget, DEFAULT_BUDGET)

    def test_custom_budget(self):
        texture = pseudo_random_texture(16, 16)
        jacobian = ((2.0, 0.0), (0.0, 2.0))
        # Default budget is fine...
        result = filter_sample(texture, 8.0, 8.0, jacobian)
        self.assertGreater(result.count, 0)
        # ...but a tight budget rejects the same footprint.
        with self.assertRaises(BudgetExceededError) as ctx:
            filter_sample(texture, 8.0, 8.0, jacobian, budget=10)
        self.assertEqual(ctx.exception.budget, 10)

    def test_budget_error_is_not_a_value_error(self):
        # Budget exhaustion must be distinguishable from invalid input.
        self.assertFalse(issubclass(BudgetExceededError, InvalidInputError))

    def test_huge_jacobian_does_not_hang(self):
        texture = pseudo_random_texture(4, 4)
        with self.assertRaises(BudgetExceededError):
            filter_sample(texture, 2.0, 2.0, ((1e9, 0.0), (0.0, 1e9)))


class TestInputValidation(unittest.TestCase):
    def setUp(self):
        self.texture = pseudo_random_texture(4, 4)
        self.jacobian = ((1.0, 0.0), (0.0, 1.0))

    def test_non_finite_scalars_rejected(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(InvalidInputError):
                filter_sample(self.texture, bad, 1.0, self.jacobian)
            with self.assertRaises(InvalidInputError):
                filter_sample(self.texture, 1.0, bad, self.jacobian)
            with self.assertRaises(InvalidInputError):
                filter_sample(self.texture, 1.0, 1.0, ((bad, 0.0), (0.0, 1.0)))
        with self.assertRaises(InvalidInputError):
            filter_sample([[math.nan]], 0.5, 0.5, self.jacobian)
        with self.assertRaises(InvalidInputError):
            filter_sample([[0.0], [math.inf]], 0.5, 0.5, self.jacobian)

    def test_bool_rejected_for_numeric_inputs(self):
        with self.assertRaises(InvalidInputError):
            filter_sample(self.texture, True, 1.0, self.jacobian)
        with self.assertRaises(InvalidInputError):
            filter_sample(self.texture, 1.0, 1.0, self.jacobian, budget=True)
        with self.assertRaises(InvalidInputError):
            filter_sample([[True, False], [False, True]], 0.5, 0.5, self.jacobian)
        with self.assertRaises(InvalidInputError):
            filter_sample(self.texture, 1.0, 1.0, ((True, 0.0), (0.0, 1.0)))

    def test_wrong_types_rejected(self):
        with self.assertRaises(InvalidInputError):
            filter_sample(self.texture, "1.0", 1.0, self.jacobian)
        with self.assertRaises(InvalidInputError):
            filter_sample(self.texture, 1.0, 1.0, self.jacobian, budget=1.5)
        with self.assertRaises(InvalidInputError):
            filter_sample(self.texture, 1.0, 1.0, self.jacobian, budget=0)
        with self.assertRaises(InvalidInputError):
            filter_sample(self.texture, 1.0, 1.0, self.jacobian, wrap="repeat")

    def test_malformed_texture_rejected(self):
        for bad in ([], [[]], [[1.0, 2.0], [3.0]], "not-a-texture"):
            with self.assertRaises(InvalidInputError):
                filter_sample(bad, 0.5, 0.5, self.jacobian)
        with self.assertRaises(InvalidInputError):
            filter_sample(make_texture(65, 1, lambda i, j: 0.0),
                          0.5, 0.5, self.jacobian)
        with self.assertRaises(InvalidInputError):
            filter_sample(make_texture(1, 65, lambda i, j: 0.0),
                          0.5, 0.5, self.jacobian)

    def test_malformed_jacobian_rejected(self):
        for bad in (((1.0,), (0.0, 1.0)), ((1.0, 0.0),), 3.5, "J"):
            with self.assertRaises(InvalidInputError):
                filter_sample(self.texture, 1.0, 1.0, bad)

    def test_max_size_texture_accepted(self):
        texture = make_texture(64, 64, lambda i, j: 1.0)
        result = filter_sample(texture, 32.5, 32.5, self.jacobian)
        self.assertAlmostEqual(result.value, 1.0, delta=1e-12)


class TestResultContract(unittest.TestCase):
    def test_value_bounded_by_contributing_texels(self):
        texture = pseudo_random_texture(16, 16)
        jacobian = ((1.2, 0.4), (-0.3, 0.9))
        result = filter_sample(texture, 7.31, 4.87, jacobian)
        self.assertIsInstance(result, FilterResult)
        values = [texture[j][i] for (i, j) in result.texels]
        self.assertGreaterEqual(result.value, min(values) - 1e-15)
        self.assertLessEqual(result.value, max(values) + 1e-15)
        self.assertAlmostEqual(sum(result.weights), 1.0, delta=1e-12)
        # Result is immutable.
        with self.assertRaises(Exception):
            result.value = 0.0


if __name__ == "__main__":
    unittest.main()
