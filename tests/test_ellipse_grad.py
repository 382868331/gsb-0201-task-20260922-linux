"""Tests for filter_sample_grad (analytic d/d(u,v) of the filtered value).

The analytic reference in this file (reference_grad) is an independent
re-implementation: it recomputes C^-1, scans its own box and applies the
quotient rule from scratch.  Finite-difference checks use the public
filter_sample (never the gradient path) as the oracle.
"""

import math
import unittest

from ellipse_filter import (
    BudgetExceededError,
    DEFAULT_BUDGET,
    GradientResult,
    InvalidInputError,
    WrapMode,
    filter_sample,
    filter_sample_grad,
)

J_SLANTED = ((1.2, 0.4), (-0.3, 0.9))


def smooth_texture(width, height):
    """Deterministic smooth texture (band-limited in index space)."""
    return [
        [math.sin(0.3 * i + 0.1) + 0.7 * math.cos(0.4 * j - 0.2)
         for i in range(width)]
        for j in range(height)
    ]


def pseudo_random_texture(width, height):
    def value(i, j):
        x = math.sin(i * 12.9898 + j * 78.233) * 43758.5453
        return x - math.floor(x)
    return [[value(i, j) for i in range(width)] for j in range(height)]


def reference_grad(texture, u, v, jacobian, wrap):
    """Independent analytic-gradient reference, derived from scratch."""
    height = len(texture)
    width = len(texture[0])
    (j00, j01), (j10, j11) = jacobian
    c00 = j00 * j00 + j01 * j01 + 0.25
    c01 = j00 * j10 + j01 * j11
    c11 = j10 * j10 + j11 * j11 + 0.25
    det = c00 * c11 - c01 * c01
    m00, m01, m11 = c11 / det, -c01 / det, c00 / det

    b = bu = bv = a = au = av = 0.0
    ci, cj = math.floor(u), math.floor(v)
    for jy in range(cj - 12, cj + 13):
        for ix in range(ci - 12, ci + 13):
            dx, dy = ix + 0.5 - u, jy + 0.5 - v
            q = m00 * dx * dx + 2.0 * m01 * dx * dy + m11 * dy * dy
            if q <= 4.0:
                w = math.exp(-2.0 * q)
                if wrap is WrapMode.REPEAT:
                    ti, tj = ix % width, jy % height
                else:
                    ti = min(max(ix, 0), width - 1)
                    tj = min(max(jy, 0), height - 1)
                t = texture[tj][ti]
                wu = 4.0 * w * (m00 * dx + m01 * dy)
                wv = 4.0 * w * (m01 * dx + m11 * dy)
                b += w
                a += w * t
                bu += wu
                bv += wv
                au += wu * t
                av += wv * t
    value = a / b
    return value, (au - value * bu) / b, (av - value * bv) / b


def central_diff(texture, u, v, jacobian, wrap, h=1e-5):
    """Central difference of the plain filter value (no gradient path)."""
    du = (filter_sample(texture, u + h, v, jacobian, wrap=wrap).value
          - filter_sample(texture, u - h, v, jacobian, wrap=wrap).value)
    dv = (filter_sample(texture, u, v + h, jacobian, wrap=wrap).value
          - filter_sample(texture, u, v - h, jacobian, wrap=wrap).value)
    return du / (2.0 * h), dv / (2.0 * h)


class TestConstantTextureZeroGradient(unittest.TestCase):
    def test_zero_gradient_repeat_and_clamp(self):
        texture = [[2.5] * 16 for _ in range(16)]
        for wrap in (WrapMode.REPEAT, WrapMode.CLAMP):
            result = filter_sample_grad(texture, 5.3, 9.7, J_SLANTED, wrap=wrap)
            self.assertTrue(result.differentiable)
            self.assertIsNone(result.nondifferentiable_reason)
            self.assertAlmostEqual(result.grad_u, 0.0, delta=1e-12)
            self.assertAlmostEqual(result.grad_v, 0.0, delta=1e-12)
            self.assertAlmostEqual(result.value, 2.5, delta=1e-12)

    def test_zero_gradient_degenerate_jacobian(self):
        texture = [[-1.25] * 8 for _ in range(8)]
        result = filter_sample_grad(
            texture, 3.3, 4.7, ((0.0, 0.0), (0.0, 0.0)))
        self.assertAlmostEqual(result.grad_u, 0.0, delta=1e-12)
        self.assertAlmostEqual(result.grad_v, 0.0, delta=1e-12)


class TestAnalyticVsIndependentReference(unittest.TestCase):
    def test_matches_reference_grad(self):
        texture = pseudo_random_texture(16, 16)
        for wrap in (WrapMode.REPEAT, WrapMode.CLAMP):
            for u, v in ((3.71, 11.23), (8.5, 8.5), (0.42, 15.91)):
                expected = reference_grad(texture, u, v, J_SLANTED, wrap)
                result = filter_sample_grad(texture, u, v, J_SLANTED, wrap=wrap)
                self.assertAlmostEqual(result.value, expected[0], delta=1e-9)
                self.assertAlmostEqual(result.grad_u, expected[1], delta=1e-9)
                self.assertAlmostEqual(result.grad_v, expected[2], delta=1e-9)


class TestCentralDifferenceVerification(unittest.TestCase):
    def test_smooth_texture_safe_points(self):
        texture = smooth_texture(16, 16)
        for wrap in (WrapMode.REPEAT, WrapMode.CLAMP):
            for u, v in ((7.31, 4.87), (2.19, 13.37), (12.73, 8.11)):
                result = filter_sample_grad(texture, u, v, J_SLANTED, wrap=wrap)
                self.assertTrue(result.differentiable, (u, v, wrap))
                fd_u, fd_v = central_diff(texture, u, v, J_SLANTED, wrap)
                self.assertAlmostEqual(result.grad_u, fd_u, delta=1e-6)
                self.assertAlmostEqual(result.grad_v, fd_v, delta=1e-6)


class TestRepeatSeam(unittest.TestCase):
    def test_both_sides_of_period_seam(self):
        # Width-4 texture: the repeat seam sits at u = 4*k.  Sample just
        # left and right of the seam at u = 4; both sides are smooth and
        # the analytic gradient matches the central difference.
        texture = smooth_texture(4, 4)
        jacobian = ((0.6, 0.1), (0.0, 0.5))
        for u in (4.0 - 0.31, 4.0 + 0.31, -0.29, 0.13):
            result = filter_sample_grad(
                texture, u, 1.73, jacobian, wrap=WrapMode.REPEAT)
            self.assertTrue(result.differentiable, u)
            fd_u, fd_v = central_diff(
                texture, u, 1.73, jacobian, WrapMode.REPEAT)
            self.assertAlmostEqual(result.grad_u, fd_u, delta=1e-6)
            self.assertAlmostEqual(result.grad_v, fd_v, delta=1e-6)

    def test_seam_gradient_continuous_across_period(self):
        # Repeat is periodic: sampling one period apart must give the same
        # value and gradient.
        texture = pseudo_random_texture(4, 4)
        jacobian = ((0.6, 0.1), (0.0, 0.5))
        a = filter_sample_grad(texture, 1.37, 2.11, jacobian)
        b = filter_sample_grad(texture, 1.37 + 4.0, 2.11 + 8.0, jacobian)
        self.assertAlmostEqual(a.value, b.value, delta=1e-12)
        self.assertAlmostEqual(a.grad_u, b.grad_u, delta=1e-12)
        self.assertAlmostEqual(a.grad_v, b.grad_v, delta=1e-12)


class TestClampOutsideEdge(unittest.TestCase):
    def test_outside_left_and_bottom_edges(self):
        texture = smooth_texture(8, 8)
        jacobian = ((0.7, 0.2), (-0.1, 0.6))
        for u, v in ((-1.31, 3.7), (2.9, -1.13), (8.83, 6.1)):
            result = filter_sample_grad(
                texture, u, v, jacobian, wrap=WrapMode.CLAMP)
            self.assertTrue(result.differentiable, (u, v))
            fd_u, fd_v = central_diff(
                texture, u, v, jacobian, WrapMode.CLAMP)
            self.assertAlmostEqual(result.grad_u, fd_u, delta=1e-6)
            self.assertAlmostEqual(result.grad_v, fd_v, delta=1e-6)

    def test_clamped_copies_each_carry_weight_derivative(self):
        # Far outside the left edge every candidate clamps to column 0;
        # only column 0 contributes, so value and gradient are both 0.
        texture = [[float(i) for i in range(4)] for _ in range(4)]
        result = filter_sample_grad(
            texture, -0.9, 1.5, ((0.4, 0.0), (0.0, 0.4)),
            wrap=WrapMode.CLAMP)
        self.assertTrue(all(i == 0 for (i, _) in result.texels))
        self.assertAlmostEqual(result.value, 0.0, delta=1e-15)
        self.assertAlmostEqual(result.grad_u, 0.0, delta=1e-12)


class TestCoincidentCopies(unittest.TestCase):
    def test_duplicate_copies_gradient(self):
        # 2x2 texture, footprint wider than one period: each stored texel
        # is hit by several candidate centers; every copy keeps its own
        # kernel weight and weight derivative.
        texture = [[1.0, 2.0], [3.0, 4.0]]
        jacobian = ((1.5, 0.2), (0.1, 1.4))
        result = filter_sample_grad(texture, 0.73, 0.41, jacobian)
        self.assertTrue(result.differentiable)
        self.assertLess(len(set(result.texels)), result.count)
        fd_u, fd_v = central_diff(texture, 0.73, 0.41, jacobian,
                                  WrapMode.REPEAT)
        self.assertAlmostEqual(result.grad_u, fd_u, delta=1e-6)
        self.assertAlmostEqual(result.grad_v, fd_v, delta=1e-6)
        # Independent per-copy analytic recomputation.
        expected = reference_grad(texture, 0.73, 0.41, jacobian,
                                  WrapMode.REPEAT)
        self.assertAlmostEqual(result.grad_u, expected[1], delta=1e-9)
        self.assertAlmostEqual(result.grad_v, expected[2], delta=1e-9)


class TestNondifferentiablePoints(unittest.TestCase):
    def test_footprint_boundary_flagged(self):
        # Zero Jacobian: C^-1 = 4I, so texel centers at distance exactly 1
        # from (0.5, 0.5) sit on the footprint boundary q = 4.
        texture = smooth_texture(4, 4)
        zero_j = ((0.0, 0.0), (0.0, 0.0))
        result = filter_sample_grad(texture, 0.5, 0.5, zero_j)
        self.assertFalse(result.differentiable)
        self.assertIn("footprint boundary", result.nondifferentiable_reason)
        # The value itself is still produced.
        plain = filter_sample(texture, 0.5, 0.5, zero_j)
        self.assertEqual(result.value, plain.value)

    def test_box_edge_flagged(self):
        # J = 0.3 I -> half_u = 2*sqrt(0.34); placing u so that the left
        # box edge lands exactly on a texel center makes the candidate set
        # switch discretely.
        half_u = 2.0 * math.sqrt(0.3 * 0.3 + 0.25)
        u = 0.5 + half_u  # left edge u - half_u = 0.5, exactly a center
        texture = smooth_texture(8, 8)
        jacobian = ((0.3, 0.0), (0.0, 0.3))
        result = filter_sample_grad(texture, u, 3.71, jacobian)
        self.assertFalse(result.differentiable)
        self.assertIn("bounding-box edge", result.nondifferentiable_reason)

    def test_safe_point_not_flagged(self):
        texture = smooth_texture(16, 16)
        result = filter_sample_grad(texture, 7.31, 4.87, J_SLANTED)
        self.assertTrue(result.differentiable)
        self.assertIsNone(result.nondifferentiable_reason)


class TestGradientBudget(unittest.TestCase):
    def test_huge_footprint_rejected_in_grad_mode(self):
        texture = pseudo_random_texture(64, 64)
        jacobian = ((32.0, 0.0), (0.0, 32.0))
        with self.assertRaises(BudgetExceededError) as ctx:
            filter_sample_grad(texture, 32.0, 32.0, jacobian)
        self.assertGreater(ctx.exception.candidates, DEFAULT_BUDGET)
        self.assertEqual(ctx.exception.budget, DEFAULT_BUDGET)

    def test_grad_mode_cannot_bypass_budget(self):
        texture = pseudo_random_texture(16, 16)
        jacobian = ((2.0, 0.0), (0.0, 2.0))
        # Same footprint passes filter_sample with the default budget but
        # must be rejected in grad mode with a tight budget as well.
        self.assertGreater(filter_sample(texture, 8.0, 8.0, jacobian).count, 0)
        with self.assertRaises(BudgetExceededError):
            filter_sample_grad(texture, 8.0, 8.0, jacobian, budget=10)

    def test_huge_jacobian_does_not_hang(self):
        with self.assertRaises(BudgetExceededError):
            filter_sample_grad([[1.0]], 0.5, 0.5, ((1e9, 0.0), (0.0, 1e9)))


class TestGradientContract(unittest.TestCase):
    def test_value_and_weights_match_plain_filter(self):
        texture = pseudo_random_texture(16, 16)
        for wrap in (WrapMode.REPEAT, WrapMode.CLAMP):
            plain = filter_sample(texture, 7.31, 4.87, J_SLANTED, wrap=wrap)
            grad = filter_sample_grad(texture, 7.31, 4.87, J_SLANTED, wrap=wrap)
            self.assertIsInstance(grad, GradientResult)
            self.assertEqual(grad.value, plain.value)
            self.assertEqual(grad.weights, plain.weights)
            self.assertEqual(grad.texels, plain.texels)
            self.assertEqual(grad.count, plain.count)
            self.assertEqual(grad.weight_sum, plain.weight_sum)

    def test_invalid_tolerance_rejected(self):
        texture = smooth_texture(4, 4)
        for bad in (-1.0, math.nan, math.inf, True, "x"):
            with self.assertRaises(InvalidInputError):
                filter_sample_grad(texture, 1.0, 1.0, J_SLANTED,
                                   boundary_tol=bad)
            with self.assertRaises(InvalidInputError):
                filter_sample_grad(texture, 1.0, 1.0, J_SLANTED,
                                   edge_tol=bad)

    def test_invalid_inputs_still_rejected(self):
        texture = smooth_texture(4, 4)
        with self.assertRaises(InvalidInputError):
            filter_sample_grad(texture, math.nan, 1.0, J_SLANTED)
        with self.assertRaises(InvalidInputError):
            filter_sample_grad(texture, 1.0, 1.0, J_SLANTED, budget=True)
        with self.assertRaises(InvalidInputError):
            filter_sample_grad(texture, 1.0, 1.0, J_SLANTED, wrap="repeat")


if __name__ == "__main__":
    unittest.main()
