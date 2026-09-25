"""Tests for the analytic u/v derivatives of the elliptical footprint filter.

Verification strategy: at positions flagged as smooth, the analytic
derivatives are checked against central differences of the filtered value
(the task-mandated cross-check).  Nondifferentiable loci (footprint
boundary crossings, repeat/clamp mapping switches) are constructed
analytically from the closed-form q of known texel centers.
"""

import math
import unittest

from ellipse_filter import (
    DEFAULT_BOUNDARY_TOL,
    BudgetExceededError,
    GradientResult,
    InvalidInputError,
    WrapMode,
    filter_sample,
    filter_sample_with_derivatives,
)


def smooth_texture(width, height):
    """Deterministic smooth texture in roughly [-1.5, 1.5]."""
    return [
        [math.sin(0.31 * i + 0.17) * math.cos(0.23 * j - 0.11) + 0.05 * i * j / height
         for i in range(width)]
        for j in range(height)
    ]


def central_differences(texture, u, v, jacobian, wrap, h=1e-6):
    plus_u = filter_sample(texture, u + h, v, jacobian, wrap=wrap).value
    minus_u = filter_sample(texture, u - h, v, jacobian, wrap=wrap).value
    plus_v = filter_sample(texture, u, v + h, jacobian, wrap=wrap).value
    minus_v = filter_sample(texture, u, v - h, jacobian, wrap=wrap).value
    return (plus_u - minus_u) / (2.0 * h), (plus_v - minus_v) / (2.0 * h)


class TestSampleConsistency(unittest.TestCase):
    """The gradient entry point must not change the plain sample values."""

    def test_sample_identical_to_filter_sample(self):
        texture = smooth_texture(16, 12)
        jacobian = ((1.3, 0.7), (-0.4, 2.1))
        for wrap in (WrapMode.REPEAT, WrapMode.CLAMP):
            for u, v in ((3.7, 5.2), (8.5, 8.5), (0.4, 11.9), (-2.3, 17.1)):
                grad = filter_sample_with_derivatives(
                    texture, u, v, jacobian, wrap=wrap)
                plain = filter_sample(texture, u, v, jacobian, wrap=wrap)
                self.assertEqual(grad.sample, plain)

    def test_result_is_frozen(self):
        grad = filter_sample_with_derivatives(
            smooth_texture(4, 4), 1.5, 1.5, ((0.0, 0.0), (0.0, 0.0)))
        self.assertIsInstance(grad, GradientResult)
        with self.assertRaises(Exception):
            grad.dvalue_du = 0.0


class TestConstantTextureZeroGradient(unittest.TestCase):
    def test_zero_gradient_repeat_and_clamp(self):
        texture = [[2.5] * 16 for _ in range(16)]
        jacobian = ((1.3, 0.7), (-0.4, 2.1))
        for wrap in (WrapMode.REPEAT, WrapMode.CLAMP):
            grad = filter_sample_with_derivatives(
                texture, 5.3, 9.7, jacobian, wrap=wrap)
            self.assertFalse(grad.nondifferentiable, msg=str(grad.reasons))
            self.assertAlmostEqual(grad.sample.value, 2.5, delta=1e-12)
            self.assertAlmostEqual(grad.dvalue_du, 0.0, delta=1e-12)
            self.assertAlmostEqual(grad.dvalue_dv, 0.0, delta=1e-12)

    def test_zero_gradient_degenerate_jacobian(self):
        texture = [[-1.25] * 8 for _ in range(8)]
        grad = filter_sample_with_derivatives(
            texture, 3.5, 4.5, ((0.0, 0.0), (0.0, 0.0)), wrap=WrapMode.CLAMP)
        self.assertAlmostEqual(grad.dvalue_du, 0.0, delta=1e-12)
        self.assertAlmostEqual(grad.dvalue_dv, 0.0, delta=1e-12)


class TestCentralDifferenceVerification(unittest.TestCase):
    """Analytic derivatives vs. central differences at safe positions."""

    def check(self, texture, u, v, jacobian, wrap, tol=1e-6):
        grad = filter_sample_with_derivatives(
            texture, u, v, jacobian, wrap=wrap)
        self.assertFalse(grad.nondifferentiable,
                         msg=f"test position must be safe: {grad.reasons}")
        fd_du, fd_dv = central_differences(texture, u, v, jacobian, wrap)
        self.assertAlmostEqual(grad.dvalue_du, fd_du, delta=tol)
        self.assertAlmostEqual(grad.dvalue_dv, fd_dv, delta=tol)

    def test_repeat_interior(self):
        texture = smooth_texture(16, 16)
        jacobian = ((1.6, 0.9), (-0.5, 1.2))
        for u, v in ((3.7, 11.2), (8.5, 8.5), (12.31, 4.87)):
            self.check(texture, u, v, jacobian, WrapMode.REPEAT)

    def test_clamp_interior(self):
        texture = smooth_texture(12, 10)
        jacobian = ((0.8, -0.6), (0.6, 0.8))
        for u, v in ((5.37, 4.13), (9.23, 7.81)):
            self.check(texture, u, v, jacobian, WrapMode.CLAMP)

    def test_degenerate_jacobian(self):
        texture = smooth_texture(16, 16)
        self.check(texture, 8.31, 8.67, ((0.0, 0.0), (0.0, 0.0)),
                   WrapMode.CLAMP)
        self.check(texture, 8.31, 8.67, ((1.1, 0.0), (0.0, 0.0)),
                   WrapMode.REPEAT)

    def test_repeat_both_sides_of_seam(self):
        # Sample positions straddling the u = 0 / u = width wrap seam,
        # away from any footprint-boundary crossing.
        texture = smooth_texture(8, 8)
        jacobian = ((0.9, 0.2), (-0.1, 0.7))
        for u in (-0.27, 0.13, 7.73, 8.21):
            self.check(texture, u, 3.83, jacobian, WrapMode.REPEAT)

    def test_clamp_outside_texture_edge(self):
        # Footprint reaches past the stored texture; clamped copies keep
        # their own kernel weights and weight derivatives.
        texture = smooth_texture(8, 8)
        jacobian = ((1.2, 0.3), (0.2, 1.0))
        for u, v in ((-1.37, 4.21), (9.63, 4.87), (4.13, -1.53), (4.71, 9.29)):
            self.check(texture, u, v, jacobian, WrapMode.CLAMP)

    def test_duplicate_copies_repeat(self):
        # 2x2 texture with a footprint wider than one period: the same
        # stored texel is hit by several candidate centers.  Each copy's
        # kernel weight (and weight derivative) is summed separately.
        texture = [[1.0, 2.0], [3.0, 4.0]]
        jacobian = ((1.5, 0.0), (0.0, 0.0))
        grad = filter_sample_with_derivatives(
            texture, 0.37, 0.5, jacobian, wrap=WrapMode.REPEAT)
        self.assertGreater(grad.sample.count, 4)
        self.assertLess(len(set(grad.sample.texels)), grad.sample.count)
        fd_du, fd_dv = central_differences(
            texture, 0.37, 0.5, jacobian, WrapMode.REPEAT)
        self.assertAlmostEqual(grad.dvalue_du, fd_du, delta=1e-6)
        self.assertAlmostEqual(grad.dvalue_dv, fd_dv, delta=1e-6)

    def test_duplicate_copies_clamped_to_edge(self):
        texture = [[float(i) for i in range(4)] for _ in range(4)]
        jacobian = ((0.0, 0.0), (0.0, 0.0))
        grad = filter_sample_with_derivatives(
            texture, 0.53, 2.47, jacobian, wrap=WrapMode.CLAMP)
        self.assertIn((0, 2), grad.sample.texels)
        fd_du, fd_dv = central_differences(
            texture, 0.53, 2.47, jacobian, WrapMode.CLAMP)
        self.assertAlmostEqual(grad.dvalue_du, fd_du, delta=1e-6)
        self.assertAlmostEqual(grad.dvalue_dv, fd_dv, delta=1e-6)


class TestNondifferentiableDetection(unittest.TestCase):
    """Explicit checks at footprint-boundary and mapping-switch loci."""

    def test_texel_center_exactly_on_boundary(self):
        # Zero Jacobian -> C^-1 = 4 I.  Texel center (1.5, 0.5) is at
        # squared distance 1 from (0.5, 0.5): q = 4 exactly.
        texture = smooth_texture(8, 8)
        grad = filter_sample_with_derivatives(
            texture, 0.5, 0.5, ((0.0, 0.0), (0.0, 0.0)), wrap=WrapMode.CLAMP)
        self.assertTrue(grad.nondifferentiable)
        self.assertTrue(grad.reasons)
        self.assertIn("footprint boundary", grad.reasons[0])
        self.assertTrue(any("(1, 0)" in r for r in grad.reasons),
                        msg=str(grad.reasons))

    def test_boundary_probe_respects_tolerance(self):
        texture = smooth_texture(8, 8)
        jacobian = ((0.0, 0.0), (0.0, 0.0))
        # Texel center (1.5, 0.5) sits exactly at q = 4 when v = 0.57 and
        # u = 1.5 - sqrt(1 - 0.07^2): flagged with the default tolerance.
        v = 0.57
        u_on = 1.5 - math.sqrt(1.0 - 0.07 ** 2)
        flagged = filter_sample_with_derivatives(
            texture, u_on, v, jacobian, wrap=WrapMode.CLAMP)
        self.assertTrue(flagged.nondifferentiable)
        # A step 1e-4 away leaves every texel far from the boundary, so
        # even a tiny tolerance reports a smooth position.
        strict = filter_sample_with_derivatives(
            texture, u_on + 1e-4, v, jacobian, wrap=WrapMode.CLAMP,
            boundary_tol=1e-12)
        self.assertFalse(strict.nondifferentiable)
        self.assertEqual(strict.reasons, ())

    def test_clamp_mapping_switch_reported(self):
        # Boundary texel lies outside the stored texture in clamp mode.
        texture = smooth_texture(8, 8)
        grad = filter_sample_with_derivatives(
            texture, 0.5, 0.5, ((0.0, 0.0), (0.0, 0.0)), wrap=WrapMode.CLAMP)
        clamp_reasons = [r for r in grad.reasons if "clamp mapping" in r]
        self.assertTrue(clamp_reasons, msg=str(grad.reasons))
        self.assertTrue(any("(-1, 0)" in r for r in clamp_reasons),
                        msg=str(clamp_reasons))

    def test_repeat_mapping_switch_reported(self):
        # Same geometry under repeat: the out-of-range boundary texel
        # wraps to the opposite edge of the stored texture.
        texture = smooth_texture(8, 8)
        grad = filter_sample_with_derivatives(
            texture, 0.5, 0.5, ((0.0, 0.0), (0.0, 0.0)), wrap=WrapMode.REPEAT)
        repeat_reasons = [r for r in grad.reasons if "repeat mapping" in r]
        self.assertTrue(repeat_reasons, msg=str(grad.reasons))
        self.assertTrue(any("(7, 0)" in r for r in repeat_reasons),
                        msg=str(repeat_reasons))

    def test_safe_position_reports_no_reasons(self):
        texture = smooth_texture(16, 16)
        grad = filter_sample_with_derivatives(
            texture, 8.31, 8.67, ((1.2, 0.4), (-0.3, 0.9)))
        self.assertFalse(grad.nondifferentiable)
        self.assertEqual(grad.reasons, ())


class TestBudgetAndValidation(unittest.TestCase):
    """Derivative mode shares the budget and validation contract."""

    def test_huge_footprint_budget_exceeded(self):
        texture = smooth_texture(64, 64)
        jacobian = ((32.0, 0.0), (0.0, 32.0))
        with self.assertRaises(BudgetExceededError) as ctx:
            filter_sample_with_derivatives(texture, 32.0, 32.0, jacobian)
        self.assertGreater(ctx.exception.candidates, 4096)
        self.assertEqual(ctx.exception.budget, 4096)

    def test_huge_jacobian_overflow_budget_exceeded(self):
        texture = smooth_texture(4, 4)
        with self.assertRaises(BudgetExceededError):
            filter_sample_with_derivatives(
                texture, 2.0, 2.0, ((1e9, 0.0), (0.0, 1e9)))

    def test_custom_budget_applies_to_derivatives(self):
        texture = smooth_texture(16, 16)
        jacobian = ((2.0, 0.0), (0.0, 2.0))
        with self.assertRaises(BudgetExceededError) as ctx:
            filter_sample_with_derivatives(
                texture, 8.0, 8.0, jacobian, budget=10)
        self.assertEqual(ctx.exception.budget, 10)

    def test_invalid_boundary_tol_rejected(self):
        texture = smooth_texture(4, 4)
        jacobian = ((1.0, 0.0), (0.0, 1.0))
        for bad in (-1.0, math.nan, math.inf, True, "1e-9"):
            with self.assertRaises(InvalidInputError, msg=repr(bad)):
                filter_sample_with_derivatives(
                    texture, 1.5, 1.5, jacobian, boundary_tol=bad)

    def test_invalid_inputs_still_rejected(self):
        texture = smooth_texture(4, 4)
        jacobian = ((1.0, 0.0), (0.0, 1.0))
        with self.assertRaises(InvalidInputError):
            filter_sample_with_derivatives(texture, math.nan, 1.0, jacobian)
        with self.assertRaises(InvalidInputError):
            filter_sample_with_derivatives(
                texture, 1.0, 1.0, jacobian, wrap="repeat")
        with self.assertRaises(InvalidInputError):
            filter_sample_with_derivatives(
                texture, 1.0, 1.0, jacobian, budget=True)

    def test_default_boundary_tol_exported(self):
        self.assertEqual(DEFAULT_BOUNDARY_TOL, 1e-9)


if __name__ == "__main__":
    unittest.main()
