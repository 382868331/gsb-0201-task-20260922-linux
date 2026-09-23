"""Tests for ewa_filter.

The brute-force wide-box reference in this file re-derives the filter from
the task contract (C = J J^T + 0.25 I, q <= 4, w = exp(-2 q)) and integrates
over a deliberately oversized box.  It never calls ewa_filter, so it is an
independent oracle for value / weight_sum / count.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ewa_filter import (
    DEFAULT_MAX_CANDIDATES,
    MAX_TEXTURE_SIZE,
    WRAP_CLAMP,
    WRAP_REPEAT,
    BudgetExceededError,
    InvalidInputError,
    filter_footprint,
)


# ---------------------------------------------------------------------------
# Independent reference implementation (NOT the module under test).
# ---------------------------------------------------------------------------

def wide_box_reference(grid, u, v, jacobian, wrap):
    """Brute-force integration over a box padded well past the footprint."""
    h = len(grid)
    w = len(grid[0])
    (j00, j01), (j10, j11) = jacobian
    c11 = j00 * j00 + j01 * j01 + 0.25
    c12 = j00 * j10 + j01 * j11
    c22 = j10 * j10 + j11 * j11 + 0.25
    det = c11 * c22 - c12 * c12
    i11, i12, i22 = c22 / det, -c12 / det, c11 / det
    # Ellipse is contained in |dx| <= 2*sqrt(c11); pad generously so that no
    # texel with q <= 4 can be missed regardless of box-rounding choices.
    rx = 2.0 * math.sqrt(c11) + 2.0
    ry = 2.0 * math.sqrt(c22) + 2.0
    weighted = 0.0
    wsum = 0.0
    count = 0
    tj = math.ceil(v - ry - 0.5)
    while tj + 0.5 <= v + ry:
        dy = tj + 0.5 - v
        ti = math.ceil(u - rx - 0.5)
        while ti + 0.5 <= u + rx:
            dx = ti + 0.5 - u
            q = (i11 * dx + i12 * dy) * dx + (i12 * dx + i22 * dy) * dy
            if q <= 4.0:
                wgt = math.exp(-2.0 * q)
                if wrap == WRAP_REPEAT:
                    mi, mj = ti % w, tj % h
                else:
                    mi = min(max(ti, 0), w - 1)
                    mj = min(max(tj, 0), h - 1)
                weighted += wgt * grid[mj][mi]
                wsum += wgt
                count += 1
            ti += 1
        tj += 1
    return weighted / wsum, wsum, count


def make_grid(w, h, fn):
    return [[fn(x, y) for x in range(w)] for y in range(h)]


def rot(deg):
    a = math.radians(deg)
    return math.cos(a), math.sin(a)


def scaled_rot_jacobian(sx, sy, deg):
    """J = R(deg) @ diag(sx, sy): anisotropic, rotated footprint."""
    c, s = rot(deg)
    return [[c * sx, -s * sy], [s * sx, c * sy]]


class TestAgainstWideBoxReference(unittest.TestCase):
    """Module output must match the independent wide-box oracle."""

    def check(self, grid, u, v, jacobian, wrap):
        res = filter_footprint(grid, u, v, jacobian, wrap=wrap)
        ref_value, ref_wsum, ref_count = wide_box_reference(grid, u, v, jacobian, wrap)
        self.assertEqual(res.count, ref_count)
        self.assertAlmostEqual(res.weight_sum, ref_wsum, delta=abs(ref_wsum) * 1e-12)
        self.assertAlmostEqual(res.value, ref_value, delta=1e-12)

    def test_rotated_anisotropic_repeat(self):
        grid = make_grid(16, 16, lambda x, y: 0.3 * x + 0.7 * y + ((x * y) % 5))
        self.check(grid, 7.3, 9.8, scaled_rot_jacobian(3.0, 0.8, 35.0), WRAP_REPEAT)

    def test_rotated_anisotropic_clamp(self):
        grid = make_grid(9, 13, lambda x, y: math.sin(x) * math.cos(y))
        self.check(grid, 0.4, 12.1, scaled_rot_jacobian(2.5, 1.2, -60.0), WRAP_CLAMP)

    def test_degenerate_zero_jacobian(self):
        grid = make_grid(8, 8, lambda x, y: x * 8 + y)
        self.check(grid, 3.5, 2.5, [[0.0, 0.0], [0.0, 0.0]], WRAP_REPEAT)

    def test_nearly_singular_jacobian(self):
        grid = make_grid(12, 10, lambda x, y: (x - y) ** 2)
        self.check(grid, 5.5, 4.5, [[1e-9, 0.0], [0.0, 2.0]], WRAP_REPEAT)

    def test_across_repeat_seam(self):
        grid = make_grid(8, 8, lambda x, y: x + 10 * y)
        self.check(grid, 0.2, 7.9, [[1.0, 0.0], [0.0, 1.0]], WRAP_REPEAT)

    def test_sheared_jacobian(self):
        grid = make_grid(16, 16, lambda x, y: x * y)
        self.check(grid, 8.5, 8.5, [[2.0, 1.5], [0.0, 0.5]], WRAP_CLAMP)


class TestConstantConservation(unittest.TestCase):
    def test_constant_texture_returns_constant(self):
        for wrap in (WRAP_REPEAT, WRAP_CLAMP):
            grid = make_grid(16, 16, lambda x, y: 3.25)
            res = filter_footprint(grid, 5.7, 11.2, scaled_rot_jacobian(4.0, 0.6, 20.0), wrap=wrap)
            self.assertAlmostEqual(res.value, 3.25, delta=1e-12)
            self.assertGreater(res.weight_sum, 0.0)
            self.assertEqual(res.count, len(res.contributions))

    def test_constant_texture_degenerate_jacobian(self):
        grid = make_grid(4, 4, lambda x, y: -1.5)
        res = filter_footprint(grid, 1.5, 1.5, [[0.0, 0.0], [0.0, 0.0]])
        self.assertAlmostEqual(res.value, -1.5, delta=1e-12)


class TestImpulseAnalytic(unittest.TestCase):
    """J = 0.5*I at a texel center: weights are analytically known.

    C = 0.5 I, C^-1 = 2 I (exactly representable), q = 2*(dx^2+dy^2),
    w = exp(-4*(dx^2+dy^2)).  The footprint boundary dx^2+dy^2 = 2 lands
    exactly on q = 4.0 in IEEE arithmetic, so the analytic expectation and
    the module agree on the boundary texels without any tolerance fudge.
    """

    def test_impulse_weights(self):
        n = 8
        cx, cy = 3, 2
        grid = make_grid(n, n, lambda x, y: 1.0 if (x, y) == (cx, cy) else 0.0)
        res = filter_footprint(grid, cx + 0.5, cy + 0.5, [[0.5, 0.0], [0.0, 0.5]])

        offsets = [
            (dx, dy)
            for dx in range(-3, 4)
            for dy in range(-3, 4)
            if dx * dx + dy * dy <= 2
        ]
        self.assertEqual(len(offsets), 9)  # (0,0), 4 axis, 4 diagonal
        self.assertEqual(res.count, 9)

        wsum = sum(math.exp(-4.0 * (dx * dx + dy * dy)) for dx, dy in offsets)
        self.assertAlmostEqual(res.weight_sum, wsum, delta=wsum * 1e-12)
        # Impulse of height 1 at the center: value = w(0,0) / sum(w).
        self.assertAlmostEqual(res.value, 1.0 / wsum, delta=1e-12)

        # Every contribution's weight matches the analytic kernel value.
        by_texel = {(c.texel_x, c.texel_y): c.weight for c in res.contributions}
        for dx, dy in offsets:
            tx, ty = (cx + dx) % n, (cy + dy) % n
            self.assertAlmostEqual(
                by_texel[(tx, ty)],
                math.exp(-4.0 * (dx * dx + dy * dy)),
                delta=1e-15,
            )


class TestFootprintShape(unittest.TestCase):
    def test_slanted_footprint_orientation(self):
        # Long axis along 45 degrees: J = R(45) @ diag(4, 0.5).
        # Eigenvalues of C: 16.25 (long) and 0.5 (short).
        grid = make_grid(32, 32, lambda x, y: 0.0)
        j = scaled_rot_jacobian(4.0, 0.5, 45.0)
        res = filter_footprint(grid, 16.5, 16.5, j)
        texels = {(c.texel_x, c.texel_y) for c in res.contributions}
        # d = (2, 2): along the long axis, q = 8/16.25 < 4 -> included.
        self.assertIn((18, 18), texels)
        # d = (-2, 2): perpendicular, q = 8/0.5 = 16 > 4 -> excluded.
        self.assertNotIn((14, 18), texels)
        self.assertNotIn((18, 14), texels)

    def test_degenerate_jacobian_is_unit_disc(self):
        # J = 0 -> C = 0.25 I, q = 4|d|^2 <= 4 -> |d| <= 1.
        grid = make_grid(8, 8, lambda x, y: 0.0)
        res = filter_footprint(grid, 3.5, 3.5, [[0.0, 0.0], [0.0, 0.0]])
        texels = {(c.texel_x, c.texel_y) for c in res.contributions}
        self.assertEqual(
            texels, {(3, 3), (2, 3), (4, 3), (3, 2), (3, 4)}
        )
        self.assertEqual(res.count, 5)


class TestWrapModes(unittest.TestCase):
    def test_repeat_wraps_across_seam(self):
        n = 8
        grid = make_grid(n, n, lambda x, y: float(x))
        res = filter_footprint(grid, 0.5, 4.5, [[1.0, 0.0], [0.0, 1.0]], wrap=WRAP_REPEAT)
        xs = {c.texel_x for c in res.contributions}
        # Footprint reaches texel columns -2..2 -> wrapped to 6, 7, 0, 1, 2.
        self.assertEqual(xs, {6, 7, 0, 1, 2})

    def test_repeat_keeps_per_copy_weights(self):
        # 4x4 texture, footprint spanning more than one period: the same
        # texel is hit by several candidate positions and each copy keeps
        # its own kernel weight (copies are not merged).
        grid = make_grid(4, 4, lambda x, y: 1.0)
        res = filter_footprint(grid, 2.3, 2.5, [[1.5, 0.0], [0.0, 1.5]], wrap=WRAP_REPEAT)
        copies_of_origin = [
            c for c in res.contributions if (c.texel_x, c.texel_y) == (0, 0)
        ]
        self.assertEqual(len(copies_of_origin), 4)  # (0,0),(4,0),(0,4),(4,4)
        self.assertGreater(res.count, len({(c.texel_x, c.texel_y) for c in res.contributions}))
        # Copies at different offsets carry different kernel weights.
        weights = sorted(c.weight for c in copies_of_origin)
        self.assertLess(weights[0], weights[-1])

    def test_clamp_sticks_to_edge(self):
        grid = make_grid(8, 8, lambda x, y: float(x))
        res = filter_footprint(grid, -5.0, 4.0, [[0.0, 0.0], [0.0, 0.0]], wrap=WRAP_CLAMP)
        self.assertTrue(all(c.texel_x == 0 for c in res.contributions))
        # Unit disc around (-5, 4): texel centers (+-0.5, +-0.5) away -> 4 hits.
        self.assertEqual(res.count, 4)
        self.assertEqual({c.texel_y for c in res.contributions}, {3, 4})
        self.assertAlmostEqual(res.value, 0.0, delta=1e-15)

    def test_repeat_vs_clamp_differ_at_border(self):
        grid = make_grid(8, 8, lambda x, y: float(x))
        rep = filter_footprint(grid, 0.5, 4.5, [[1.0, 0.0], [0.0, 1.0]], wrap=WRAP_REPEAT)
        clp = filter_footprint(grid, 0.5, 4.5, [[1.0, 0.0], [0.0, 1.0]], wrap=WRAP_CLAMP)
        self.assertNotAlmostEqual(rep.value, clp.value, places=6)


class TestBudget(unittest.TestCase):
    def test_budget_exceeded_raises(self):
        grid = make_grid(8, 8, lambda x, y: 1.0)
        with self.assertRaises(BudgetExceededError) as ctx:
            filter_footprint(grid, 4.0, 4.0, [[40.0, 0.0], [0.0, 40.0]])
        err = ctx.exception
        self.assertGreater(err.candidates, DEFAULT_MAX_CANDIDATES)
        self.assertEqual(err.limit, DEFAULT_MAX_CANDIDATES)

    def test_budget_is_checked_before_sampling(self):
        # A tiny custom budget must trip even on a small footprint, and the
        # error must be BudgetExceededError (not a silent point sample).
        grid = make_grid(8, 8, lambda x, y: 1.0)
        with self.assertRaises(BudgetExceededError) as ctx:
            filter_footprint(grid, 4.5, 4.5, [[1.0, 0.0], [0.0, 1.0]], max_candidates=4)
        self.assertEqual(ctx.exception.limit, 4)
        # Same call with adequate budget succeeds.
        res = filter_footprint(grid, 4.5, 4.5, [[1.0, 0.0], [0.0, 1.0]], max_candidates=64)
        self.assertGreater(res.count, 0)

    def test_budget_error_is_not_input_error(self):
        self.assertFalse(issubclass(BudgetExceededError, InvalidInputError))


class TestInputValidation(unittest.TestCase):
    def setUp(self):
        self.grid = make_grid(4, 4, lambda x, y: 1.0)
        self.j = [[1.0, 0.0], [0.0, 1.0]]

    def test_nan_and_inf_rejected(self):
        bad = float("nan")
        inf = float("inf")
        with self.assertRaises(InvalidInputError):
            filter_footprint([[1.0, bad], [1.0, 1.0]], 0.5, 0.5, self.j)
        with self.assertRaises(InvalidInputError):
            filter_footprint([[1.0, inf], [1.0, 1.0]], 0.5, 0.5, self.j)
        with self.assertRaises(InvalidInputError):
            filter_footprint(self.grid, bad, 0.5, self.j)
        with self.assertRaises(InvalidInputError):
            filter_footprint(self.grid, 0.5, inf, self.j)
        with self.assertRaises(InvalidInputError):
            filter_footprint(self.grid, 0.5, 0.5, [[1.0, 0.0], [bad, 1.0]])
        with self.assertRaises(InvalidInputError):
            filter_footprint(self.grid, 0.5, 0.5, [[1.0, 0.0], [0.0, inf]])

    def test_bool_rejected_for_numeric_params(self):
        with self.assertRaises(InvalidInputError):
            filter_footprint(self.grid, True, 0.5, self.j)
        with self.assertRaises(InvalidInputError):
            filter_footprint(self.grid, 0.5, 0.5, self.j, max_candidates=True)
        with self.assertRaises(InvalidInputError):
            filter_footprint([[True, 1.0], [1.0, 1.0]], 0.5, 0.5, self.j)

    def test_non_numeric_rejected(self):
        with self.assertRaises(InvalidInputError):
            filter_footprint(self.grid, "0.5", 0.5, self.j)
        with self.assertRaises(InvalidInputError):
            filter_footprint(self.grid, 0.5, 0.5, self.j, max_candidates=4.5)

    def test_texture_shape_validation(self):
        with self.assertRaises(InvalidInputError):
            filter_footprint([], 0.5, 0.5, self.j)
        with self.assertRaises(InvalidInputError):
            filter_footprint([[]], 0.5, 0.5, self.j)
        with self.assertRaises(InvalidInputError):
            filter_footprint([[1.0, 2.0], [1.0]], 0.5, 0.5, self.j)  # ragged
        with self.assertRaises(InvalidInputError):
            filter_footprint(
                make_grid(MAX_TEXTURE_SIZE + 1, 1, lambda x, y: 0.0), 0.5, 0.5, self.j
            )
        with self.assertRaises(InvalidInputError):
            filter_footprint(
                make_grid(1, MAX_TEXTURE_SIZE + 1, lambda x, y: 0.0), 0.5, 0.5, self.j
            )
        # Exactly at the limit is accepted.
        res = filter_footprint(
            make_grid(MAX_TEXTURE_SIZE, MAX_TEXTURE_SIZE, lambda x, y: 2.0),
            32.5, 32.5, self.j,
        )
        self.assertAlmostEqual(res.value, 2.0, delta=1e-12)

    def test_jacobian_shape_validation(self):
        with self.assertRaises(InvalidInputError):
            filter_footprint(self.grid, 0.5, 0.5, [[1.0, 0.0]])
        with self.assertRaises(InvalidInputError):
            filter_footprint(self.grid, 0.5, 0.5, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        with self.assertRaises(InvalidInputError):
            filter_footprint(self.grid, 0.5, 0.5, 1.0)

    def test_wrap_mode_validation(self):
        with self.assertRaises(InvalidInputError):
            filter_footprint(self.grid, 0.5, 0.5, self.j, wrap="mirror")
        with self.assertRaises(InvalidInputError):
            filter_footprint(self.grid, 0.5, 0.5, self.j, wrap=1)

    def test_max_candidates_validation(self):
        with self.assertRaises(InvalidInputError):
            filter_footprint(self.grid, 0.5, 0.5, self.j, max_candidates=0)
        with self.assertRaises(InvalidInputError):
            filter_footprint(self.grid, 0.5, 0.5, self.j, max_candidates=-3)

    def test_error_message_names_parameter(self):
        with self.assertRaises(InvalidInputError) as ctx:
            filter_footprint(self.grid, float("nan"), 0.5, self.j)
        self.assertIn("u", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
