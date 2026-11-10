import unittest
from itertools import combinations

import numpy as np

from multirocket.kernel_groups import (
    NUM_KERNELS,
    select_kernel_indices,
    validate_random_state,
)


class KernelGroupsTest(unittest.TestCase):
    def test_baseline_keeps_all_patterns(self):
        np.testing.assert_array_equal(
            select_kernel_indices("none"),
            np.arange(NUM_KERNELS, dtype=np.int32),
        )

    def test_temporal_scale_selects_across_all_groups(self):
        selected = select_kernel_indices("temporal_scale")
        patterns = np.asarray(list(combinations(range(9), 3)), dtype=np.int32)
        spans = patterns[selected, 2] - patterns[selected, 0]

        self.assertEqual(len(selected), 43)
        self.assertEqual(len(np.unique(selected)), len(selected))
        self.assertTrue(np.all(selected[:-1] < selected[1:]))
        self.assertTrue(np.any(spans <= 3))
        self.assertTrue(np.any((spans >= 4) & (spans <= 5)))
        self.assertTrue(np.any(spans >= 6))

    def test_unknown_grouping_is_rejected(self):
        with self.assertRaises(ValueError):
            select_kernel_indices("unknown")

    def test_random_state_validation(self):
        self.assertIsNone(validate_random_state(None))
        self.assertEqual(validate_random_state(np.int64(7)), 7)
        for invalid_state in (-1, 2 ** 31, 1.5, True):
            with self.subTest(random_state=invalid_state):
                with self.assertRaises(ValueError):
                    validate_random_state(invalid_state)


if __name__ == "__main__":
    unittest.main()
