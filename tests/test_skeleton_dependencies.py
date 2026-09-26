import unittest

import numpy as np
from osteoid import Skeleton


class SkeletonDependencyTests(unittest.TestCase):
    def test_merging_uint32_edges_offsets_vertex_indices(self):
        skeleton = Skeleton(
            np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
            np.array([[0, 1]], dtype=np.uint32),
        )
        merged = Skeleton.simple_merge([skeleton, skeleton])
        np.testing.assert_array_equal(merged.edges, [[0, 1], [2, 3]])
        self.assertEqual(merged.vertices.shape, (4, 3))


if __name__ == '__main__':
    unittest.main()
