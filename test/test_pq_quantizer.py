import unittest
import numpy as np
from core.quantization.pq_quantizer import PQQuantizer, restore_from_clusters

class PQQuantizerTest(unittest.TestCase):
    X = None
    quantizer = None

    @classmethod
    def setUpClass(cls):
        cls.X = np.array([
            [0, 2, 4, 2],
            [0, 3, 4, 2],
            [1, 3, 5, 1],
            [1, 3, 6, 0]
        ])
        cls.quantizer = PQQuantizer(2, 2)
        cls.quantizer.fit(cls.X)

    def test_predict_subspace_indices_shape(self):
        predicted_base_vectors = PQQuantizerTest.quantizer.predict_subspace_indices(PQQuantizerTest.X)
        predicted_base_vectors_shape = predicted_base_vectors.shape
        truth_shape = (4, 2)
        self.assertEqual(predicted_base_vectors_shape, truth_shape)

    def test_clusters_shape(self):
        cluster_centers = PQQuantizerTest.quantizer.get_cluster_centers()
        cluster_centers_shape = cluster_centers.shape
        truth_shape = (2, 2, 2)
        self.assertEqual(cluster_centers_shape, truth_shape)

    def test_predict_shape(self):
        predicted_base_vectors = PQQuantizerTest.quantizer.predict(PQQuantizerTest.X)
        predicted_base_vectors_shape = predicted_base_vectors.shape
        truth_shape = (4,)
        self.assertEqual(predicted_base_vectors_shape, truth_shape)

    def test_restore_from_clusters(self):
        cluster_centers = PQQuantizerTest.quantizer.get_cluster_centers()
        pq_quantizer_restored = restore_from_clusters(cluster_centers)
        clusters_restored = pq_quantizer_restored.get_cluster_centers()
        self.assertEqual(clusters_restored.shape, cluster_centers.shape)

    def test_max_scalar_index(self):
        max_scalar_index = PQQuantizerTest.quantizer.max_scalar_index
        self.assertEqual(max_scalar_index, 1+2*(1))


if __name__ == '__main__':
    unittest.main()
