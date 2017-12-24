import numpy as np
# from core.quantization import py_multi_index_util as pymiu
from sklearn.cluster import KMeans
# from core.common.measure_execution_time import mesure_time_wrapper
from core.quantization.quantizer import Quantizer
import re


class PQQuantizer(Quantizer):
    def __init__(self, n_quantizers=2, n_clusters=256, **kmeans_kwargs):
        self.n_quantizers = n_quantizers
        self.n_clusters = n_clusters
        self.subquantizers = []
        # self.py_multi_index_util = pymiu.PyMultiIndexUtil(n_quantizers, n_clusters)
        self.max_scalar_index = (n_clusters) ** n_quantizers - 1
        self.kmeans_kwargs = kmeans_kwargs
        # cpu_count = multiprocessing.cpu_count()
        self.kmeans_kwargs.setdefault('n_jobs', 2)
        self.kmeans_kwargs.setdefault('precompute_distances', 'auto')
        self.kmeans_kwargs.setdefault('n_init', 10)
        # self.kmeans_kwargs.setdefault('max_iter', 30)
        # print(kmeans_kwargs, self.kmeans_kwargs)
        self.kmeans_kwargs.setdefault('verbose', True)

    def fit(self, X: np.ndarray):
        # subvector_length = len(X[0]) // self.n_quantizers
        assert len(X.shape) == 2
        self.subvector_length = X.shape[1] // self.n_quantizers
        # print(subvector_length)
        X = X.reshape((len(X), self.n_quantizers, self.subvector_length))

        kmeans_kwargs = dict(self.kmeans_kwargs)
        kmeans_kwargs.setdefault('tol', float(0.000001))

        self.subquantizers = []
        self.quantization_info = []

        for i in range(self.n_quantizers):
            subvectors = X[:, i, :]
            # subvectors = np.copy(X[:, i, :], order='C')
            # print("subvectorsshape",subvectors.shape)
            kmeans = KMeans(n_clusters=self.n_clusters, **kmeans_kwargs)
            # print(kmeans.max_iter)
            kmeans.fit(subvectors)
            self.subquantizers.append(kmeans)
            self.quantization_info.append(
                {'subspace': i, 'samples_dtype': str(X.dtype), 'subspace_samples_shape': str(subvectors.shape),
                 'kmeans_kwargs': kmeans_kwargs, 'inertia': float(kmeans.inertia_)})

    def get_cluster_centers(self):
        """
            returns (n_quantizers, n_clusters, subvector_length):
        """
        cluster_centers = np.array((
            [self.subquantizers[i].cluster_centers_ for i in range(self.n_quantizers)]
        ))
        return cluster_centers

    def predict(self, X: np.ndarray):
        """
            X - matrix, rows: vectors
            get cluster indices for vectors in X
            X: [
                [x00,x01],
                [x10,x11],
                ...
            ]
            returns:
            [
                i0,
                i1,
                ...
            ]
        """
        subspaced_indices = self.predict_subspace_indices(X)
        pqcodes = np.ravel_multi_index(subspaced_indices.T, (self.n_clusters,) * self.n_quantizers)
        # indices = self.py_multi_index_util.flat_indices(subspaced_indices)
        return pqcodes

    def predict_subspace_indices(self, X) -> np.ndarray:
        """
            X - matrix, rows: vectors
            get cluster indices for vectors in X
            X: [
                [x00,x01],
                [x10,x11],
                ...
            ]
            returns len(X) codes:
            [
                [u0,v0],
                ...
                [u_len(X),v_len(X)]
            ]
        """
        # print("X.shape", X.shape)
        assert len(X.shape) == 2
        centroids = np.empty(shape=(self.n_quantizers, len(X)), dtype=np.int32)
        subvector_length = len(X[0]) // self.n_quantizers
        # print(self.subvector_length, subvector_length)
        assert self.subvector_length == subvector_length
        X = X.reshape((len(X), self.n_quantizers, subvector_length))
        for i in range(self.n_quantizers):
            subvectors = X[:, i, :]
            subquantizer = self.subquantizers[i]
            # print("subvectorsshape", subvectors.shape)
            centroid_indexes = subquantizer.predict(subvectors)
            centroids[i, :] = centroid_indexes

        return centroids.T

    def get_quantization_info(self):
        import json
        quantization_info_josn = json.dumps(self.quantization_info)
        return quantization_info_josn


def restore_from_clusters(subspaced_clusters: np.ndarray) -> PQQuantizer:
    n_subspaces = subspaced_clusters.shape[0]
    n_clusters = subspaced_clusters.shape[1]
    subvector_length = subspaced_clusters.shape[2]
    subspaced_clusters = np.swapaxes(subspaced_clusters, 0, 1)
    subspaced_clusters = subspaced_clusters.reshape((n_clusters, -1))
    pq_quantizer = PQQuantizer(n_clusters=n_clusters, n_quantizers=n_subspaces, n_init=1, max_iter=1, verbose=False)
    pq_quantizer.fit(subspaced_clusters)
    return pq_quantizer


def build_pq_params_str(pq_params: dict):
    pq_params_str = 'pq-{}-{}'.format(pq_params['n_clusters'], pq_params['n_quantizers'])
    return pq_params_str


def extract_pq_params_from_str(pq_params_str):
    result = re.search(r'.*pq-([0-9]+)-([0-9]+).*', pq_params_str)
    # print(result.groups())
    try:
        k = int(result.group(1))
        m = int(result.group(2))
        pq_params = {'n_clusters': k, 'n_quantizers': m}
        return pq_params
    except:
        return None


"""
 self.flatindex_multipliers = np.ones((n_quantizers))
        for i in range(n_quantizers - 2, -1, -1):
            self.flatindex_multipliers[i] = self.flatindex_multipliers[i + 1] * n_clusters

 subspace_indices = self.predict_subspace_indices(X)
        n = X.shape[0]
        flat_indices = np.empty(n)
        for i, subspaces_index in enumerate(subspace_indices):
            flatindex = 0
            for dim in range(len(self.flatindex_multipliers)):
                flatindex += subspaces_index[dim] * self.flatindex_multipliers[dim]
            flat_indices[i] = flatindex

        return flat_indices
"""
