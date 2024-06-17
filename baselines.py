import time

import numpy as np
import scipy.sparse as sp
import networkx as nx
from pygsp.graphs import Graph
from sklearn.cluster import SpectralClustering
from networkx.linalg.laplacianmatrix import laplacian_matrix

import graphzoom
from spectral_coarsening.coarsening import multilevel_graph_coarsening, spectral_graph_coarsening
from graph_coarsening.coarsening_utils import coarsen


def mgc(G, ratio):
    N = G.number_of_nodes()
    if N <= 5:
        return G, 0
    n = max(int(np.ceil(N * ratio)), 5)
    t0 = time.perf_counter()
    As, *_ = multilevel_graph_coarsening(nx.to_numpy_array(G), n)
    t1 = time.perf_counter()
    Gs = nx.from_numpy_array(As)
    return Gs, t1 - t0


def sgc(G, ratio):
    N = G.number_of_nodes()
    if N <= 5:
        return G, 0
    n = max(int(np.ceil(N * ratio)), 5)
    t0 = time.perf_counter()
    As, *_ = spectral_graph_coarsening(nx.to_numpy_array(G), n)
    t1 = time.perf_counter()
    Gs = nx.from_numpy_array(As)
    return Gs, t1 - t0


# Spectral Clustering
def sc_summarize(G, ratio):
    A = nx.to_scipy_sparse_array(G)
    N = G.number_of_nodes()
    if N <= 5:
        return G, 0
    n = max(int(np.ceil(N * ratio)), 5)
    if n == N:
        n -= 1

    t0 = time.perf_counter()
    sc = SpectralClustering(
        n_clusters=n, affinity='precomputed', n_init=3, n_jobs=-1)
    sc.fit(A)
    t1 = time.perf_counter()

    res = sc.labels_
    P = sp.csr_matrix(([1] * N, (res, np.arange(N))), shape=(n, N))
    As = P @ (A @ P.T)
    Gs = nx.from_scipy_sparse_array(As)
    return Gs, t1 - t0


# Source: https://github.com/cornell-zhang/GraphZoom
def graphzoom_summarize(G, ratio):
    N = G.number_of_nodes()
    if N <= 5:
        return G, 0
    n = max(int(np.ceil(N * ratio)), 5)
    t0 = time.perf_counter()
    Gs, Projs, *_ = graphzoom.sim_coarse(laplacian_matrix(G), 100, n)
    t1 = time.perf_counter()
    if len(Projs) > 1:
        P = Projs[0]
        for P2 in Projs[1:]:
            P = P @ P2
        P = P.T
        As = P @ nx.to_scipy_sparse_array(G) @ P.T
        Gs = nx.from_scipy_sparse_array(As)
    return Gs, t1-t0


# Source: https://github.com/loukasa/graph-coarsening
def LV_nei_summarize(G, ratio):
    N = G.number_of_nodes()
    if N <= 5:
        return G, 0
    k = min(6, G.number_of_nodes()//2)
    if k <= 0:
        k = 1
    n = max(int(np.ceil(N * ratio)), 5)
    ratio = n / N
    G = Graph(nx.adjacency_matrix(G))
    t0 = time.perf_counter()
    C, Gc, Call, Gall = coarsen(G, K=k, r=1-ratio, method='variation_neighborhoods')
    t1 = time.perf_counter()
    # Gc = nx.from_scipy_sparse_array(Gc.A)
    Gc = nx.from_scipy_sparse_array(C @ G.A @ C.T)
    return Gc, t1 - t0


# Source: https://github.com/loukasa/graph-coarsening
def LV_edge_summarize(G, ratio):
    N = G.number_of_nodes()
    if N <= 5:
        return G, 0
    k = min(6, G.number_of_nodes()//2)
    if k <= 0:
        k = 1
    n = max(int(np.ceil(N * ratio)), 5)
    ratio = n / N
    G = Graph(nx.adjacency_matrix(G))
    t0 = time.perf_counter()
    C, Gc, Call, Gall = coarsen(G, K=k, r=1-ratio, method='variation_edges')
    t1 = time.perf_counter()
    # Gc = nx.from_scipy_sparse_array(Gc.A)
    Gc = nx.from_scipy_sparse_array(C @ G.A @ C.T)
    return Gc, t1 - t0
