import os
import sys
import time
import warnings
from argparse import ArgumentParser

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

import baselines
import utils
import evaluation
import testing

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning)


parser = ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
parser.add_argument('--ratio', type=float, help='Summarization ratio')


def spectral_summarize(G, args):
    N = G.number_of_nodes()
    if N < 5:
        return G, 0
    A = nx.to_scipy_sparse_matrix(G)
    L = nx.normalized_laplacian_matrix(G)
    degs = np.array(A.sum(axis=1)).flatten()
    D_inv_sqrt = sp.diags(np.power(degs, -0.5))
    t0 = time.perf_counter()
    Z = D_inv_sqrt @ (sp.eye(N) - L)
    n = max(int(np.ceil(N * args.ratio)), 5)

    cluster = AgglomerativeClustering(n_clusters=n, linkage='ward')
    cluster.fit(Z.toarray())
    t1 = time.perf_counter()
    res = cluster.labels_
    P = sp.csr_matrix(([1] * len(res), (np.arange(N), res))).T
    As = P @ A @ P.T
    Gs = nx.from_scipy_sparse_matrix(As)
    return Gs, t1-t0


def main(args):
    graphs, labels = utils.load_dataset(args.dataset)
    summary_graphs, times = [], []
    for G in graphs:
        Gs, t = spectral_summarize(G, args)
        summary_graphs.append(Gs)
        times.append(t)
    times = np.array(times)
    print('Average time: ', times.mean())
    moment_loss = evaluation.eval_moment(graphs, summary_graphs)
    heat_loss = evaluation.eval_heat_trace(graphs, summary_graphs)
    acc = evaluation.eval_NetLSD(summary_graphs, labels)
    print("Mean acc: ", acc)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

