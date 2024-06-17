import os
import pickle
import sys
import time
import warnings
from argparse import ArgumentParser
from tqdm import tqdm

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

import baselines
import utils
import evaluation

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", RuntimeWarning)


parser = ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
parser.add_argument('--ratio', type=float, help='Summarization ratio')
parser.add_argument('--method', type=str, help='Summarization method')
parser.add_argument('--output', type=str,
                    default='./output', help='Output directory')
parser.add_argument('--setting', type=str, default='')
parser.add_argument('--seed', type=int, default=0, help='random state')


def print_results(args, moment_loss, heat_loss, spec_loss, wass_loss, acc, time):
    print('-' * 32)
    print(f'Method: {args.method}')
    print(f'Dataset: {args.dataset}')
    print(f'Ratio: {args.ratio}')
    print(f'Moment loss: {moment_loss}')
    print(f'Heat loss: {heat_loss}')
    print(f'Eigen loss: {spec_loss}')
    print(f'Acc: {acc}')
    print(f'Time: {time}')


# Proposed method
def SDSumm(G, args):
    N = G.number_of_nodes()
    if N <= 5:
        return G, 0
    A = nx.to_scipy_sparse_array(G)
    L = nx.normalized_laplacian_matrix(G)
    degs = np.array(A.sum(axis=1)).flatten()
    degs_inv = 1.0 / degs
    degs_inv[np.isinf(degs_inv)] = 0
    D_inv_sqrt = sp.diags(np.power(degs_inv, 0.5))
    t0 = time.perf_counter()
    Z = (sp.eye(N) - L) @ D_inv_sqrt
    n = max(int(np.ceil(N * args.ratio)), 5)
    cluster = AgglomerativeClustering(n_clusters=n, linkage='ward')
    cluster.fit(Z.toarray())
    t1 = time.perf_counter()
    res = cluster.labels_
    P = sp.csr_matrix(([1] * len(res), (np.arange(N), res))).T
    As = P @ A @ P.T
    Gs = nx.from_scipy_sparse_array(As)
    return Gs, t1-t0


def main(args):
    dir_ = os.path.join(args.output, args.dataset)
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    graphs, labels = utils.load_dataset(args.dataset)
    dir_ = os.path.join(dir_, args.method)
    if not os.path.exists(dir_):
        os.mkdir(dir_)
    output_file = os.path.join(dir_, f'{args.method}.pkl')
    if os.path.exists(os.path.join(dir_, 'obj.pkl')):
        summary_graphs = pickle.load(open(output_file, 'rb'))
        spec_dis, moment_loss, heat_loss = evaluation.eval_all(graphs, summary_graphs)
        print_results(args, moment_loss, heat_loss, spec_dis, spec_dis, 0, 0)
        sys.exit(0)

    summary_graphs, times = [], []
    if args.method == 'SDSumm':
        for G in tqdm(graphs):
            Gs, t = SDSumm(G, args)
            summary_graphs.append(Gs)
            times.append(t)
    # spectral clustering
    elif args.method == 'sc':
        for G in tqdm(graphs):
            Gs, t = baselines.sc_summarize(G, args.ratio)
            summary_graphs.append(Gs)
            times.append(t)
    elif args.method == 'mgc':
        for G in tqdm(graphs):
            Gs, t = baselines.mgc(G, args.ratio)
            summary_graphs.append(Gs)
            times.append(t)
    elif args.method == 'sgc':
        for G in tqdm(graphs):
            Gs, t = baselines.sgc(G, args.ratio)
            summary_graphs.append(Gs)
            times.append(t)
    elif args.method == 'graphzoom':
        for G in tqdm(graphs):
            Gs, t = baselines.graphzoom_summarize(G, args.ratio)
            summary_graphs.append(Gs)
            times.append(t)
    elif args.method == 'LV_nei':
        for G in tqdm(graphs):
            Gs, t = baselines.LV_nei_summarize(G, args.ratio)
            summary_graphs.append(Gs)
            times.append(t)
    elif args.method == 'LV_edge':
        for G in tqdm(graphs):
            Gs, t = baselines.LV_edge_summarize(G, args.ratio)
            summary_graphs.append(Gs)
            times.append(t)
    else:
        print("Method not found!")
        sys.exit(1)
    times = np.array(times)
    acc = evaluation.eval_NetLSD(summary_graphs, labels)
    spec_dis, moment_loss, heat_loss = evaluation.eval_all(graphs, summary_graphs)
    print_results(args, moment_loss, heat_loss, spec_dis, spec_dis, acc, times.sum())


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

