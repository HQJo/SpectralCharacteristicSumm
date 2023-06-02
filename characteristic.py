import numpy as np
import scipy.sparse as sp
import networkx as nx
from sklearn.preprocessing import normalize

import utils


def get_P_matrix(N, a, b):
    rows, cols = [0, 0], [a, b]
    idx = 1
    for i in range(N):
        if i == a or i == b:
            continue
        rows.append(idx)
        cols.append(i)
        idx += 1
    P = sp.csr_matrix(([1] * N, (rows, cols)), shape=(N-1, N))
    return P


def heat_trace_loss(G, a, b):
    degs = nx.laplacian_matrix(G).diagonal()
    L = nx.normalized_laplacian_matrix(G)
    N = L.shape[0]
    A_norm = sp.eye(N) - L
    eigs = np.linalg.eigvalsh(A_norm.toarray())
    P = get_P_matrix(N, a, b)
    R = (P @ sp.diags(np.sqrt(degs))).T
    R = normalize(R, axis=0)
    A_r_norm = R @ R.T @ A_norm @ R @ R.T
    eigs2 = np.linalg.eigvalsh(A_r_norm.toarray())
    lam = 1-eigs
    lam2 = 1-eigs2

    deg_inv_sqrt = np.power(degs, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0
    M = sp.diags(deg_inv_sqrt) @ A_norm

    times = np.logspace(-2, 1, 100)
    y1s = []
    y2s = []
    y3s = []
    tmp = sp.linalg.norm(A_norm-A_r_norm) ** 2
    tmp2 = sp.linalg.norm(M[a] - M[b])
    for t in times:
        # Two-order approximation
        y1 = np.exp(-t) * (tmp * (t**2 / 2) + t * np.sum(eigs - eigs2))
        # Keep only the two-order term
        y2 = np.exp(-t) * (tmp * (t**2 / 2))
        # Our metric
        y3 = np.exp(-t) * (2 * degs[a] * degs[b] / (degs[a] + degs[b])) * tmp2 * (t**2 / 2)
        y1s.append(y1)
        y2s.append(y2)
        y3s.append(y3)
    y1s = np.array(y1s)
    y2s = np.array(y2s)
    y3s = np.array(y3s)
    return y1s, y2s, y3s


def ER_model():
    Ns = [100, 1000]
    results = {}
    for N in Ns:
        results[N] = {}
        g_approx1, g_approx2, g_approx3 = [], [], []
        for i in range(100):
            G = nx.gnp_random_graph(N, 0.20)
            approx1, approx2, approx3 = [], [], []
            for j in range(100):
                a, b = np.random.randint(0, N, 2)
                while a == b:
                    a, b = np.random.randint(0, N, 2)
                y1, y2, y3 = heat_trace_loss(G, a, b)
                approx1.append(y1)
                approx2.append(y2)
                approx3.append(y3)
            g_approx1.append(np.array(approx1).mean(axis=0))
            g_approx2.append(np.array(approx2).mean(axis=0))
            g_approx3.append(np.array(approx3).mean(axis=0))
        results[N]['y1'] = np.array(g_approx1).mean(axis=0)
        results[N]['y2'] = np.array(g_approx2).mean(axis=0)
        results[N]['y3'] = np.array(g_approx3).mean(axis=0)
    return results

def BA_model():
    Ns = [100, 1000]
    results = {}
    for N in Ns:
        results[N] = {}
        g_approx1, g_approx2, g_approx3 = [], [], []
        for i in range(100):
            G = nx.barabasi_albert_graph(N, 10)
            approx1, approx2, approx3 = [], [], []
            for j in range(100):
                a, b = np.random.randint(0, N, 2)
                while a == b:
                    a, b = np.random.randint(0, N, 2)
                y1, y2, y3 = heat_trace_loss(G, a, b)
                approx1.append(y1)
                approx2.append(y2)
                approx3.append(y3)
            g_approx1.append(np.array(approx1).mean(axis=0))
            g_approx2.append(np.array(approx2).mean(axis=0))
            g_approx3.append(np.array(approx3).mean(axis=0))
        results[N]['y1'] = np.array(g_approx1).mean(axis=0)
        results[N]['y2'] = np.array(g_approx2).mean(axis=0)
        results[N]['y3'] = np.array(g_approx3).mean(axis=0)
    return results


def real_world(dataset):
    graphs, labels = utils.load_dataset(dataset)
    results = {}
    g_approx1, g_approx2, g_approx3 = [], [], []
    for G in graphs:
        N = G.number_of_nodes()
        if N <= 2:
            continue
        approx1, approx2, approx3 = [], [], []
        upper_bound = N * (N-1) // 2
        for j in range(min(10, upper_bound)):
            a, b = np.random.randint(0, N, 2)
            while a == b:
                a, b = np.random.randint(0, N, 2)
            y1, y2, y3 = heat_trace_loss(G, a, b)
            approx1.append(y1)
            approx2.append(y2)
            approx3.append(y3)
        g_approx1.append(np.array(approx1).mean(axis=0))
        g_approx2.append(np.array(approx2).mean(axis=0))
        g_approx3.append(np.array(approx3).mean(axis=0))
    results['y1'] = np.array(g_approx1).mean(axis=0)
    results['y2'] = np.array(g_approx2).mean(axis=0)
    results['y3'] = np.array(g_approx3).mean(axis=0)
    return results

