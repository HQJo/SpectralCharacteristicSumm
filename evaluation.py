import netlsd
import numpy as np
import networkx as nx
from scipy.stats import wasserstein_distance

import utils


def eval_NetLSD(graphs, labels, n_fold=10, random_state=42, **kwargs):
    X = []
    for G in graphs:
        X.append(netlsd.heat(G))
    X = np.asarray(X)
    accs = []
    for i in range(10):
        best_acc = utils.search_and_test(X, labels, n_fold, random_state=i, **kwargs)
        accs.append(best_acc)
    accs = np.array(accs)
    print('Mean acc:', accs.mean())
    return accs.mean()


def eval_spectral_dis(graphs, summary_graphs):
    loss = []
    for g, gs in zip(graphs, summary_graphs):
        eig1 = nx.normalized_laplacian_spectrum(g)
        eig2 = nx.normalized_laplacian_spectrum(gs)
        eig2 = np.concatenate([eig2, [1] * (len(eig1) - len(eig2))])
        eig1, eig2 = np.sort(eig1), np.sort(eig2)
        loss.append(np.mean(np.abs(eig1-eig2)))
    loss = np.array(loss)
    print('Spectral distance loss: ', loss.mean())
    return loss.mean()


def eval_wasserstein(graphs, summary_graphs):
    loss = []
    for g, gs in zip(graphs, summary_graphs):
        eig1 = nx.normalized_laplacian_spectrum(g)
        eig2 = nx.normalized_laplacian_spectrum(gs)
        eig2 = np.concatenate([eig2, [1] * (len(eig1) - len(eig2))])
        eig1, eig2 = np.sort(eig1), np.sort(eig2)
        loss.append(wasserstein_distance(eig1, eig2))
    loss = np.array(loss)
    print('Wasserstein distance loss: ', loss.mean())
    return loss.mean()


def eval_moment(graphs, summary_graphs):
    loss = []
    i = 0
    for g, gs in zip(graphs, summary_graphs):
        i += 1
        m1 = utils.spectral_moment(g) * g.number_of_nodes()
        m2 = utils.spectral_moment(gs) * gs.number_of_nodes()
        loss.append(np.linalg.norm(m1 - m2))
    loss = np.array(loss)
    print('Moment loss: ', loss.mean())
    return loss.mean()


def eval_heat_trace(graphs, summary_graphs):
    loss = []
    ts = np.logspace(-1, 1, 100)
    for g, gs in zip(graphs, summary_graphs):
        h1 = netlsd.heat(g, ts)
        h2 = netlsd.heat(gs, ts)
        h2 += np.exp(-ts) * (g.number_of_nodes() - gs.number_of_nodes())
        loss.append(np.linalg.norm(h1 - h2) / len(ts))
    loss = np.array(loss)
    print('Heat trace loss: ', loss.mean())
    return loss.mean()


def eval_all(graphs, summary_graphs):
    ts = np.geomspace(0.025, 2.5, 100)
    eig_loss, moment_loss, heat_loss = [], [], []
    for g, gs in zip(graphs, summary_graphs):
        if g.number_of_nodes() == gs.number_of_nodes():
            continue

        eig1 = nx.normalized_laplacian_spectrum(g)
        eig2 = nx.normalized_laplacian_spectrum(gs)
        eig3 = np.concatenate([eig2, [1] * (len(eig1) - len(eig2))])
        eig1, eig3 = np.sort(eig1), np.sort(eig3)
        eig_loss.append(np.mean(np.abs(eig1-eig3)))

        mu1, mu2 = 1-eig1, 1-eig2
        moment1 = np.power(mu1, np.arange(5).reshape(-1, 1)).sum(axis=1)
        moment2 = np.power(mu2, np.arange(5).reshape(-1, 1)).sum(axis=1)
        moment_loss.append(np.mean(np.abs(moment1-moment2)))

        h1 = np.exp(-ts.reshape(-1, 1) * eig1).sum(axis=1)
        h2 = np.exp(-ts.reshape(-1, 1) * eig3).sum(axis=1)
        heat_loss.append(np.mean(np.abs(h1-h2)))
    eig_loss = np.mean(eig_loss)
    moment_loss = np.mean(moment_loss)
    heat_loss = np.mean(heat_loss)
    print('Spectral distance loss: ', eig_loss)
    print('Moment loss: ', moment_loss)
    print('Heat loss: ', heat_loss)

    return eig_loss, moment_loss, heat_loss
