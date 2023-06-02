import netlsd
import numpy as np
import networkx as nx

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
    print(np.mean(accs))
    return np.mean(accs)


def eval_moment(graphs, summary_graphs):
    loss = []
    i = 0
    for g, gs in zip(graphs, summary_graphs):
        i += 1
        m1 = utils.spectral_moment(g)
        m2 = utils.spectral_moment(gs)
        loss.append(np.linalg.norm(m1 - m2))
    loss = np.array(loss)
    print('Moment loss: ', loss.mean())
    return loss.mean()


def eval_heat_trace(graphs, summary_graphs):
    loss = []
    for g, gs in zip(graphs, summary_graphs):
        h1 = netlsd.heat(g)
        h2 = netlsd.heat(gs)
        loss.append(np.linalg.norm(h1 - h2))
    loss = np.array(loss)
    print('Heat trace loss: ', loss.mean())
    return loss.mean()

