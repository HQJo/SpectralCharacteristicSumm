import glob
import os
import time

import netlsd
import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

from chebyshev_approx import ChebyShevApproxMatrix


def load_dataset(dataset, path='./data'):
    if dataset in ('MUTAG', 'NCI1', 'ENZYMES', 'NCI109', 'PROTEINS', 'PTC_MR'):
        graphs, labels = parse_dataset(dataset)
        labels = labels.astype(int)
        graphs = [nx.from_scipy_sparse_matrix(g) for g in graphs]
    else:
        graphs, labels = _load_data(dataset)
    return graphs, labels


def _load_data(dataset, path="./data"):
    dir_ = os.path.join(path, dataset)
    Gs = []
    for fname in glob.glob(f'{dir_}/G_*.npz'):
        Gs.append(nx.from_scipy_sparse_array(sp.load_npz(fname)))
    labels = np.load(f'{dir_}/labels.npy')
    return Gs, labels


# From Jin's code
def parse_dataset(DS):
    dir_ = './data/bio'
    prefix = dir_ + '/' + DS + '/' + DS
    A = prefix + '_A.txt'
    offsets = np.loadtxt(prefix +'_graph_indicator.txt', dtype=np.int, delimiter=',') - 1
    offs = np.append([0], np.append(np.where((offsets[1:] - offsets[:-1])>0)[0]+1, len(offsets)))
    labels = np.loadtxt(prefix+'_graph_labels.txt', dtype=np.float64).reshape(-1)
    A_data = np.loadtxt(prefix+'_A.txt', dtype=np.int, delimiter=',') - 1
    A_mat = sp.csr_matrix((np.ones(A_data.shape[0]), (A_data[:, 0], A_data[:, 1])), dtype=np.int)

    As = []
    for i in range(1, len(offs)):
        As.append(A_mat[offs[i-1]:offs[i],offs[i-1]:offs[i]])

    am = [np.array(sp.csr_matrix.todense(x.astype(np.float64))) for x in As]
    # am = [sp.csr_matrix(x.astype(np.float64)) for x in As]
    am_corrected = []
    label_corrected = []
    N = len(am)
    for i in range(N):
        d = sum(am[i], 0)
        if not np.any(d == 0):
            am_corrected.append(sp.csr_matrix(am[i]))
            label_corrected.append(labels[i])
    # print(len(am))
    # print(len(am_corrected))
    le = LabelEncoder()
    label_corrected = le.fit_transform(label_corrected)
    return am_corrected, label_corrected


def heat_trace(G, taus = np.logspace(-2, 2, 250)):
    N = G.number_of_nodes()
    L_norm = nx.normalized_laplacian_matrix(G)
    T = 1000
    signature = []
    for t in taus:
        cheby = ChebyShevApproxMatrix(0, 2, 6, lambda x: np.exp(-t * x), 100)
        X = np.random.randn(N, T)
        Y = cheby.approximate(L_norm, X)
        signature.append(np.trace(X.T @ Y) / T)
    signature = np.asarray(signature)
    return signature


def diffusion_feat(G, A, taus=np.logspace(-1, 1, 250)):
    N = G.number_of_nodes()
    L_norm = nx.normalized_laplacian_matrix(G)
    T = 1000
    Z = []
    for t in taus:
        cheby = ChebyShevApproxMatrix(-1e-3, 2+1e-3, 10, lambda x: np.exp(-t * x), 100)
        X = np.random.randn(N, T)
        Y = cheby.approximate(L_norm, X)
        Z.append((X * Y).mean(axis=1))
    Z = np.asarray(Z).T
    return Z


def spectral_moment(G, order=10):
    N = G.number_of_nodes()
    A_norm = (sp.eye(N) - nx.normalized_laplacian_matrix(G)).toarray()
    moments = np.zeros(order)
    for i in range(order):
        moments[i] = np.trace(A_norm ** i) / N
    return moments


def spectral_moment_approx(G, order=10):
    N = G.number_of_nodes()
    A_norm = sp.eye(N) - nx.normalized_laplacian_matrix(G)
    moments = np.zeros(order)
    T = 1000
    X = np.random.randn(N, T)
    Y = X.copy()
    for i in range(order):
        Y = A_norm @ Y
        moments[i] += np.trace(X.T @ Y) / N
    moments /= T
    return moments


def search_and_test(X, Y, n_fold=10, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=random_state)
    param_grid = [
        {'C': np.logspace(-3, 2, 6), 'kernel': ['linear']},
        {'C': np.logspace(-3, 2, 6), 'gamma': list(np.logspace(-3, 2, 6)) + ['auto'], 'kernel': ['rbf']},
    ]
    clf = GridSearchCV(SVC(), param_grid, cv=n_fold, scoring='accuracy', n_jobs=-1)
    clf.fit(X_train, y_train)
    best_params_ = clf.best_params_
    # print(best_params_)
    if best_params_['kernel'] == 'linear':
        clf = SVC(kernel='linear', C=best_params_['C'])
    elif best_params_['kernel'] == 'rbf':
        clf = SVC(kernel='rbf', C=best_params_['C'], gamma=best_params_['gamma'])

    k_fold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=random_state)
    cvs = cross_val_score(clf, X, Y, n_jobs=-1, cv=k_fold)
    acc = cvs.mean()
    return acc

