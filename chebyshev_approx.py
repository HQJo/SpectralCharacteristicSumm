import numpy as np
import scipy.sparse as ssp


class ChebyshevApprox:
    """Approximate a specific function (defined on a specific range $[a, b]$) by Chebyshev polynomial approximation.
    """
    def __init__(self, a, b, m, f, N) -> None:
        """

        Parameters
        ----------
        a : float
            Minimum of the range.
        b : float
            Maximum of the range.
        m : int
            Order of approximation.
        f : callabl
            Function to be approximated.
        N : int
            Number of sample points in integrals.
        """
        self.a, self.b = a, b
        self.m = m
        l, r = (b-a) / 2, (b+a) / 2
        self.coefs = np.zeros(m+1)

        x = np.cos(np.pi * (np.arange(N) + 0.5) / N)
        if isinstance(f, np.ufunc):
            fx = f(l * x + r)
        else:
            fx = np.array([f(l * t + r) for t in x])
        for i in range(m+1):
            self.coefs[i] = 2.0 / N * np.dot(fx, np.cos(np.pi * i * (np.arange(N) + 0.5) / N))

    def approximate(self, x):
        """Calculate approximate function value $f(x)$.

        Parameters
        ----------
        x : float
            Input value.

        Returns
        -------
        float
            Approximate function value.
        """
        x = 2 * (x - self.a) / (self.b - self.a) - 1
        old, current = 1, x
        res = 0.5 * self.coefs[0] * old + self.coefs[1] * current

        for k in range(2, self.m+1):
            new = 2 * x * current - old
            res += self.coefs[k] * new
            old, current = current, new
        return res


class ChebyShevApproxMatrix(ChebyshevApprox):
    """Approximate $g(L) x$ by Chebyshev polynomial approximation.
    $g(L)$ is defined as $U g(\Lambda) U^T$ where $U \Lambda U^T=L$ is the eigen-decomposition of matrix $L$.
    To avoid performing full eigen-decomposition, a recursive matrix-vector multiplication manner is used.
    """
    def approximate(self, L, x):
        """Approximate $g(L) x$ by Chebyshev polynomial approximation.

        Parameters
        ----------
        L : scipy.sparse.spmatrix
            Matrix whose eigenvalues lies in $[a, b]$.
        x : numpy.ndarray
            Numpy array.

        Returns
        -------
        numpy.ndarray
            Result of $g(L) x$.
        """
        N = L.shape[0]
        L = (2 * L - (self.b + self.a) * ssp.eye(N)) / (self.b - self.a)
        # L = 2 / (self.b-self.a) * (L - self.a * ssp.eye(N)) - ssp.eye(N)
        old, current = x, L @ x
        res = 0.5 * self.coefs[0] * old + self.coefs[1] * current

        for k in range(2, self.m+1):
            new = 2 * L @ current - old
            res += self.coefs[k] * new
            old, current = current, new
        return res
