"""2D Poisson equation: finite-difference setup and iterative solvers (Jacobi, SOR/Gauss-Seidel, CG).

Every solver returns (solution, relative-residual history).
Jacobi and SOR update a padded grid directly (no matrix assembled); CG uses the sparse matrix.
"""
import numpy as np
import scipy.sparse as sp

# optional numba JIT compilation (same results without it, but much slower)
try:
    from numba import njit
except ImportError:
    def njit(f):
        return f


def build(N, rhs="manufactured"):
    """Return A (sparse, N^2 x N^2), b, u_exact (flattened) and h for an N x N interior grid."""
    h = 1.0 / (N + 1)
    x = h * np.arange(1, N + 1)
    T = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(N, N)) / h**2
    I = sp.identity(N)
    A = (sp.kron(I, T) + sp.kron(T, I)).tocsr()
    X, Y = np.meshgrid(x, x, indexing="ij")
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    b = 2 * np.pi**2 * u_exact if rhs == "manufactured" else np.ones_like(u_exact)
    return A, b.ravel(), u_exact.ravel(), h


@njit
def _residual_norm(u, B, h2, N):
    s = 0.0
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            r = B[i - 1, j - 1] - (4 * u[i, j] - u[i - 1, j] - u[i + 1, j]
                                   - u[i, j - 1] - u[i, j + 1]) / h2
            s += r * r
    return np.sqrt(s)


@njit
def _sor(B, h, omega, tol, maxit):
    N = B.shape[0]
    h2 = h * h
    u = np.zeros((N + 2, N + 2))
    bn = np.sqrt(np.sum(B * B))
    hist = np.zeros(maxit + 1)
    for k in range(maxit + 1):
        hist[k] = _residual_norm(u, B, h2, N) / bn
        if hist[k] < tol:
            return u[1:-1, 1:-1].copy().ravel(), hist[:k + 1]
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                gs = (h2 * B[i - 1, j - 1] + u[i - 1, j] + u[i + 1, j]
                      + u[i, j - 1] + u[i, j + 1]) / 4
                u[i, j] += omega * (gs - u[i, j])
    return u[1:-1, 1:-1].copy().ravel(), hist


@njit
def _jacobi(B, h, tol, maxit):
    N = B.shape[0]
    h2 = h * h
    u = np.zeros((N + 2, N + 2))
    v = np.zeros((N + 2, N + 2))
    bn = np.sqrt(np.sum(B * B))
    hist = np.zeros(maxit + 1)
    for k in range(maxit + 1):
        hist[k] = _residual_norm(u, B, h2, N) / bn
        if hist[k] < tol:
            return u[1:-1, 1:-1].copy().ravel(), hist[:k + 1]
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                v[i, j] = (h2 * B[i - 1, j - 1] + u[i - 1, j] + u[i + 1, j]
                           + u[i, j - 1] + u[i, j + 1]) / 4
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                u[i, j] = v[i, j]
    return u[1:-1, 1:-1].copy().ravel(), hist


def jacobi(b, h, tol=1e-10, maxit=200_000):
    N = int(round(np.sqrt(b.size)))
    return _jacobi(b.reshape(N, N), h, tol, maxit)


def sor(b, h, omega, tol=1e-10, maxit=200_000):
    """SOR; omega = 1 gives Gauss-Seidel."""
    N = int(round(np.sqrt(b.size)))
    return _sor(b.reshape(N, N), h, omega, tol, maxit)


def optimal_omega(h):
    return 2.0 / (1.0 + np.sin(np.pi * h))


def cg(A, b, tol=1e-10, maxit=20_000):
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rs = r @ r
    bn = np.linalg.norm(b)
    hist = [np.sqrt(rs) / bn]
    for _ in range(maxit):
        if hist[-1] < tol:
            break
        Ap = A @ p
        alpha = rs / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = r @ r
        hist.append(np.sqrt(rs_new) / bn)
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x, np.array(hist)
