"""Run from the repository root with:  pytest"""
import numpy as np
from scipy.sparse.linalg import spsolve

from poisson import build, jacobi, sor, cg, optimal_omega

N_REF = 15


def reference():
    A, b, _, h = build(N_REF, rhs="ones")
    return A, b, h, spsolve(A.tocsc(), b)


def test_matrix_structure():
    A, _, _, _ = build(7)
    assert A.shape == (49, 49)
    assert abs(A - A.T).max() == 0
    assert A.getnnz(axis=1).max() == 5


def test_discretisation_error_is_second_order():
    errs, hs = [], []
    for n in (7, 15, 31):
        A, b, u_exact, h = build(n)
        errs.append(np.max(np.abs(spsolve(A.tocsc(), b) - u_exact)))
        hs.append(h)
    slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    assert abs(slope - 2) < 0.05


def test_jacobi():
    _, b, h, ref = reference()
    x, _ = jacobi(b, h, tol=1e-10)
    assert np.max(np.abs(x - ref)) < 1e-6


def test_gauss_seidel():
    _, b, h, ref = reference()
    x, _ = sor(b, h, 1.0, tol=1e-10)
    assert np.max(np.abs(x - ref)) < 1e-6


def test_sor():
    _, b, h, ref = reference()
    x, _ = sor(b, h, optimal_omega(h), tol=1e-10)
    assert np.max(np.abs(x - ref)) < 1e-6


def test_cg():
    A, b, h, ref = reference()
    x, _ = cg(A, b, tol=1e-10)
    assert np.max(np.abs(x - ref)) < 1e-6
