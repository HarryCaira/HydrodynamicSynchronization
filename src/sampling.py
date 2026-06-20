"""Samplers that draw a single correlated Gaussian displacement N(0, covariance).

The Brownian step needs x = sqrt(Sigma) @ z with z ~ N(0, I), where Sigma = 2 dt D is
the 3n x 3n diffusion-tensor covariance. Two strategies are provided:

- EighSampler: factor Sigma exactly with a symmetric eigendecomposition. Robust and
  simple, but O((3n)^3) per step - the dominant cost at large n.
- ChebyshevSampler: the Fixman method. Approximate sqrt(Sigma) @ z with a Chebyshev
  polynomial in Sigma, evaluated via the Clenshaw recurrence using only matrix-vector
  products. Cost is O(degree * (3n)^2), which wins once 3n is large.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.polynomial import chebyshev as _chebyshev


class GaussianSampler(ABC):
    @abstractmethod
    def sample(self, covariance: np.ndarray) -> np.ndarray:
        """Return a single draw from N(0, covariance) as a 1-D array of length covariance.shape[0]."""


class EighSampler(GaussianSampler):
    """
    Exact sampler. Factor Sigma = Q diag(w) Q^T with a symmetric eigendecomposition;
    Q diag(sqrt(w)) is a valid square-root factor, so x = Q (sqrt(w) * z) has covariance
    Sigma. Negative eigenvalues are clipped to zero so a non-positive-definite covariance
    still yields the nearest PSD sample. Cost is O(m^3) for m = covariance.shape[0].
    """

    def sample(self, covariance: np.ndarray) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        sqrt_eigenvalues = np.sqrt(np.clip(eigenvalues, 0.0, None))
        return eigenvectors @ (sqrt_eigenvalues * np.random.standard_normal(covariance.shape[0]))


def _power_iteration(matvec, dim: int, iterations: int, seed: int) -> float:
    """Estimate the largest eigenvalue of a symmetric operator via power iteration."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    v /= np.linalg.norm(v)
    for _ in range(iterations):
        w = matvec(v)
        norm = np.linalg.norm(w)
        if norm == 0.0:
            return 0.0
        v = w / norm
    return float(v @ matvec(v))


def _spectral_bounds(covariance: np.ndarray, iterations: int) -> tuple[float, float]:
    """Estimate (lambda_min, lambda_max) of a symmetric matrix via power iteration.

    lambda_max is the largest eigenvalue directly; lambda_min uses that the largest
    eigenvalue of (lambda_max I - Sigma) equals lambda_max - lambda_min (no shifted
    matrix is formed - the shift is applied inside the matvec).
    """
    dim = covariance.shape[0]
    lambda_max = _power_iteration(lambda v: covariance @ v, dim, iterations, seed=0)
    spread = _power_iteration(lambda v: lambda_max * v - covariance @ v, dim, iterations, seed=1)
    return lambda_max - spread, lambda_max


def _sqrt_chebyshev_coefficients(lower: float, upper: float, degree: int) -> np.ndarray:
    """Chebyshev coefficients approximating sqrt on [lower, upper]."""

    def sqrt_on_interval(x: np.ndarray) -> np.ndarray:
        eigenvalue = 0.5 * ((upper - lower) * x + (upper + lower))  # map [-1, 1] -> [lower, upper]
        return np.sqrt(np.clip(eigenvalue, 0.0, None))

    return _chebyshev.chebinterpolate(sqrt_on_interval, degree)


def _chebyshev_matrix_sqrt_apply(matrix: np.ndarray, z: np.ndarray, coefficients: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Evaluate (sum_k c_k T_k(M)) z by the Clenshaw recurrence, where M maps the
    spectrum [lower, upper] of `matrix` onto [-1, 1]. Uses only matrix-vector products."""
    scale, shift = 2.0 / (upper - lower), (upper + lower) / (upper - lower)

    def mapped(vector: np.ndarray) -> np.ndarray:  # ((2 matrix - (lower+upper) I) / (upper - lower)) @ vector
        return scale * (matrix @ vector) - shift * vector

    b_kplus1 = np.zeros_like(z)
    b_kplus2 = np.zeros_like(z)
    for coefficient in coefficients[:0:-1]:  # k = degree, ..., 1
        b_k = coefficient * z + 2 * mapped(b_kplus1) - b_kplus2
        b_kplus2, b_kplus1 = b_kplus1, b_k
    return coefficients[0] * z + mapped(b_kplus1) - b_kplus2


class ChebyshevSampler(GaussianSampler):
    """
    Fixman sampler. Approximate sqrt(Sigma) @ z with a Chebyshev polynomial in Sigma,
    evaluated via the Clenshaw recurrence using only matrix-vector products, so no
    eigendecomposition is formed. Cost is O(degree * m^2) for m = covariance.shape[0].

    Spectral bounds for the Chebyshev interval are estimated by power iteration. sqrt is
    only well defined for a (numerically) positive semi-definite covariance, which suits
    the Rotne-Prager tensor (whose covariance is positive-definite); ill-conditioned
    covariances need a larger degree for the same accuracy.
    """

    def __init__(self, degree: int = 50, power_iterations: int = 20) -> None:
        self._degree = degree
        self._power_iterations = power_iterations

    def sample(self, covariance: np.ndarray) -> np.ndarray:
        z = np.random.standard_normal(covariance.shape[0])
        lambda_min, lambda_max = _spectral_bounds(covariance, self._power_iterations)

        # Bracket the true spectrum: bias the lower bound down (safe - an over-wide
        # interval just needs more degree) and the upper bound up.
        lower = max(lambda_min * 0.5, lambda_max * 1e-8)
        upper = lambda_max * 1.1
        if upper - lower <= lower * 1e-12:  # degenerate (near-constant) spectrum
            return np.sqrt(max(0.5 * (lower + upper), 0.0)) * z

        coefficients = _sqrt_chebyshev_coefficients(lower, upper, self._degree)
        return _chebyshev_matrix_sqrt_apply(covariance, z, coefficients, lower, upper)
