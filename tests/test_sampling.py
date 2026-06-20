import numpy as np

from src.sampling import (
    EighSampler,
    ChebyshevSampler,
    _spectral_bounds,
    _sqrt_chebyshev_coefficients,
    _chebyshev_matrix_sqrt_apply,
)


def _random_spd(dim: int, seed: int, floor: float = 1.0) -> np.ndarray:
    """A symmetric positive-definite matrix with eigenvalues >= floor."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((dim, dim))
    return a @ a.T + floor * np.identity(dim)


def _spd_with_spectrum(eigenvalues: np.ndarray, seed: int) -> np.ndarray:
    """A symmetric matrix with exactly the given eigenvalues (random eigenvectors)."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((len(eigenvalues), len(eigenvalues))))
    return (q * eigenvalues) @ q.T


class TestSpectralBounds:
    def test__estimates_isolated_extremes(self) -> None:
        # Power iteration converges quickly when the extreme eigenvalues are isolated;
        # (the shifted iteration for lambda_min is slow only for clustered spectra, which
        # the sampler tolerates via its downward bias on the lower bound).
        eigenvalues = np.concatenate([[1.0], np.linspace(40.0, 60.0, 18), [100.0]])
        cov = _spd_with_spectrum(eigenvalues, seed=0)
        lambda_min, lambda_max = _spectral_bounds(cov, iterations=100)

        np.testing.assert_allclose(lambda_max, 100.0, rtol=2e-2)
        np.testing.assert_allclose(lambda_min, 1.0, rtol=1e-1)


class TestChebyshevMatrixSqrt:
    def test__approximates_symmetric_sqrt(self) -> None:
        cov = _random_spd(30, seed=1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        z = np.random.default_rng(2).standard_normal(30)
        exact = eigenvectors @ (np.sqrt(eigenvalues) * (eigenvectors.T @ z))  # symmetric sqrt(cov) @ z

        lower, upper = eigenvalues.min() * 0.9, eigenvalues.max() * 1.1
        coeffs = _sqrt_chebyshev_coefficients(lower, upper, degree=60)
        approx = _chebyshev_matrix_sqrt_apply(cov, z, coeffs, lower, upper)

        np.testing.assert_allclose(approx, exact, rtol=1e-4, atol=1e-6)

    def test__higher_degree_is_more_accurate(self) -> None:
        cov = _random_spd(30, seed=3)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        z = np.random.default_rng(4).standard_normal(30)
        exact = eigenvectors @ (np.sqrt(eigenvalues) * (eigenvectors.T @ z))
        lower, upper = eigenvalues.min() * 0.9, eigenvalues.max() * 1.1

        def error(degree: int) -> float:
            coeffs = _sqrt_chebyshev_coefficients(lower, upper, degree)
            approx = _chebyshev_matrix_sqrt_apply(cov, z, coeffs, lower, upper)
            return float(np.linalg.norm(approx - exact) / np.linalg.norm(exact))

        assert error(40) < error(10)


def _empirical_covariance(sampler, cov: np.ndarray, draws: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    samples = np.array([sampler.sample(cov) for _ in range(draws)])
    return np.cov(samples, rowvar=False)


class TestEighSampler:
    def test__reproduces_covariance(self) -> None:
        cov = _random_spd(6, seed=5)
        empirical = _empirical_covariance(EighSampler(), cov, draws=20000, seed=0)
        np.testing.assert_allclose(empirical, cov, atol=0.15 * cov.max())


class TestChebyshevSampler:
    def test__reproduces_covariance(self) -> None:
        cov = _random_spd(6, seed=6)
        empirical = _empirical_covariance(ChebyshevSampler(degree=40), cov, draws=20000, seed=0)
        np.testing.assert_allclose(empirical, cov, atol=0.15 * cov.max())

    def test__matches_eigh_distribution(self) -> None:
        # The two samplers should produce the same covariance (different factors, same N(0, cov)).
        cov = _random_spd(6, seed=7)
        eigh_cov = _empirical_covariance(EighSampler(), cov, draws=20000, seed=1)
        cheb_cov = _empirical_covariance(ChebyshevSampler(degree=40), cov, draws=20000, seed=1)
        np.testing.assert_allclose(cheb_cov, eigh_cov, atol=0.15 * cov.max())
