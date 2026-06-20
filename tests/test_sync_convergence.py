import numpy as np

from src.constants import GlobalConstants
from benchmarks.sync_convergence import (
    _orbital_period,
    _kuramoto,
    _rollout,
    _time_to_threshold,
    _fit_power_law,
    _make_constants,
)


class TestOrbitalPeriod:
    def test__matches_closed_form(self) -> None:
        c = GlobalConstants.create()
        mobility = 1 / (6 * np.pi * c.eta * c.a)
        expected = 2 * np.pi / (mobility * c.driving_force / c.orbit_radius)
        assert np.isclose(_orbital_period(c), expected)


class TestKuramoto:
    def test__identical_phases_is_one(self) -> None:
        centers = np.zeros((3, 4))
        positions = centers + np.array([[1.0], [0.0], [0.0]])  # all at phase 0
        assert np.isclose(_kuramoto(positions, centers), 1.0)

    def test__antiphase_pair_is_zero(self) -> None:
        centers = np.zeros((3, 2))
        positions = centers + np.array([[1.0, -1.0], [0.0, 0.0], [0.0, 0.0]])  # opposite phases
        assert np.isclose(_kuramoto(positions, centers), 0.0, atol=1e-9)


class TestTimeToThreshold:
    def test__returns_period_normalised_crossing(self) -> None:
        history = np.array([0.0, 0.5, 0.95, 0.99])  # crosses 0.9 at index 2
        # dt=period/10 => index 2 is 0.2 periods
        assert np.isclose(_time_to_threshold(history, dt=0.1, period=1.0, threshold=0.9), 0.2)

    def test__nan_if_never_reached(self) -> None:
        history = np.array([0.0, 0.3, 0.5])
        assert np.isnan(_time_to_threshold(history, dt=0.1, period=1.0, threshold=0.9))


class TestFitPowerLaw:
    def test__recovers_quadratic(self) -> None:
        dts = [1e-6, 2e-6, 4e-6, 8e-6]
        errs = [3.0 * dt**2 for dt in dts]  # exact dt^2
        assert np.isclose(_fit_power_law(dts, errs), 2.0, atol=1e-6)


class TestRollout:
    def test__history_shape_and_bounds(self) -> None:
        c = _make_constants(dt=1e-5, steps=20, modulation=0.2, spring_constant=0.001, orbit_separation=6e-6)
        history = _rollout(c, nx=3, ny=3, seed=0, with_noise=False)
        assert history.shape == (21,)
        assert np.all(history >= -1e-9) and np.all(history <= 1 + 1e-9)

    def test__deterministic_without_noise(self) -> None:
        # Drift-only rollouts are reproducible for a fixed seed/initial condition.
        c = _make_constants(dt=1e-5, steps=15, modulation=0.2, spring_constant=0.001, orbit_separation=6e-6)
        a = _rollout(c, nx=3, ny=3, seed=1, with_noise=False)
        b = _rollout(c, nx=3, ny=3, seed=1, with_noise=False)
        np.testing.assert_array_equal(a, b)
