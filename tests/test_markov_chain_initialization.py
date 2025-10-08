import numpy as np
import bayesbay as bb


class _FailingForward:
    """Forward function that fails for the first N calls"""

    def __init__(self, fail_count):
        self.fail_count = fail_count
        self.call_count = 0

    def __call__(self, state):
        self.call_count += 1
        print(f"Forward called {self.call_count} times")
        if self.call_count <= self.fail_count:
            raise RuntimeError("Forward failed intentionally")
        return np.array([1.0])


class _FailingLogLike:
    """Log likelihood function that fails for the first N calls"""

    def __init__(self, fail_count):
        self.fail_count = fail_count
        self.call_count = 0

    def __call__(self, state):
        self.call_count += 1
        print(f"Log likelihood called {self.call_count} times")
        if self.call_count <= self.fail_count:
            raise RuntimeError("Log likelihood failed intentionally")
        return 0.0


class _FailingLogLikeRatio:
    """Log likelihood ratio function that fails for the first N calls"""

    def __init__(self, fail_count):
        self.fail_count = fail_count
        self.call_count = 0

    def __call__(self, old_state, new_state):
        self.call_count += 1
        print(f"Log likelihood ratio called {self.call_count} times")
        if self.call_count <= self.fail_count:
            raise RuntimeError("Log likelihood ratio failed intentionally")
        return 0.0


def _setup_parameterization():
    param = bb.prior.UniformPrior("param", 0, 1, 0.1)
    return bb.parameterization.Parameterization(
        bb.discretization.Voronoi1D(
            name="voronoi",
            vmin=0,
            vmax=1,
            perturb_std=0.1,
            n_dimensions=None,
            n_dimensions_min=1,
            n_dimensions_max=3,
            parameters=[param],
        )
    )


def test_initialization_with_forward_functions():
    """Test retry logic with targets + fwd_functions"""
    print("\n=== Test 1: With forward functions (targets + fwd_functions) ===")
    forward = _FailingForward(fail_count=2)
    parameterization = _setup_parameterization()
    target = bb.likelihood.Target("data", np.array([1.0]), covariance_mat_inv=1.0)
    log_likelihood = bb.likelihood.LogLikelihood([target], [forward])

    inversion = bb.BayesianInversion(
        parameterization=parameterization,
        log_likelihood=log_likelihood,
        n_chains=1,
    )

    print(f"✓ Initialization succeeded after {forward.call_count} attempts")
    assert forward.call_count == 3, f"Expected 3 calls, got {forward.call_count}"
    print("✓ Test passed!\n")


def test_initialization_with_log_like_func():
    """Test retry logic with log_like_func"""
    print("=== Test 2: With log_like_func ===")
    log_like = _FailingLogLike(fail_count=2)
    parameterization = _setup_parameterization()
    log_likelihood = bb.likelihood.LogLikelihood(log_like_func=log_like)

    inversion = bb.BayesianInversion(
        parameterization=parameterization,
        log_likelihood=log_likelihood,
        n_chains=1,
    )

    print(f"✓ Initialization succeeded after {log_like.call_count} attempts")
    assert log_like.call_count == 3, f"Expected 3 calls, got {log_like.call_count}"
    print("✓ Test passed!\n")


def test_initialization_with_log_like_ratio_func():
    """Test retry logic with log_like_ratio_func"""
    print("=== Test 3: With log_like_ratio_func ===")
    log_like_ratio = _FailingLogLikeRatio(fail_count=2)
    parameterization = _setup_parameterization()
    log_likelihood = bb.likelihood.LogLikelihood(log_like_ratio_func=log_like_ratio)

    inversion = bb.BayesianInversion(
        parameterization=parameterization,
        log_likelihood=log_likelihood,
        n_chains=1,
    )

    print(f"✓ Initialization succeeded after {log_like_ratio.call_count} attempts")
    assert (
        log_like_ratio.call_count == 3
    ), f"Expected 3 calls, got {log_like_ratio.call_count}"
    print("✓ Test passed!\n")


def test_initialization_fails_after_max_attempts():
    """Test that initialization raises error after too many failures"""
    print("=== Test 4: Raises error after max attempts ===")
    forward = _FailingForward(fail_count=1000)
    parameterization = _setup_parameterization()
    target = bb.likelihood.Target("data", np.array([1.0]), covariance_mat_inv=1.0)
    log_likelihood = bb.likelihood.LogLikelihood([target], [forward])

    try:
        inversion = bb.BayesianInversion(
            parameterization=parameterization,
            log_likelihood=log_likelihood,
            n_chains=1,
        )
        print("✗ Test failed: Should have raised RuntimeError")
        assert False
    except RuntimeError as e:
        print(f"✓ Got expected error: {e}")
        assert "Unable to initialize" in str(e)
        print("✓ Test passed!\n")


if __name__ == "__main__":
    test_initialization_fails_after_max_attempts()
    test_initialization_with_forward_functions()
    test_initialization_with_log_like_func()
    test_initialization_with_log_like_ratio_func()
    print("=" * 50)
    print("All tests passed! ✓")
