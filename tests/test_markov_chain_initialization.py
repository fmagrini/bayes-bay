import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class _UtilsStub(types.ModuleType):
    def __getattr__(self, name):
        def _stub(*args, **kwargs):  # pragma: no cover - placeholder implementation
            raise NotImplementedError(name)

        return _stub


utils_stub = _UtilsStub("bayesbay._utils_1d")
utils_stub.inverse_covariance = lambda std, correlation, n: np.eye(n)
sys.modules.setdefault("bayesbay._utils_1d", utils_stub)

import bayesbay._markov_chain as mc_mod
from bayesbay._markov_chain import MarkovChain
from bayesbay._state import ParameterSpaceState, State


def _make_state(label: str) -> State:
    param_state = ParameterSpaceState(1, {"value": np.array([0.0])})
    return State({"model": param_state}, extra_storage={"label": label})


class SequentialParameterization:
    def __init__(self, states):
        self.states = list(states)
        if not self.states:
            raise ValueError("states must not be empty")
        self.index = 0
        self.parameter_spaces = {}

    @property
    def perturbation_funcs(self):
        return []

    @property
    def perturbation_weights(self):
        return []

    def initialize(self):
        if self.index < len(self.states):
            state = self.states[self.index]
            self.index += 1
            return state
        return self.states[-1]


class DummyLogLikelihood:
    def __init__(self, failing_labels):
        self.failing_labels = set(failing_labels)
        self.initialize_calls = 0
        self.forward_calls = 0
        self.fwd_functions = [self._forward]
        self.log_like_ratio_func = None
        self.log_like_func = None

    def initialize(self, state):
        self.initialize_calls += 1

    def _forward(self, state):
        self.forward_calls += 1
        if state.extra_storage.get("label") in self.failing_labels:
            raise RuntimeError("forward failed")
        return np.array([0.0])

    def log_likelihood_ratio(self, old_state, new_state, temperature=1):
        if new_state.extra_storage.get("label") in self.failing_labels:
            raise RuntimeError("forward failed")
        return 0.0


def test_initialization_retries_until_valid_state():
    bad_states = [_make_state("bad1"), _make_state("bad2")]
    good_state = _make_state("good")
    parameterization = SequentialParameterization(bad_states + [good_state])
    log_likelihood = DummyLogLikelihood({"bad1", "bad2"})

    chain = MarkovChain(
        id=0,
        parameterization=parameterization,
        log_likelihood=log_likelihood,
        perturbation_funcs=[],
        perturbation_weights=[],
    )

    assert chain.current_state.extra_storage["label"] == "good"
    assert log_likelihood.initialize_calls == 3
    assert log_likelihood.forward_calls == 3


def test_initialization_raises_after_exhaustion(monkeypatch):
    bad_state = _make_state("bad")
    parameterization = SequentialParameterization([bad_state])
    log_likelihood = DummyLogLikelihood({"bad"})
    monkeypatch.setattr(mc_mod, "_MAX_INITIAL_STATE_ATTEMPTS", 3)

    with pytest.raises(RuntimeError) as exc:
        MarkovChain(
            id=0,
            parameterization=parameterization,
            log_likelihood=log_likelihood,
            perturbation_funcs=[],
            perturbation_weights=[],
        )

    assert "Unable to initialize" in str(exc.value)
    assert log_likelihood.initialize_calls == 3
    assert log_likelihood.forward_calls == 3

