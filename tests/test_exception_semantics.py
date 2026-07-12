import pytest
import numpy as np

from bayesbay import BaseBayesianInversion, BaseMarkovChain, State
from bayesbay.exceptions import ForwardException, InvalidProposalException
from bayesbay.likelihood import LogLikelihood, Target


class _ConstantLogLikelihood:
    def log_likelihood_ratio(self, old_state, new_state, temperature):
        return 0.0


class _FailingLogLikelihood:
    def __init__(self, exc):
        self.exc = exc

    def log_likelihood_ratio(self, old_state, new_state, temperature):
        raise self.exc


def _chain(perturb, log_likelihood=None, on_forward_error="reject"):
    chain = BaseMarkovChain(
        id=0,
        starting_state=0,
        perturbation_funcs=[perturb],
        perturbation_weights=[1],
        log_likelihood=log_likelihood or _ConstantLogLikelihood(),
        save_dpred=False,
        on_forward_error=on_forward_error,
    )
    chain.save_current_iteration = False
    return chain


def test_invalid_perturbation_is_one_counted_rejection():
    def perturb(state):
        raise InvalidProposalException("outside support")

    chain = _chain(perturb)
    chain._next_iteration()

    assert chain.current_state == 0
    assert chain.statistics["n_proposed_models_total"] == 1
    assert chain.statistics["n_accepted_models_total"] == 0
    assert chain.statistics["exceptions"]["InvalidProposalException"] == 1
    assert chain.statistics["n_proposed_models"]["perturb"] == 1


def test_invalid_forward_model_is_one_counted_rejection():
    chain = _chain(
        lambda state: (state + 1, 0.0),
        _FailingLogLikelihood(InvalidProposalException("invalid model")),
    )
    chain._next_iteration()

    assert chain.current_state == 0
    assert chain.statistics["n_proposed_models_total"] == 1
    assert chain.statistics["n_accepted_models_total"] == 0
    assert chain.statistics["exceptions"]["InvalidProposalException"] == 1


def test_unexpected_forward_error_reject_policy():
    chain = _chain(
        lambda state: (state + 1, 0.0),
        _FailingLogLikelihood(ForwardException(RuntimeError("solver failed"))),
    )
    chain._next_iteration()

    assert chain.current_state == 0
    assert chain.statistics["n_proposed_models_total"] == 1
    assert chain.statistics["n_accepted_models_total"] == 0
    assert chain.statistics["exceptions"]["ForwardException"] == 1


def test_unexpected_forward_error_raise_policy():
    chain = _chain(
        lambda state: (state + 1, 0.0),
        _FailingLogLikelihood(ForwardException(RuntimeError("solver failed"))),
        on_forward_error="raise",
    )
    with pytest.raises(ForwardException, match="solver failed"):
        chain._next_iteration()

    assert chain.current_state == 0
    assert chain.statistics["n_proposed_models_total"] == 0
    assert chain.statistics["exceptions"]["ForwardException"] == 1


def test_invalid_forward_error_policy_is_rejected_at_construction():
    with pytest.raises(ValueError, match="on_forward_error"):
        _chain(lambda state: (state, 0.0), on_forward_error="retry")


def test_invalid_proposal_passes_through_public_function_wrappers():
    def invalid_perturbation(state):
        raise InvalidProposalException("deliberately invalid")

    inversion = BaseBayesianInversion(
        walkers_starting_states=[0],
        perturbation_funcs=[invalid_perturbation],
        log_like_func=lambda state: 0.0,
        n_chains=1,
        save_dpred=False,
    )
    chain = inversion.chains[0]
    chain.save_current_iteration = False
    chain._next_iteration()
    assert chain.current_state == 0
    assert chain.statistics["n_proposed_models_total"] == 1
    assert chain.statistics["exceptions"]["InvalidProposalException"] == 1


def test_invalid_model_from_custom_log_likelihood_is_rejected():
    def log_likelihood(state):
        if state == 1:
            raise InvalidProposalException("deliberately invalid")
        return 0.0

    inversion = BaseBayesianInversion(
        walkers_starting_states=[0],
        perturbation_funcs=[lambda state: (state + 1, 0.0)],
        log_like_func=log_likelihood,
        n_chains=1,
        save_dpred=False,
    )
    chain = inversion.chains[0]
    chain.save_current_iteration = False
    chain._next_iteration()
    assert chain.current_state == 0
    assert chain.statistics["n_proposed_models_total"] == 1
    assert chain.statistics["exceptions"]["InvalidProposalException"] == 1


def test_invalid_proposal_passes_through_target_forward_wrapper():
    def forward(state):
        raise InvalidProposalException("deliberately invalid")

    log_likelihood = LogLikelihood([Target("data", np.array([0.0]), 1.0)], [forward])
    with pytest.raises(InvalidProposalException, match="deliberately invalid"):
        log_likelihood.log_likelihood_ratio(State({}), State({}), temperature=1)
