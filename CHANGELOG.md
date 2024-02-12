# Change Log

<!--next-version-placeholder-->

## v0.1.10 (xx/02/2024)

- API change: `bayesbay.parameters` -> `bayesbay.prior`
    - `UniformParameter` -> `UniformPrior`
    - `GaussianParameter` -> `GaussianPrior`
    - `CustomParameter` -> `CustomPrior`
- API change: `pertubation_weights`
    - weights can be updated
        - `BaseBayesianInversion.set_perturbation_funcs(funcs, weights)`
        - `BaseMarkovChain.set_perturbation_funcs(funcs, weights)`
    - weights can be assigned during initialization
        - `BaseBayesianInversion(perturbation_weights, ...)`
        - `BaseMarkovChain(perturbation_weights, ...)`
        - `MarkovChain(perturbation_weights, ...)`

## v0.1.9 (05/02/2024)

- API change: `LogLikelihood` can accept log likelihood function or log likelihood ratio function
- API change: `BayesianInversion` now takes in instance of `LogLikelihood`, instead of a list of targets and forward functions
- New API: `LogLikelihood.add_targets()` method that allows more data to join the inversion
- New API: `BayesianInversion.get_results_from_chains`
- New API: `Parameter.sample` method
- Enhancement: `__repr__` and `__str__` for all classes
- Fix: bug in nearest index when birth from neighbour

## v0.1.1 (15/01/2024)

- Add pyx file to Manifest.in

## v0.1.0 (15/01/2024)

- First release
