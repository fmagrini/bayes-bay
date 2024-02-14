# Change Log

<!--next-version-placeholder-->

## v0.2.0 (14/02/2024)

- New feature:
    - `Voronoi2D` added
- New feature: `pertubation_weights`
    - weights can be updated
        - `BaseBayesianInversion.set_perturbation_funcs(funcs, weights)`
        - `BaseMarkovChain.set_perturbation_funcs(funcs, weights)`
    - weights can be assigned during initialization
        - `BaseBayesianInversion(perturbation_weights, ...)`
        - `BaseMarkovChain(perturbation_weights, ...)`
        - `MarkovChain(perturbation_weights, ...)`
- API change: `bayesbay.parameters` -> `bayesbay.prior`
    - `UniformParameter` -> `UniformPrior`
    - `GaussianParameter` -> `GaussianPrior`
    - `CustomParameter` -> `CustomPrior`
- API change: `ParameterSpaceState` and `State`
    - `cache` added to `ParameterSpaceState`
    - `save_to_cache`, `load_from_cache`, `saved_in_cache` added to `ParameterSpaceState`
    - By default, the `ParameterSpaceState.cache` is carried over when calling `State.copy()`, i.e., it is copied to the new ParameterSpaceState
- API change: `bayesbay.discretization._voronoi`
    - All methods in `Voronoi1D`, except for static ones, have been moved to `Voronoi`, which is now more general

    

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
