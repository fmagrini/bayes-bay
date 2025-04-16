# Change Log

<!--next-version-placeholder-->

## v0.3.2 (16/04/2025)
- API change: Enabled custom starting states in high-level API
	- BayesianInversion now takes optional argument `walkers_starting_states`
	- MarkovChain now takes optional argument `starting_state`

## v0.3.1 (20/03/2024)
- API change: removed `DimensionalityException`
    - DimensionalityException replaced with -math.inf log probability in birth and death perturbations
- New feature: `Voronoi1D`
    - `Voronoi1D.compute_interface_positions()` added
- Enhancement: `Voronoi1D.compute_cell_extents` now deals with negative lower boundaries

## v0.3.0 (15/03/2024)

- New feature: Nested parameter space (and discretization)

## v0.2.4 (07/03/2024)

- Enhancement: `__repr__` in some classes

## v0.2.3 (07/03/2024)
- New feature: `bayesbay.prior.LaplacePrior`
- New feature: `bayesbay.LogLikelihood.__repr__`
- New feature: `bayesbay.Target.__repr__`

## v0.2.2 (04/03/2024)
- New feature: `Voronoi2D`
    - `Voronoi2D.get_tessellation_statistics()` added
- API change: `Voronoi1D`
    - generalized all static methods:
        - interpolate_tessellation
        - _interpolate_tessellations
        - get_tessellation_statistics
        - plot_tessellation
        - plot_tessellations
        - plot_tessellation_statistics
        - plot_tessellation_density

## v0.2.1 (27/02/2024)
- New feature: `Voronoi2D`
    - `Voronoi2D.interpolate_tessellation()` added
    - `Voronoi2D.polygon` added, allowing for prior probabilities defined within polygons
- API change: `bayesbay.samplers._samplers`
    - multiprocessing now carried out with `joblib`

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
