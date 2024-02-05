States
======

A ``State`` in BayesBay is a data structure containing all numerical values associated with a certain Markov chain step. Depending on the inference problem at hand, it could contain information about:

- Discretization. For example, the nuclei positions in a Voronoi tessellation.
- Free parameters. For example, the value of the physical property to be inferred, associated with each Voronoi cell.
- Data noise. This includes standard deviation and noise correlation.
- Cached information. This consists of a Python dictionary containing arbitrary objects that the user might want to store and reuse at the next Markov chain step (see :meth:`bayesbay.State.save_to_cache`).

.. autosummary::
    :toctree: generated/
    :nosignatures:

    bayesbay.State
    bayesbay.ParameterSpaceState
    bayesbay.DataNoiseState
