Partition Modeling
------------------

In this tutorial, we apply BayesBay to a regression task involving observed data generated from a discontinuous piecewise function. Throughout, we consider the function 

.. math::

    f(x) = \left\{
        \begin{array}{ll}
        1 & \quad x \leq 1 \\
        20 & \quad 1 < x \leq 2.5 \\
        0 & \quad 2.5 < x \leq 3 \\
        -3 & \quad 3 < x \leq 4 \\
        -10 & \quad 4 < x \leq 6 \\
        -20 & \quad 6 < x \leq 6.5 \\
        25 & \quad 6.5 < x \leq 8 \\
        0 & \quad 8 < x \leq 9 \\
        10 & \quad 9 < x \leq 10, \\
        \end{array}
    \right.
    
Our goal is to infer :math:`f(x)` from noisy observations :math:`\mathbf{d}_{obs} = f(x_i) + \mathcal{N}(0, \sigma)` via Bayesian sampling.

This tutorial comprises:

.. toctree::
   :maxdepth: 1

   41_simple_partition_mod
   42_transd_partition_mod
