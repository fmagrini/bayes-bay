Polynomial Fitting
------------------

We demonstrate in this tutorial how to use BayesBay to tackle a polynomial fitting problem. In the following, we utilise a third-degree polynomial, :math:`y(x) = m_0 + m_1 x + m_2 x^2 + m_3 x^3`, to generate data points :math:`y_1(x_1), y_2(x_2), \ldots, y_n(x_n)`. In matrix notation, these can be expressed as

.. math::
   \underbrace{\begin{pmatrix}y_0\\y_1\\\vdots\\y_N\end{pmatrix}}_{\text{Data}} = \underbrace{\begin{pmatrix}1&x_0&x_0^2&x_0^3\\1&x_1&x_1^2&x_1^3\\\vdots&\vdots&\vdots\\1&x_N&x_N^2&x_N^3\end{pmatrix}}_{\text{Forward operator or data kernel}} \underbrace{\begin{pmatrix}m_0\\m_1\\m_2\\m_3\end{pmatrix}}_{\text{Model}},

or :math:`\mathbf{d}_{pred} = \mathbf{G m}`.

Having generated a synthetic data set using the model coefficients :math:`m_0`, :math:`m_1`, :math:`m_2`, and :math:`m_3`, we add noise to define the observed data :math:`\mathbf{d}_{obs} = \mathbf{d}_{pred} + \mathbf{e}`, where the :math:`i`\ th entry of the observational error :math:`\mathbf{e}` is randomly sampled from the normal distribution :math:`\mathcal{N}(0, \sigma)`. Finally, we use BayesBay to retrieve the true coefficients from the observations.

This tutorial comprises two parts:

.. toctree::
   :maxdepth: 1

   01_polyfit
   02_hierarchical_polyfit
