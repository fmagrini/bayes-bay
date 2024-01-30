Gaussian Mixture Model
----------------------

In this tutorial, we demonstrate how to use BayesBay to retrieve a Gaussian Mixture Model (GMM). GMMs rest on the assumption that a complex probability distribution can be decomposed into simpler Gaussian distributions. By the weighted sum of these individual Gaussians, GMMs enable the calculation of the probability density function

.. math::
   f(x) = \sum_{i=1}^N \omega_i \cdot \mathcal{N}(x; \mu_i, \sigma_i),

where :math:`\mu_i` and :math:`\sigma_i` denote the mean and standard deviation of the :math:`i`\ th Gaussian and the weights are chosen such that :math:`\sum_{i=1}^N \omega_i = 1`.

GMMs are particularly useful to analyze datasets whose underlying distribution is a combination of different Gaussians. This is a common scenario in real-world data, where the considered statistical population often comprises several distinct subpopulations, each characterized by its own distribution. For example, imagine you have the following height measurements from a mixed group of adults and children:

1. **Children (Aged 8 to 12 years)**
   
   - Measurements: 4000
   - Mean height: 140 cm
   - Standard deviation: 12 cm

2. **Adult Women**

   - Measurements: 3000
   - Mean height: 162 cm
   - Standard deviation: 5 cm

3. **Adult Men**

   - Measurements: 3000
   - Mean height: 177 cm
   - Standard deviation: 6 cm

The height distribution for the above subgroups each follows a Gaussian distribution, but the combined dataset will have three peaks, which cannot be accurately modeled by a single Gaussian. A GMM can model this dataset as a combination of three Gaussians, each capturing the distribution of one subgroup --- children, adult women, and adult men.

This tutorial comprises two parts:

.. toctree::
   :maxdepth: 1

   11_gaussian_mixture
   12_transd_gaussian_mixture
