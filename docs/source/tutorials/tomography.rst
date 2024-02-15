Tomography
----------

We demonstrate in this tutorial BayesBay's capabilities in tackling tomographic imaging tasks. While we focus in particular on seismic tomography problems, the concepts presented are also relevant to other areas, such as X-ray medical imaging.

Wave propagation in elastic media, such as the Earth's subsurface, results in a continuous flow of energy. At high frequencies, where wavelengths are much shorter than medium's heterogeneities, the wavefront of a propagating seismic wave can be approximated using ray theory [1]_. Ray theory models waves as rays traveling along paths determined by the medium's velocity structure (or its reciprocal, the slowness :math:`s`).

Given a seismic source and a receiver, and assuming the propagation path is known, the total time required for the wave to travel between these two points reads

.. math::
   t = \int_{path} s(\phi(r), \theta(r)) dr,
   
where :math:`\phi` and :math:`\theta` denote longitude and latitude. The seismic tomographic inverse problem consists in inferring the Earth's slowness structure, :math:`s(\phi, \theta)`, from a set of wave arrival-time measurements.

This tutorial comprises:

.. toctree::
   :maxdepth: 1

   31_sw_tomography
   
   

.. rubric:: References
.. [1] Cervený 2001, Seismic Ray Theory, `Cambridge University Press`
