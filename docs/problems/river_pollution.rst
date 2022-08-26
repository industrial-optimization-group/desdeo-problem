River pollution problem
========================
The problem is to improve the dissolved oxygen concentration in the city :math:`(f_1)`
and at the municipality border :math:`(f_2)`, maximize the percent return on investment
at the fishery :math:`(f_3)` and minimize addition to the city tax :math:`(f_4)`.
Here is optional fifth objective [1]_ to keeping to propotional amount of biomechanical
oxygen demanding material (BOD) removed from the water close to the idal value :math:`(f_5)`.
More details about the test problem can be found in [2]_.

**Definition**

.. math::

  \min \; f_1(x) = & -4.07 - 2.27x_1 \\[2mm]
  \min \; f_2(x) = & -2.60 - 0.03x_1 - 0.02x_2 - \frac{0.01}{1.39 - x_1^2} - \frac{0.30}{1.39-x_2^2} \\[2mm]
  \min \; f_3(x) = & -8.21 + \frac{0.71}{1.09 - x_1^2} \\[2mm]
  \min \; f_4(x) = & -0.96 + \frac{0.96}{1.09 - x_2^2} \\
  \\
  \text{Optional fifth objective:}\\[2mm]
  \min \; f_5(x) = & \max \{ |x_1 - 0.65|, |x_2 - 0.65| \} \\

The two variables represent the proportionate amount of BOD removed from water discharge
at the fishery :math:`(x_1)` and at the city :math:`(x_2)`.

Variable bounds are given as follows:

.. math::

  0.3 \leq x_1 \leq 1.0 \quad \quad 0.3 \leq x_2 \leq 1.0


.. [1] Miettinen, K., Mäkelä, M.M. (1997). Interactive Method NIMBUS for Nondifferentiable 
  Multiobjective Optimization Problems. In: Clímaco, J. (eds) Multicriteria Analysis. 
  Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-60667-0_30

.. [2] NARULA, S. C. & WEISTROFFER, H. R. (1989). A flexible method for 
  nonlinear multicriteria decisionmaking problems. IEEE transactions on systems, 
  man, and cybernetics, 19(4), 883-887.