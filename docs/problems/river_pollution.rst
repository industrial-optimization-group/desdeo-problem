River pollution
====================

The River pollution problem

**Definition**

.. math::

  \min \; f_1(x) = & -4.07 - 2.27x_1 \\[2mm]
  \min \; f_2(x) = & -2.60 - 0.03x_1 - 0.02x_2 - \frac{0.01}{1.39 - x_1^2} - \frac{0.30}{1.39-x_2^2} \\[2mm]
  \min \; f_3(x) = & -8.21 + \frac{0.71}{1.09 - x_1^2} \\[2mm]
  \min \; f_4(x) = & -0.96 + \frac{0.96}{1.09 - x_2^2} \\
  \\
  \text{Optional fifth objective:}\\[2mm]
  \min \; f_5(x) = & \max \{ |x_1 - 0.65|, |x_2 - 0.65| \} \\

**Variable bounds are given as follows:**

.. math::

  0.3 \leq x_1 \leq 1.0 \quad \quad 0.3 \leq x_2 \leq 1.0

.. [Narula1989] NARULA, S. C. & WEISTROFFER, H. R. (1989). A flexible method for 
  nonlinear multicriteria decisionmaking problems. IEEE transactions on systems, 
  man, and cybernetics, 19(4), 883-887.