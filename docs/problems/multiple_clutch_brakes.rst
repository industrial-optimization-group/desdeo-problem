Multiple clutch brake design problem
=====================================
The five objectives of the problem are: :math:`(f_1)` minimize mass of the brake, 
:math:`(f_2)` minimize stopping time, :math:`(f_3)` minimize number of friction surfaces,
:math:`(f_4)` minimize outer radius and :math:`(f_5)` minimize actuating force. 
More details about the test problem can be found in [Osyczka1992]_

**Definition**

.. math::

  \min \; f_1(x) = & \; \pi \; (x_2^2 - x_1^2) \; x_3 \; (x_5 + 1) \; \rho \\[2mm]
  \min \; f_2(x) = & \; t_h = \frac{J_z \; \omega}{M_h + M_f} \\[2mm]
  \text{Where braking torque } M_h \text{ is:}\\[2mm]
  M_h = & \frac{2}{3} \mu \; x_4 \; x_5 \frac{x_2^3 - x_1^3}{x_2^2 - x_1^2} \\[2mm]
  \text{and input angular velocity } \omega \text{ is:}\\[2mm]
  \omega = & \frac{\pi \; n}{30} \\[2mm]
  \min \; f_3(x) = & \; x_5 \\[2mm]
  \min \; f_4(x) = & \; x_2 \\[2mm]
  \min \; f_5(x) = & \; x_4

The five variables represent inner radius :math:`(x_1)`,
outer radius :math:`(x_2)`, thickness of the disc :math:`(x_3)`,
actuating force :math:`(x_4)` and number of friction surfaces :math:`(x_5)`.

Variable bounds are given as follows:

.. math::

  55 \leq x_1 \leq 80 \quad \quad 75 \leq x_2 \leq 110 \quad \quad 1.5 \leq x_3 \leq 3
  \quad \quad 300 \leq x_4 \leq 1000 \quad \quad 2 \leq x_5 \leq 10 

Constraints are given as follows:

.. math::
  g_1(x) = & \; x_1 - 55 \geq 0 \quad & \quad g_9(x) = & \; p_{max} - p_{rz} \geq 0 \\[2mm]
  g_2(x) = & \; 110 - x_2 \geq 0 \quad & \quad p_{rz} = & \; \frac{F}{S} \\[2mm]
  g_3(x) = & \; (x_2 - x_1) - 20 \geq 0 \quad & \quad S = & \; \pi \; x_2^2 - \pi \; x_1^2 \\[2mm]
  g_4(x) = & \; x_3 - 1.5 \geq 0 \quad & \quad g_{10}(x) = & \; p_{max} \; V_{srmax} - p_{rz} \; V_{sr} \geq 0 \\[2mm]
  g_5(x) = & \; 3 - x_3 \geq 0 \quad & \quad V_{sr} = & \; \frac{\pi \; R_{sr} \; n}{30} \\[2mm]
  g_6(x) = & \; 30 - (x_5 + 1) \; (x_3 + 0.5) \geq 0 \quad & \quad R_{sr} = & \; \frac{2}{3} \frac{x_2^3-x_1^3}{x_2^2-x_1^2} \\[2mm]
  g_7(x) = & \; 10 - (x_5 + 1) \geq 0 \quad & \quad g_{11}(x) = & \; V_{srmax} - \; V_{sr} \geq 0 \\[2mm]
  g_8(x) = & \; x_5 - 1 \geq 0 \quad & \quad g_{12}(x) = & \; t_{max} - t_h \geq 0 \\[2mm]
  \quad & \quad \quad & \quad
  g_{13}(x) = & \; M_h \; s \; M_s \geq 0 \\[2mm]
  \quad & \quad \quad & \quad
  g_{14}(x) = & \; t_h \geq 0 \\[2mm]
  \quad & \quad \quad & \quad
  g_{15}(x) = & \; x_4 \geq 0 \\[2mm]
  \quad & \quad \quad & \quad
  g_{16}(x) = & \; F_{max} - x_4 \geq 0 \\[2mm]

Parameters are given as follows:

.. math::

  J_z &= 55 \text{ kg/mm}^2 \quad & \quad
  \rho &= 0.0000078 \text{ kg/mm}^3 \\[2mm]
  M_f &= 3 \text{ Nm} \quad & \quad
  n &= 250 \text{ rev/min} \\[2mm]
  \mu &= 0.5 \quad & \quad
  p_{max} &= 1 \text{ MPa} \\[2mm]
  V_{srmax} &= 10 \text{ m/s} \quad & \quad
  t_{max} &= 15 \text{ s} \\[2mm]
  s &= 1.5 \quad & \quad
  F_{max} &= 1000 \text{ N} \\

.. [Osyczka1992] Osyczka, A. (1992). Computer Aided Multicriterion Optimization System (CAMOS): 
  Software Package in Fortran: with 32 Figures and a Diskette Containing 5526 Lines 
  of Source Version of Programs. International Software Publishers.