Engineering real-world
===========================

Collection of real-world multi-objective optimization test problems. More details about 
the test problems can be found in [1]_

RE-21: Four bar truss design problem
----------------------------------------
The two objectives of the problem are: :math:`(f_1)` minimize structural volume 
and :math:`(f_2)` minimize the joint displacement.

**Definition**

.. math::

  \min \; f_1(x) = & \; L \big( 2x_1 + \sqrt{2} x_2 + \sqrt{x_3} + x_4 \big) \\[2mm]
  \min \; f_2(x) = & \; \frac{FL}{E} \Bigg( \frac{2}{x_1} + \frac{2\sqrt{2}}{x_2} - \frac{2\sqrt{2}}{x_3} + \frac{2}{x_4} \Bigg)

The four variables determine the length of four bars, respectively.

Variable bounds are given as follows:

.. math::

  &a \leq x_1 \leq 3a \quad \quad \sqrt{2}a &\leq x_2 \leq 3a \quad \quad
  \sqrt{2}a \leq x_3 \leq 3a \quad  \quad a \leq x_4 \leq 3a \\
  \\
  &\text{where: }a = F/\sigma

Parameters are given as follows:

.. math::

  F &= 10 \text{ kN} \quad & \quad
  E &= 2 \times 10^5 \text{ kN/cm}^2 \\
  L &= 200 \text{ cm} \quad & \quad
  \sigma &= 10 \text{ kN/cm}^2

RE-22 Reinforced concrete beam design problem
-------------------------------------------------
The two objectives of the problem are: :math:`(f_1)` minimize total cost of
concrete and reinforcing steel of the beam and :math:`(f_2)` minimize the sum of constraint violations.

**Definition**

.. math::

  \min \; f_1(x) = & \; 29.4x_1 + 0.6x_2x_3 \quad & \quad
  g_1(x) = & \; x_1x_3 - 7.735 \frac{x_1^2}{x_2} - 180 \geq 0 \\
  \min \; f_2(x) = & \; \displaystyle\sum_{i=1}^{2} \max \{ g_i(x), 0 \} \quad & \quad
  g_2(x) = & \; 4 - \frac{x_3}{x_2} \geq 0

The three variables represent the area of reinforcement :math:`(x_1)`, the width of the beam
:math:`(x_2)` and the depth of the beam :math:`(x_3)`. :math:`x_1` has a pre-defined discrete 
value from 0.2 to 15.

Variable bounds are given as follows:

.. math::

  0.2 \leq x_1 \leq 15 \quad \quad 0 \leq x_2 \leq 20 \quad \quad 
  0 \leq x_3 \leq 40

RE-23 Pressure vessel design problem
-----------------------------------------
The two objectives of the problem are: :math:`(f_1)` minimize total cost of a
cylindrical pressure vessel and :math:`(f_2)` minimize the sum of constraint violations.

**Definition**

.. math::

  \min \; f_1(x) = & \; 0.6224x_1x_3x_4 + 1.7781x_2x_3^2 \quad & \quad
  g_1(x) = & \; x_1 - 0.0193x_3 \geq 0 \\[2mm]
  & + 3.1661x_1^2x_4 + 19.84x_1^2x_3 \\[2mm]
  \min \; f_2(x) = & \; \displaystyle\sum_{i=1}^{3} \max \{ g_i(x), 0 \} \quad & \quad
  g_2(x) = & \; x_2 - 0.00954x_3 \geq 0 \\[2mm]
  \quad & \quad \quad & \quad
  g_3(x) = & \; \pi x_3^2 x_4 + \frac{4}{3} \pi x_3^3 - 1 \; 296 \; 000 \geq 0

The four variables represent the thicknesses of the shell :math:`(x_1)`, 
the head of a pressure vessel :math:`(x_2)`, the inner radius :math:`(x_3)` and 
the length of the cylindrical section :math:`(x_4)`.

Variable bounds are given as follows:

.. math::

  1 \leq x_1 \leq 100 \quad \quad 1 \leq x_2 \leq 100 \quad \quad 
  10 \leq x_3 \leq 200 \quad \quad 10 \leq x_4 \leq 240
    
RE-24 Hatch cover design problem
------------------------------------
The two objectives of the problem are: :math:`(f_1)` minimize the weight of the hatch cover
and :math:`(f_2)` minimize the sum of constraint violations.

**Definition**

.. math::

  \min \; f_1(x) = & \; x_1 + 120 x_2 \quad & \quad
  g_1(x) = & \; 1.0 - \frac{\sigma_b}{\sigma_{b,max}} \geq 0 \\[2mm]
  \min \; f_2(x) = & \; \displaystyle\sum_{i=1}^{4} \max \{ g_i(x), 0 \} \quad & \quad
  g_2(x) = & \; 1.0 - \frac{\tau}{\tau_{max}} \geq 0 \\[2mm]
  \quad & \quad \quad & \quad
  g_3(x) = & \; 1.0 - \frac{\delta}{\delta_{max}} \geq 0 \\[2mm]
  \quad & \quad \quad & \quad
  g_4(x) = & \; 1.0 - \frac{\sigma_b}{\sigma_k} \geq 0

The two variables represent the flange thickness :math:`(x_1)` and 
the beam height of the hatch cover :math:`(x_2)`.

Variable bounds are given as follows:

.. math::

  0.5 \leq x_1 \leq 4 \quad \quad 4\leq x_2 \leq 40 

Parameters are given as follows:

.. math::

  \sigma_{b,max} &= 700 \text{ kg/cm}^2 \quad & \quad
  \tau_{max} &= 450 \text{ kg/cm}^2 \\
  \delta_{}max &= 1.5 \text{ cm} \quad & \quad
  \sigma_k &= Ex_1^2/100 \text{ kg/cm}^2 \\
  \sigma_b &= 4500/(x_1x_2)\text{ kg/cm}^2 \quad & \quad
  \tau &= 1800/x_2\text{ kg/cm}^2 \\
  \delta &= 56.2 \times 10^4 /(Ex_1x_2^2) \quad & \quad
  E &= 700 \; 000 \text{ kg/cm}^2  

RE-25 Coil compression spring design problem
------------------------------------------------
The two objectives of the problem are: :math:`(f_1)` minimize the volume of spring steel
wire which is used to manufacture the spring and :math:`(f_2)` minimize the sum of constraint violations.

**Definition**

.. math::

  \min \; f_1(x) = & \; \frac{\pi^2 x_2 x_3^2 (x_1 + 2)}{4} \quad & \quad
  C_f = & \; \frac{4(x_2/x_3) - 1}{4(x_2/x_3) - 4} + \frac{0.615x_3}{x_2} \\[2mm]
  \min \; f_2(x) = & \; \displaystyle\sum_{i=1}^{6} \max \{ g_i(x), 0 \} \quad & \quad
  K = & \; \frac{Gx_3^4}{8x_1x_2^3} \\[2mm]
  g_1(x) = & \; - \frac{8C_f F_{max} x_2}{\pi x_3^3} + S \geq 0 \quad & \quad
  \sigma_p = & \; \frac{F_p}{K} \\[2mm]
  g_2(x) = & \; -l_f + l_{max} \geq 0 \quad & \quad
  l_f = & \; \frac{F_{max}}{K} + 1.05(x_1 + 2) x_3 \\[2mm]
  g_3(x) = & \; -3 + \frac{x_2}{x_3} \geq 0 \\[2mm]
  g_4(x) = & \; - \sigma_p + \sigma_{pm} \geq 0 \\[2mm]
  g_5(x) = & \; - \sigma_p - \frac{F_{max} - F_p}{K} \\ & - 1.05 (x_1 + 2) x_3 + l_f \geq 0 \\[2mm]
  g_6(x) = & \; - \sigma_w + \frac{F_{max} - F_p}{K} \geq 0 \\[2mm]

The three variables represent the number of of spring coils :math:`(x_1)`, 
the outside diameter of the spring :math:`(x_2)` and the spring wire diameter :math:`(x_3)`.
:math:`x_3` has a pre-defined discrete value from 0.009 to 0.5.

Variable bounds are given as follows:

.. math::

  1 \leq x_1 \leq 70 \quad \quad 0.6 \leq x_2 \leq 30 \quad \quad 0.009 \leq x_3 \leq 0.5 

Parameters are given as follows:

.. math::

  F_{max} &= 1000 \text{ lb} \quad & \quad
  S &= 189 \; 000 \text{ psi} \\
  l_{max} &= 14 \text{ inch} \quad & \quad
  d_{min} &= 0.2 \text{ inch} \\
  D_{max} &= 3 \text{ inch} \quad & \quad
  F_p &= 300 \text{ lb} \\
  \sigma_{pm} &= 6 \text{ inch} \quad & \quad
  \sigma_w &= 1.25 \text{ inch} \\
  G &= 11.5 \times 10^6

RE-31 Two bar truss design problem
--------------------------------------
The three objectives of the problem are: :math:`(f_1)` minimize the structural weight, 
:math:`(f_2)` minimize the resultant displacement of joint and :math:`(f_3)` minimize 
the sum of constraint violations.

**Definition**

.. math::

  \min \; f_1(x) = & \; x_1 \sqrt{16 + x_3^2} + x_2 \sqrt{1 + x_3^2} \quad & \quad
  g_1(x) = & \; 0.1 - f_1(x) \geq 0 \\[2mm]
  \min \; f_2(x) = & \; \frac{20 \sqrt{16 + x_3^2}}{x_3x_1} \quad & \quad
  g_2(x) = & \; 10^5 - f_2(x) \geq 0 \\[2mm]
  \min \; f_3(x) = & \; \displaystyle\sum_{i=1}^{3} \max \{ g_i(x), 0 \} \quad & \quad
  g_3(x) = & \; 10^5 - \frac{80 \sqrt{1 + x_3^2}}{x_3x_2} \geq 0

:math:`x_1` and :math:`x_2` represent the length of the two bars and :math:`x_3` 
represents the vertical distance from the second bar.

Variable bounds are given as follows:

.. math::

  10^{-5} \leq x_1 \leq 100 \quad \quad 10^{-5} \leq x_2 \leq 100 
  \quad \quad 1 \leq x_3 \leq 3

RE-32 Welded beam design problem
------------------------------------
The three objectives of the problem are: minimize the cost :math:`(f_1)` and end deflection 
:math:`(f_2)` of a welded beam and :math:`(f_3)` minimize the sum of constraint violations.

**Definition**

.. math::

  \min \; f_1(x) = & \; 1.10471x_1^2x_2 \quad & \quad
  \tau(x) = & \; \sqrt{(\tau')^2 + \frac{2\tau'\tau''x_2}{2R} + (\tau'')^2} \\
  & + 0.04811x_3x_4 (14 + x_2) \\[2mm]
  \min \; f_2(x) = & \; \frac{4PL^3}{Ex_4x_3^3} \quad & \quad
  \tau' = & \; \frac{P}{\sqrt{2}x_1x_2} \\[2mm]
  \min \; f_3(x) = & \; \displaystyle\sum_{i=1}^{4} \max \{ g_i(x), 0 \} \quad & \quad
  \tau'' = & \; \frac{MR}{J} \\[2mm]
  g_1(x) = & \; \tau_{max} - \tau(x) \geq 0 \quad & \quad
  M = & \; P \Big( L + \frac{x_2}{2} \Big) \\[2mm]
  g_2(x) = & \; \sigma_{max} - \sigma(x) \geq 0 \quad & \quad
  R = & \; \sqrt{\frac{x_2^2}{4} + \bigg( \frac{x_1 + x_3}{2} \bigg)^2 } \\[2mm]
  g_3(x) = & \; x_4 - x_1 \geq 0 \quad & \quad
  J = & \; 2 \Bigg( \sqrt{2} x_1x_2 \bigg( \frac{x_2^2}{12} + \Big( \frac{x_1 + x_3}{2} \Big)^2 \bigg) \Bigg) \\[2mm]
  g_4(x) = & \; P_C(x) - P \geq 0 \quad & \quad
  \sigma(x) = & \; \frac{6PL}{x_4x_3^2} \\[2mm]
  \quad & \quad \quad & \quad
  P_C(x) = & \; \frac{4.013E \sqrt{x_3^2x_4^6 / 36}}{L^2} \Bigg( 1 - \frac{x_3}{2L} \sqrt{\frac{E}{4G}} \Bigg)
  
The four variables adjust the size of the beam.

Variable bounds are given as follows:

.. math::

  0.125 \leq x_1 \leq 5 \quad \quad 0.1 \leq x_2 \leq 10 \quad \quad 0.1 \leq x_3 \leq 10
  \quad \quad 0.125 \leq x_4 \leq 5 

Parameters are given as follows:

.. math::

  P &= 6000 \text{ lb} \quad & \quad
  L &= 14 \text{ in} \\
  E &= 30 \times 10^6 \text{ psi} \quad & \quad
  G &= 12 \times 10^6 \text{ psi} \\
  \tau_{max} &= 13 \; 600 \text{ psi} \quad & \quad
  \sigma_{max} &= 30 \; 000 \text{ psi}

RE-33 Disc brake design problem
------------------------------------
The three objectives of the problem are: minimize the the mass of the brake :math:`(f_1)` 
and the minimum stopping time :math:`(f_2)` of a disc brake and :math:`(f_3)` 
minimize the sum of constraint violations.

**Definition**

.. math::

  \min \; f_1(x) = & \; 4.9 \times 10^{-5}(x_2^2 - x_1^2)(x_4 - 1) \quad & \quad
  g_1(x) = & \; (x_2 - x_1) - 20 \geq 0 \\[2mm]
  \min \; f_2(x) = & \; 9.82 \times 10^6 \bigg(\frac{x_2^2 - x_1^2}{x_3x_4(x_2^3 - x_1^3)} \bigg) \quad & \quad
  g_2(x) = & \; 0.4 - \frac{x_3}{3.14(x_2^2 - x_1^2)} \geq 0 \\[2mm]
  \min \; f_3(x) = & \; \displaystyle\sum_{i=1}^{4} \max \{ g_i(x), 0 \} \quad & \quad
  g_3(x) = & \; 1 - \frac{2.22 \times 10^{-3}x_3 (x_2^3 - x_1^3)}{(x_2^2 - x_1^2)^2} \geq 0 \\[2mm]
  \quad & \quad \quad & \quad
  g_4(x) = & \; \frac{2.66 \times 10^{-2}x_3 x_4 (x_2^3-x_1^3)}{(x_2^2-x_1^2)} - 900 \geq 0

The four variables represent the inner radius of the discs :math:`(x_1)`, 
the outer radius of the discs :math:`(x_2)`, the engaging force :math:`(x_3)`
and the number of friction surfaces :math:`(x_4)`

Variable bounds are given as follows:

.. math::

  55 \leq x_1 \leq 80 \quad \quad 75 \leq x_2 \leq 110 \quad \quad 1000 \leq x_3 \leq 3000
  \quad \quad 11 \leq x_4 \leq 20

.. [1] Tanabe, R. & Ishibuchi, H. (2020). An easy-to-use real-world 
  multi-objective optimization problem suite. 
  Applied soft computing, 89, 106078. 
  https://doi.org/10.1016/j.asoc.2020.106078