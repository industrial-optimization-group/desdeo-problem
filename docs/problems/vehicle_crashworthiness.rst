Vehicle crashworthiness
=================

The crash safety design problem with three objectives, which have to be minimized.

f1: the mass of the vehicle 
f2: accelration-induced biomechanical damage of occupants
f3: the toe board intrusion in the 'offset-frontal crash'

More details about the problem can be found in [Liao2007]_

**Definition**

.. math::

  \min f_1(x) = & \; 1640.2823 + 2.3573285x_1 + 2.3220035x_2 \\
  & + 4.5688768x_3 + 7.7213633x_4 + 4.4559504x_5 \\
  \min f_2(x) = & \; 6.5856 + 1.15x_1 - 1.0427x_2 + 0.9738x_3 \\
  & + 0.8364x_4 - 0.3695x_1x_4 + 0.0861x_1x_5 + 0.3628x_2x_4 \\
  & - 0.1106x_1^2 - 0.3437x_3^2 + 0.1764x_4^2 \\
  \min f_3(x) = & -0.0551 + 0.0181x_1 + 0.1024x_2 \\
  & + 0.0421x_3 - 0.0073x_1x_2 + 0.024x_2x_3 - 0.0118x_2x_4 \\
  & - 0.0204x_3x_4 - 0.008x_3x_5 - 0.0241x_2^2 + 0.0109x_4^2 \\
  & s.t. \; 1mm \leq x \leq 3mm

"The thickness of the five reinforced members around the frontal structure is chosen as the design variables 
which could significantly affect the crach safety."

.. [Liao2007] Liao, X., Li, Q., Yang, X., Zhang, W. & Li, W. (2007).
Multiobjective optimization for crash safety design of vehicles
using stepwise regression model. Structural and multidisciplinary
optimization, 35(6), 561-569. https://doi.org/10.1007/s00158-007-0163-x
