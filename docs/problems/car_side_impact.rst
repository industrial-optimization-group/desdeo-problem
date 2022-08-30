Car side impact design problem
=================================
The three objectives of the problem are: :math:`(f_1)` minimize the weight of the car, 
:math:`(f_2)` minimize the pubic force experienced by a passenger, and :math:`(f_3)` 
minimize the average velocity of the V-pillar responsible for withstanding the impact load.
Here is optional fourth objective [2]_ :math:`(f_4)` the sum of constraint violations.
More details about the test problem can be found in [1]_.

**Definition**

.. math::

  \min \; f_1(x) = & \; 1.98 + 4.9x_1 + 6.67x_2 + 6.98x_3 \\[2mm]
  & + 4.01x_4 + 1.78x_5 + 10^{-5} x_6 + 2.73x_7 \\[2mm]
  \min  \; f_2(x) = & \; 4.72 - 0.5x_4 - 0.19x_2x_3 \\[2mm]
  \min  \; f_3(x) = & \; 0.5(V_{MBP}(x) V_{FD}(x)) \\[2mm]  
  \text{Optional fourth objective:}\\
  \min \; f_4(x) = & \; \displaystyle\sum_{i=1}^{10} \max \{ g_i(x), 0 \}\\[2mm]
  V_{MBP}(x) = & \; 10.58 - 0.674x_1x_2 - 0.67275x_2 \\[2mm]
  V_{FD}(x) = & \; 16.45 - 0.489x_3x_7 - 0.843x_5x_6 \\[2mm]
  

Constraints:

.. math::
  
  g_1(x) = & \; 1 - 1.16 + 0.3717x_2x_4 + 0.0092928x_3 \geq 0 \\[2mm]
  g_2(x) = & \; 0.32 - 0.261 + 0.0159x_1x_2 + 0.06486x_1 \\[2mm]
  & + 0.019x_2x_7 - 0.0144x_3x_5 - 0.0154464x_6 \geq 0 \\[2mm]
  g_3(x) = & \; 0.32 - 0.214 - 0.00817x_5 + 0.045195x_1 \\[2mm]
  & + 0.0135168x_1 - 0.03099x_2x_6 + 0.018x_2x_7 - 0.007176x_3 \\[2mm]
  & - 0.023232x_3 + 0.00364x_5x_6 + 0.018x_2^2 \geq 0 \\[2mm]
  g_4(x) = & \; 0.32 - 0.74 + 0.61x_2 + 0.031296x_3 \\[2mm]
  & + 0.031872x_7 - 0.227x_2^2 \geq 0 \\[2mm]
  g_5(x) = & \; 32 - 28.98 - 3.818x_3 + 4.2x_1x_2 \\[2mm]
  & - 1.27296x_6 + 2.68065x_7 \geq 0 \\[2mm]
  g_6(x) = & \; 32 - 33.86 - 2.95x_3 + 5.057x_1x_2 \\[2mm]
  & + 3.795x_2 + 3.4431x_7 - 1.45728 \geq 0 \\[2mm]
  g_7(x) = & \; 32 - 46.36 + 9.9x_2 + 4.4505x_1 \geq 0 \\[2mm]
  g_8(x) = & \; 4 - f_2(x) \geq 0  \\[2mm]
  g_9(x) = & \; 9.9 - V_{MBP}(x) \geq 0\\[2mm]
  g_{10}(x) = & \; 15.7 - V_{FD}(x) \geq 0

The seven variables represent the thickness of B-Pillar inner :math:`(x_1)`, B-Pillar reinforcement
:math:`(x_2)`, floor side inner :math:`(x_3)`, cross members :math:`(x_4)`, door beam :math:`(x_5)`,
door beltline reinforcement :math:`(x_6)`, and roof rail :math:`(x_7)`.  

Variable bounds are given as follows:

.. math::

  0.5 & \leq x_1 \leq 1.5 \quad & \quad 0.45 &\leq x_2 \leq 1.35 \quad & \quad 0.5 &\leq x_3 \leq 1.5 \\
  0.5 & \leq x_4 \leq 1.5 \quad & \quad 0.875 &\leq x_5 \leq 2.625 \quad & \quad 0.4 &\leq x_6 \leq 1.2 \\
  0.4 & \leq x_7 \leq 1.2

.. [1] Jain, H. & Deb, K. (2014). An Evolutionary Many-Objective Optimization Algorithm 
  Using Reference-Point Based Nondominated Sorting Approach, Part II: Handling Constraints 
  and Extending to an Adaptive Approach. IEEE transactions on evolutionary computation, 
  18(4), 602-622. https://doi.org/10.1109/TEVC.2013.2281534 

.. [2] Tanabe, R. & Ishibuchi, H. (2020). An easy-to-use real-world 
  multi-objective optimization problem suite. 
  Applied soft computing, 89, 106078. 
  https://doi.org/10.1016/j.asoc.2020.106078
