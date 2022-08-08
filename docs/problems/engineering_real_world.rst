Engineering real-world
=============

Real-world optimization problem suite

RE-21
-----------
Four bar truss design problem

**Definition**

.. math::

  \begin{equation}
    \begin{split}
        \min \; f_1(x) = & \; L \big( 2x_1 + \sqrt{2} x_2 + \sqrt{x_3} + x_4 \big) \\
        \min \; f_2(x) = & \; \frac{FL}{E} \Bigg( \frac{2}{x_1} + \frac{2\sqrt{2}}{x_2} - \frac{2\sqrt{2}}{x_3} + \frac{2}{x_4} \Bigg) \\
    \end{split}
  \end{equation}

where x1, x4 ∈ [a, 3a], x2, x3 ∈ [√2a, 3a], and a = F/σ. The four variables
determine the length of four bars, respectively. The parameters are given as
follows: F = 10 kN, E = 2 × 105kN/cm2, L = 200 cm, σ = 10 kN/cm2.

RE-22
-----------
Reinforced concrete beam design problem

**Definition**

.. math::

  \begin{equation}
    \begin{split}
        \min \; f_1(x) = & \; 29.4x_1 + 0.6x_2x_3 \\
        \min \; f_2(x) = & \; \displaystyle\sum_{i=1}^{2} \max \{ g_i(x), 0 \} \\
        g_1(x) = & \; x_1x_3 - 7.735 \frac{x_1^2}{x_2} - 180 \geq 0 \\
        g_2(x) = & \; 4 - \frac{x_3}{x_2} \geq 0\\
    \end{split}
  \end{equation}

where x2 ∈ [0, 20] and x3 ∈ [0, 40]. We determined the ranges of x2 and x3
according to initial solutions in [S.2]. x1 has a pre-defined discrete value from
0.2 to 15. The three variables (x1, x2, and x3) represent the area of the rein-
forcement, the width of the beam, and the depth of the beam, respectively.

For details, x1 ∈ {0.2, 0.31, 0.4, 0.44, 0.6, 0.62, 0.79, 0.8, 0.88, 0.93, 1, 1.2, 1.24, 1.32, 1.4, 1.55,
1.58, 1.6, 1.76, 1.8, 1.86, 2, 2.17, 2.2, 2.37, 2.4, 2.48, 2.6, 2.64, 2.79, 2.8, 3, 3.08, 3, 1, 3.16, 3.41, 3.52,
3.6, 3.72, 3.95, 3.96, 4, 4.03, 4.2, 4.34, 4.4, 4.65, 4.74, 4.8, 4.84, 5, 5.28, 5.4, 5.53, 5.72, 6, 6.16, 6.32,
6.6, 7.11, 7.2, 7.8, 7.9, 8, 8.4, 8.69, 9, 9.48, 10.27, 11, 11.06, 11.85, 12, 13, 14, 15}.

RE-23
-----------
Pressure vessel design problem

**Definition**

.. math::

  \begin{equation}
    \begin{split}
        \min \; f_1(x) = & \; 0.6224x_1x_3x_4 + 1.7781x_2x_3^2 \\
        & + 3.1661x_1^2x_4 + 19.84x_1^2x_3 \\
        \min \; f_2(x) = & \; \displaystyle\sum_{i=1}^{3} \max \{ g_i(x), 0 \} \\
        g_1(x) = & \; x_1 - 0.0193x_3 \geq 0 \\
        g_2(x) = & \; x_2 - 0.00954x_3 \geq 0 \\
        g_2(x) = & \; \pi x_3^2 x_4 + \frac{4}{3} \pi x_3^3 - 1 \; 296 \; 000 \geq 0 \\
    \end{split}
  \end{equation}

where x1, x2 ∈ {1, ..., 100}, x3 ∈ [10, 200], and x4 ∈ [10, 240]. x1 and x2 are
integer multiples of 0.0625. x1, x2, x3, and x4 represent the thicknesses of
the shell, the head of a pressure vessel, the inner radius, and the length of
the cylindrical section, respectively.
    

RE-24
-----------
Hatch cover design problem

**Definition**

.. math::

  \begin{equation}
    \begin{split}
        \min \; f_1(x) = & \; x_1 + 120 x_2 \\
        \min \; f_2(x) = & \; \displaystyle\sum_{i=1}^{4} \max \{ g_i(x), 0 \} \\
        g_1(x) = & \; 1.0 - \frac{\sigma_b}{\sigma_{b,max}} \geq 0 \\
        g_2(x) = & \; 1.0 - \frac{\tau}{\tau_{max}} \geq 0 \\
        g_3(x) = & \; 1.0 - \frac{\delta}{\delta_{max}} \geq 0 \\
        g_4(x) = & \; 1.0 - \frac{\sigma_b}{\sigma_k} \geq 0 \\
    \end{split}
  \end{equation}

where x1 ∈ [0.5, 4] and x2 ∈ [4, 50]. The ranges of the two variables were deter-
mined according to solutions reported in [S.2]. x1 and x2 represent the flange
thickness and the beam height of the hatch cover, respectively. The parameters
are given as follows: σb,max = 700kg/cm2, τmax = 450 kg/cm, δmax = 1.5cm,
σk = Ex2
1/100 kg/cm2, σb = 4 500/(x1x2)kg/cm2 τ = 1 800/x2kg/cm2, δ =
56.2 × 104/(Ex1x2
2), and E = 700 000 kg/cm2.

RE-25
-----------
Coil compression spring design problem

**Definition**

.. math::

  \begin{equation}
    \begin{split}
        \min \; f_1(x) = & \; \frac{\pi^2 x_2 x_3^2 (x_1 + 2)}{4} \\
        \min \; f_2(x) = & \; \displaystyle\sum_{i=1}^{6} \max \{ g_i(x), 0 \} \\
        g_1(x) = & \; \frac{8C_f F_{max} x_2}{\pi x_3^3} + S \geq 0 \\
        g_2(x) = & \; -l_f + l_{max} \geq 0 \\
        g_3(x) = & \; 3 + \frac{x_2}{x_3} \geq 0 \\
        g_4(x) = & \; - \sigma_p + \sigma_{pm} \geq 0 \\
        g_5(x) = & \; - \sigma_p - \frac{F_{max} - F_p}{K} - 1.05 (x_1 + 2) x_3 + l_f \geq 0 \\
        g_6(x) = & \; - \sigma_w + \frac{F_{max} - F_p}{K} \geq 0 \\
        C_f = & \; \frac{4(x_2/x_3) - 1}{4(x_2/x_3) - 4} + \frac{0.615x_3}{x_2} \\
        K = & \; \frac{Gx_3^4}{8x_1x_2^3} \\
        \sigma_p = & \; \frac{F_p}{K} \\
        l_f = & \; \frac{F_{max}}{K} + 1.05(x_1 + 2) x_3
    \end{split}
  \end{equation}  

where x1 ∈ {1, ..., 70}, x2 ∈ [0.6, 30], and x3 has a predefined discrete value
from 0.009 to 0.5. x1, x2, and x3 indicate the number of spring coils, the
outside diameter of the spring, and the spring wire diameter, respectively. The
parameters are given as follows: Fmax = 1 000lb, S = 189 000psi, lmax = 14inch,
dmin = 0.2inch, Dmax = 3inch, Fp = 300lb, σpm = 6 inch, σw = 1.25inch,
G = 11.5 × 106.

For details, x3 ∈ {0.009, 0.0095, 0.0104, 0.0118, 0.0128, 0.0132, 0.014, 0.015, 0.0162, 0.0173,
0.018, 0.02, 0.023, 0.025, 0.028, 0.032, 0.035, 0.041, 0.047, 0.054, 0.063, 0.072, 0.08, 0.092, 0.105,
0.12, 0.135, 0.148, 0.162, 0.177, 0.192, 0.207, 0.225, 0.244, 0.263, 0.283, 0.307, 0.331, 0.362, 0.394,
0.4375, 0.5}.


RE-31
-----------

**Definition**

.. math::

  \begin{equation}
    \begin{split}
        \min \; f_1(x) = & \; x_1 \sqrt{16 + x_3^2} + x_2 \sqrt{1 + x_3^2} \\
        \min \; f_2(x) = & \; \frac{20 \sqrt{16 + x_3^2}}{x_3x_1} \\ 
        \min \; f_3(x) = & \; \displaystyle\sum_{i=1}^{3} \max \{ g_i(x), 0 \} \\
        g_1(x) = & \; 0.1 - f_1(x) \geq 0 \\
        g_2(x) = & \; 10^5 - f_2(x) \geq 0 \\
        g_3(x) = & \; 10^5 - \frac{80 \sqrt{1 + x_3^2}}{x_3x_2} \geq 0 \\
    \end{split}
  \end{equation}

where $x_1, x_2 \in [10^{-5}, 100]$ and $x_3 \in [1, 3]$. $x_1$ and $x_2$ indicate the length of the
two bars. $x_3$ represents the vertical distance from the second bar. 

RE-32
-----------
Welded beam design problem

**Definition**

.. math::

  \begin{equation}
    \begin{split}
        \min \; f_1(x) = & \; 1.10471x_1^2x_2 + 0.04811x_3x_4 (14 + x_2) \\
        \min \; f_2(x) = & \; \frac{4PL^3}{Ex_4x_3^3} \\ 
        \min \; f_3(x) = & \; \displaystyle\sum_{i=1}^{4} \max \{ g_i(x), 0 \} \\
        g_1(x) = & \; \tau_{max} - \tau(x) \geq 0 \\
        g_2(x) = & \; \sigma_{max} - \sigma(x) \geq 0 \\
        g_3(x) = & \; x_4 - x_1 \geq 0 \\
        g_4(x) = & \; P_C(x) - P \geq 0 \\
        \tau(x) = & \; \sqrt{(\tau')^2 + \frac{2\tau'\tau''x_2}{2R} + (\tau'')^2} \\
        \tau' = & \; \frac{P}{\sqrt{2}x_1x_2} \\
        \tau'' = & \; \frac{MR}{J} \\
        M = & \; P \Big( L + \frac{x_2}{2} \Big) \\
        R = & \; \sqrt{\frac{x_2^2}{4} + \bigg( \frac{x_1 + x_3}{2} \bigg)^2 } \\
        J = & \; 2 \Bigg( \sqrt{2} x_1x_2 \bigg( \frac{x_2^2}{12} + \Big( \frac{x_1 + x_3}{2} \Big)^2 \bigg) \Bigg) \\
        \sigma(x) = & \; \frac{6PL}{x_4x_3^2} \\
        P_C(x) = & \; \frac{4.013E \sqrt{x_3^2x_4^6 / 36}}{L^2} \Bigg( 1 - \frac{x_3}{2L} \sqrt{\frac{E}{4G}} \Bigg)
    \end{split}
  \end{equation}

where $x_1, x_4 \in [0.125, 5]$ and $x_2, x_3 \in [0.1, 10]$. The four variables adjust the size of the beam. 
The parameters are given as follows: $P = 6 000$lb, $L = 14$in, $E = 30 \times 10^6$psi, $G = 12 \times 10^6$psi, 
$\tau_{max}$ = 13 600psi, and $\sigma_{max}$ = 30 000psi.

RE-33
-----------
Disc brake design problem

**Definition**

.. math::

  \begin{equation}
    \begin{split}
        \min \; f_1(x) = & \; 4.9 \times 10^{-5}(x_2^2 - x_1^2)(x_4 - 1) \\
        \min \; f_2(x) = & \; 9.82 \times 10^6 \bigg(\frac{x_2^2 - x_1^2}{x_3x_4(x_2^3 - x_1^3)} \bigg) \\ 
        \min \; f_3(x) = & \; \displaystyle\sum_{i=1}^{4} \max \{ g_i(x), 0 \} \\
        g_1(x) = & \; (x_2 - x_1) - 20 \geq 0 \\
        g_2(x) = & \; 0.4 - \frac{x_3}{3.14(x_2^2 - x_1^2)} \geq 0 \\
        g_3(x) = & \; 1 - \frac{2.22 \times 10^{-3}x_3 (x_2^3 - x_1^3)}{(x_2^2 - x_1^2)^2} \geq 0 \\
        g_4(x) = & \; \frac{2.66 \times 10^{-2}x_3 x_4 (x_2^3-x_1^3)}{(x_2^2-x_1^2)} - 900 \geq 0
    \end{split}
  \end{equation}

where $x_1 \in [55, 80],\; x_2 \in [75, 110],\; x_3 \in [1000, 3000]$ and $x_4 \in [11, 20]$. 
The four variables $(x_1, \; x_2,\; x_3,\; \text{and} \; x_4)$ represent the inner radius of the discs, 
the outer radius of the discs, the engaging force, and the number of friction surfaces, respectively.
    