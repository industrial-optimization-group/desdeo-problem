"""
The General Aviation Aircraft (GAA) problem.

T. W. Simpson, W. Chen, J. K. Allen, and F. Mistree (1996), 
"Conceptual design of a family of products through the use of the robust
concept exploration method," in 6th AIAA/USAF/NASA/ ISSMO Symposium on 
Multidiciplinary Analysis and Optimization, vol. 2, pp. 1535-1545.

T. W. Simpson, B. S. D'Souza (2004), "Assessing variable levels of platform 
commonality within a product family using a multiobjective genetic algorithm," 
Concurrent Engineering: Research and Applications, vol. 12, no. 2, pp. 119-130.

R. Shah, P. M. Reed, and T. W. Simpson (2011), "Many-objective evolutionary optimization 
and visual analytics for product family design," Multiobjective Evolutionary Optimisation 
for Product Design and Manufacturing, Springer, London, pp. 137-159.

M. Woodruff, T. W. Simpson, P. M. Reed (2013), "Diagnostic Analysis of Metamodels' 
Multivariate Dependencies and their Impacts in Many-Objective Design Optimization," 
Proceedings of the ASME 2013 IDETC/CIE Conference, Paper No. DETC2013-13125.

https://github.com/matthewjwoodruff/generalaviation

"""

from desdeo_problem.problem.Variable import Variable
from desdeo_problem.problem.Objective import ScalarObjective
from desdeo_problem.problem.Problem import MOProblem, ProblemBase
from desdeo_problem import ScalarConstraint, problem

import numpy as np

def gaa(var_iv: np.array) -> MOProblem:
    """The General Aviation Aircraft (GAA) problem.
    27 design variables, 10 objectives and 1 constraint.

    Arguments:
        var_iv (np.array): Optional, initial variable values.

    Returns:
        MOProblem: a problem object.
    """
    
    # Check the number of variables
    if (np.shape(np.atleast_2d(var_iv)[0]) != (27,)):
        raise RuntimeError("Number of variables must be 27")

    # Lower bounds
    lb = np.array([
        0.24, 7, 0, 5.5, 19, 85, 14, 3, 0.46,
	    0.24, 7, 0, 5.5, 19, 85, 14, 3, 0.46,
	    0.24, 7, 0, 5.5, 19, 85, 14, 3, 0.46
    ])
    
    # Upper bounds
    ub = np.array([
        0.48, 11, 6, 5.968, 25, 110, 20, 3.75, 1,
	    0.48, 11, 6, 5.968, 25, 110, 20, 3.75, 1,
	    0.48, 11, 6, 5.968, 25, 110, 20, 3.75, 1
    ])

    # Check the variable bounds
    if np.any(lb > var_iv) or np.any(ub < var_iv):
        raise ValueError("Initial variable values need to be between lower and upper bounds")

    # Scaled decision variables
    var_iv[0] = (var_iv[0] - 0.36) / 0.12
    var_iv[1] = (var_iv[1] - 9) / 2
    var_iv[2] = (var_iv[2] - 3) / 3
    var_iv[3] = (var_iv[3] - 5.734) / 0.234
    var_iv[4] = (var_iv[4] - 22) / 3
    var_iv[5] = (var_iv[5] - 97.5) / 12.5
    var_iv[6] = (var_iv[6] - 17) / 3
    var_iv[7] = (var_iv[7] - 3.375) / 0.375
    var_iv[8] = (var_iv[8] - 0.73) / 0.27

    var_iv[9] = (var_iv[9] - 0.36) / 0.12
    var_iv[10] = (var_iv[10] - 9) / 2
    var_iv[11] = (var_iv[11] - 3) / 3
    var_iv[12] = (var_iv[12] - 5.734) / 0.234
    var_iv[13] = (var_iv[13] - 22) / 3
    var_iv[14] = (var_iv[14] - 97.5) / 12.5
    var_iv[15] = (var_iv[15] - 17) / 3
    var_iv[16] = (var_iv[16] - 3.375) / 0.375
    var_iv[17] = (var_iv[17] - 0.73) / 0.27

    var_iv[18] = (var_iv[18] - 0.36) / 0.12
    var_iv[19] = (var_iv[19] - 9) / 2
    var_iv[20] = (var_iv[20] - 3) / 3
    var_iv[21] = (var_iv[21] - 5.734) / 0.234
    var_iv[22] = (var_iv[22] - 22) / 3
    var_iv[23] = (var_iv[23] - 97.5) / 12.5
    var_iv[24] = (var_iv[24] - 17) / 3
    var_iv[25] = (var_iv[25] - 3.375) / 0.375
    var_iv[25] = (var_iv[26] - 0.73) / 0.27

    # Objective functions
    def f_1(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)

        return
    
    def f_2(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return

    def f_3(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return

    def f_4(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return

    def f_5(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return

    def f_6(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return

    def f_7(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return

    def f_8(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return

    def f_9(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return

    def f_10(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return

    # Constraint function
    def g_1(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return constraints

    objective_1 = ScalarObjective(name="", evaluator=f_1, maximize=[False])
    objective_2 = ScalarObjective(name="", evaluator=f_2, maximize=[False])
    objective_3 = ScalarObjective(name="", evaluator=f_3, maximize=[False])
    objective_4 = ScalarObjective(name="", evaluator=f_4, maximize=[False])
    objective_5 = ScalarObjective(name="", evaluator=f_5, maximize=[False])
    objective_6 = ScalarObjective(name="", evaluator=f_6, maximize=[False])
    objective_7 = ScalarObjective(name="", evaluator=f_7, maximize=[False])
    objective_8 = ScalarObjective(name="", evaluator=f_8, maximize=[False])
    objective_9 = ScalarObjective(name="", evaluator=f_9, maximize=[False])
    objective_10 = ScalarObjective(name="", evaluator=f_10, maximize=[False])

    objectives = [objective_1, objective_2, objective_3, objective_4, objective_5, 
                objective_6, objective_7, objective_8, objective_9, objective_10]

    cons_1 = ScalarConstraint("c_1", 27, 10, g_1)
    
    constraints = [cons_1]

    x_1 = Variable("cruise speed 2 seats", 0.36, 0.24, 0.48)
    x_2 = Variable("aspect ratio 2 seats", 9, 7, 11)
    x_3 = Variable("sweep angle 2 seats", 3, 0, 6)
    x_4 = Variable("propeller diameter 2 seats", 5.7, 5.5, 5.968)
    x_5 = Variable("wing loading 2 seats", 22, 19, 25)
    x_6 = Variable("engine activity factor 2 seats", 97, 85, 110)
    x_7 = Variable("seat width 2 seats", 17, 14, 20)
    x_8 = Variable("tail length/diameter ratio 2 seats", 3.4, 3, 3.75)
    x_9 = Variable("taper ratio 2 seats", 0.75, 0.46, 1)
    x_10 = Variable("cruise speed 4 seats", 0.36, 0.24, 0.48)
    x_11 = Variable("aspect ratio 4 seats", 9, 7, 11)
    x_12 = Variable("sweep angle 4 seats", 3, 0, 6)
    x_13 = Variable("propeller diameter 4 seats", 5.7, 5.5, 5.968)
    x_14 = Variable("wing loading 4 seats", 22, 19, 25)
    x_15 = Variable("engine activity factor 4 seats", 97, 85, 110)
    x_16 = Variable("seat width 4 seats", 17, 14, 20)
    x_17 = Variable("tail length/diameter ratio 4 seats", 3.4, 3, 3.75)
    x_18 = Variable("taper ratio 4 seats", 0.75, 0.46, 1)
    x_19 = Variable("cruise speed 6 seats", 0.36, 0.24, 0.48)
    x_20 = Variable("aspect ratio 6 seats", 9, 7, 11)
    x_21 = Variable("sweep angle 6 seats", 3, 0, 6)
    x_22 = Variable("propeller diameter 6 seats", 5.7, 5.5, 5.968)
    x_23 = Variable("wing loading 6 seats", 22, 19, 25)
    x_24 = Variable("engine activity factor 6 seats", 97, 85, 110)
    x_25 = Variable("seat width 6 seats", 17, 14, 20)
    x_26 = Variable("tail length/diameter ratio 6 seats", 3.4, 3, 3.75)
    x_27 = Variable("taper ratio 6 seats", 0.75, 0.46, 1)

    variables = [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, 
                x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, 
                x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27]

    problem = MOProblem(variables=variables, objectives=objectives, constraints=constraints)


    return problem
