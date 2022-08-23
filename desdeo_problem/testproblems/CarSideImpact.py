"""
Car-side impact problem.

Three-objective problem from:

Jain, H. & Deb, K. (2014). An Evolutionary Many-Objective Optimization Algorithm 
Using Reference-Point Based Nondominated Sorting Approach, Part II: Handling Constraints 
and Extending to an Adaptive Approach. IEEE transactions on evolutionary computation, 
18(4), 602-622. https://doi.org/10.1109/TEVC.2013.2281534 

Optional fourth objective from:

Tanabe, R. & Ishibuchi, H. (2020). An easy-to-use real-world 
multi-objective optimization problem suite. 
Applied soft computing, 89, 106078. 
https://doi.org/10.1016/j.asoc.2020.106078

Variable names from:

Deb, K., Gupta, S., Daum, D., Branke, J., Mall, A. & Padmanabhan, D. (2009). 
Reliability-Based Optimization Using Evolutionary Algorithms. 
IEEE transactions on evolutionary computation, 13(5), 1054-1074. 
https://doi.org/10.1109/TEVC.2009.2014361 

"""

from desdeo_problem.problem.Variable import Variable
from desdeo_problem.problem.Objective import ScalarObjective
from desdeo_problem.problem.Problem import MOProblem, ProblemBase
from desdeo_problem import ScalarConstraint, problem

import numpy as np

def car_side_impact(three_obj: bool = True, var_iv: np.array = np.array([1, 0.9, 1, 1, 1.75, 0.8, 0.8])) -> MOProblem:
    """ Car-side impact problem.
    
    Arguments:
        three_obj (bool): If true, utilize three objectives version. 
            If false, utilize four objectives version. Default is true.
        var_iv (np.array): Optional, initial variable values.
            Defaults are [1, 0.9, 1, 1, 1.75, 0.8, 0.8]. 
            x1 ∈ [0.5, 1.5], x2 ∈ [0.45, 1.35], x3 ∈ [0.5, 1.5], 
            x4 ∈ [0.5, 1.5], x5 ∈ [0.875, 2.625],
            x6 ∈ [0.4, 1.2] and x7 ∈ [0.4, 1.2].

    Returns:
        MOProblem: a problem object.
    """

    # Check the number of variables
    if (np.shape(np.atleast_2d(var_iv)[0]) != (7,)):
        raise RuntimeError("Number of variables must be seven")

    # Lower bounds
    lb = np.array([0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4])
    
    # Upper bounds
    ub = np.array([1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2])

    # Check the variable bounds
    if np.any(lb > var_iv) or np.any(ub < var_iv):
        raise ValueError("Initial variable values need to be between lower and upper bounds")
        
    def v_mbp(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            10.58 - 0.674 * x[:, 0] * x[:, 1] - 0.67275 * x[:, 1]
        )
    
    def v_fd(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            16.45 - 0.489 * x[:, 2] * x[:, 6] - 0.843 * x[:, 4] * x[:, 5]
        )

    # Objective functions
    def f_1(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            1.98 + 4.9 * x[:, 0] + 6.67 * x[:, 1] + 6.98 * x[:, 2] +
            4.01 * x[:, 3] + 1.78 * x[:, 4] + 10**-5 * x[:, 5] + 2.73 * x[:, 6]
        )

    def f_2(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            4.72 - 0.5 * x[:, 3] - 0.19 * x[:, 1] * x[:, 2]
        )
    
    def f_3(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            0.5 * (v_mbp(x) + v_fd(x))
        )

    # Constrain functions
    def g_1(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            1 - 1.16 + 0.3717 * x[:, 1] * x[:, 3] + 0.0092928 * x[:, 2]
        )

    def g_2(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            0.32 - 0.261 + 0.0159 * x[:, 0] * x[:, 1] + 0.06486 * x[:, 0] +
            0.019 * x[:, 1] * x[:, 6] - 0.0144 * x[:, 2] * x[:, 4] - 0.0154464 * x[:, 5]
        )

    def g_3(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            0.32 - 0.214 - 0.00817 * x[:, 4] + 0.045195 * x[:, 0] + 0.0135168 * x[:, 0] -
            0.03099 * x[:, 1] * x[:, 5] + 0.018 * x[:, 1] * x[:, 6] -
            0.007176 * x[:, 2] - 0.023232 * x[:, 2] + 0.00364 * x[:, 4] *
            x[:, 5] + 0.018 * x[:, 1]**2
        )
            
    def g_4(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            0.32 - 0.74 + 0.61 * x[:, 1] + 0.031296 * x[:, 2] +
            0.031872 * x[:, 6] - 0.227 * x[:, 1]**2
        )

    def g_5(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            32 - 28.98 - 3.818 * x[:, 2] + 4.2 * x[:, 0] * x[:, 1] -
            1.27296 * x[:, 5] + 2.68065 * x[:, 6]
        )

    def g_6(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            32 - 33.86 - 2.95 * x[:, 2] + 5.057 * x[:, 0] * x[:, 1] +
            3.795 * x[:, 1] + 3.4431 * x[:, 6] - 1.45728
        )

    def g_7(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            32 - 46.36 + 9.9 * x[:, 1] + 4.4505 * x[:, 0]
        )

    def g_8(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            4 - f_2(x)
        )

    def g_9(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            9.9 - v_mbp(x)
        )

    def g_10(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            15.7 - v_fd(x)
        )

    objective_1 = ScalarObjective(name="minimize the weight of the car", evaluator=f_1, maximize=[False])
    objective_2 = ScalarObjective(name="minimize the pubic force experienced by a passenger", 
        evaluator=f_2, maximize=[False])
    objective_3 = ScalarObjective(
        name="minimize the average velocity of the V-pillar responsible for whitstanding the impact load", 
        evaluator=f_3, maximize=[False])

    if three_obj:
        objectives = [objective_1, objective_2, objective_3]

        # Cons with three objective functions
        cons_1 = ScalarConstraint("c_1", 7, 3, g_1)
        cons_2 = ScalarConstraint("c_2", 7, 3, g_2)
        cons_3 = ScalarConstraint("c_3", 7, 3, g_3)
        cons_4 = ScalarConstraint("c_4", 7, 3, g_4)
        cons_5 = ScalarConstraint("c_5", 7, 3, g_5)
        cons_6 = ScalarConstraint("c_6", 7, 3, g_6)
        cons_7 = ScalarConstraint("c_7", 7, 3, g_7)
        cons_8 = ScalarConstraint("c_8", 7, 3, g_8)
        cons_9 = ScalarConstraint("c_9", 7, 3, g_9)
        cons_10 = ScalarConstraint("c_10", 7, 3, g_10)

    else:
        # If three_obj is false, then problem is with 4 objectives.
        
        # the sum of constraint violations
        def f_4(x: np.ndarray) -> np.ndarray:
            x = np.atleast_2d(x)
            sum1 = g_1(x)
            sum2 = g_2(x)
            sum3 = g_3(x)
            sum4 = g_4(x)
            sum5 = g_5(x)
            sum6 = g_6(x)
            sum7 = g_7(x)
            sum8 = g_8(x)
            sum9 = g_9(x)
            sum10 = g_10(x)
            sum1 = np.where(sum1 > 0, sum1, 0)
            sum2 = np.where(sum2 > 0, sum2, 0)
            sum3 = np.where(sum3 > 0, sum3, 0)
            sum4 = np.where(sum4 > 0, sum4, 0)
            sum5 = np.where(sum5 > 0, sum5, 0)
            sum6 = np.where(sum6 > 0, sum6, 0)
            sum7 = np.where(sum7 > 0, sum7, 0)
            sum8 = np.where(sum8 > 0, sum8, 0)
            sum9 = np.where(sum9 > 0, sum9, 0)
            sum10 = np.where(sum10 > 0, sum10, 0)
            return sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 + sum8 + sum9 + sum10

        objective_4 = ScalarObjective(name="the sum of the four constraint violations", evaluator=f_4, maximize=[False])

        objectives = [objective_1, objective_2, objective_3, objective_4]

        # Cons with four objective functions
        cons_1 = ScalarConstraint("c_1", 7, 4, g_1)
        cons_2 = ScalarConstraint("c_2", 7, 4, g_2)
        cons_3 = ScalarConstraint("c_3", 7, 4, g_3)
        cons_4 = ScalarConstraint("c_4", 7, 4, g_4)
        cons_5 = ScalarConstraint("c_5", 7, 4, g_5)
        cons_6 = ScalarConstraint("c_6", 7, 4, g_6)
        cons_7 = ScalarConstraint("c_7", 7, 4, g_7)
        cons_8 = ScalarConstraint("c_8", 7, 4, g_8)
        cons_9 = ScalarConstraint("c_9", 7, 4, g_9)
        cons_10 = ScalarConstraint("c_10", 7, 4, g_10)


    constraints = [cons_1, cons_2, cons_3, cons_4, cons_5, cons_6, cons_7, cons_8, cons_9, cons_10]

    x_1 = Variable("Thickness of B-Pillar inner", 1, 0.5, 1.5)
    x_2 = Variable("Thickness of B-Pillar reinforcement", 0.9, 0.45, 1.35)
    x_3 = Variable("Thickness of floor side inner", 1, 0.5, 1.5)
    x_4 = Variable("Thickness of cross members", 1, 0.5, 1.5)
    x_5 = Variable("Thickness of door beam", 1.75,  0.875, 2.625)
    x_6 = Variable("Thickness of door beltline reinforcement", 0.8, 0.4, 1.2)
    x_7 = Variable("Thickness of roof rail", 0.8, 0.4, 1.2)

    variables = [x_1, x_2, x_3, x_4, x_5, x_6, x_7]

    problem = MOProblem(variables=variables, objectives=objectives, constraints=constraints)

    return problem