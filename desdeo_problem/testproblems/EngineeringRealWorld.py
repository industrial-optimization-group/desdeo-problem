import numpy as np
from desdeo_problem.problem import MOProblem
from desdeo_problem.problem import Variable
from desdeo_problem.problem import ScalarObjective
from desdeo_problem.problem.Problem import MOProblem, ProblemBase

"""
A real-world multi-objective problem suite (the RE benchmark set)

Tanabe, R. & Ishibuchi, H. (2020). An easy-to-use real-world multi-objective 
optimization problem suite. Applied soft computing, 89, 106078. 
https://doi.org/10.1016/j.asoc.2020.106078 

https://github.com/ryojitanabe/reproblems/blob/master/reproblem_python_ver/reproblem.py
https://github.com/ryojitanabe/reproblems/blob/master/doc/re-supplementary_file.pdf

"""

def re21(var_iv: np.array = np.array([2, 2, 2, 2])) -> MOProblem:
    """ Four bar truss design problem. 
    Two objectives and four variables.
    
    Arguments:
        var_iv (np.array): Optional, initial variable values.
            Defaults are [2, 2, 2, 2]. x1, x4 ∈ [a, 3a], x2, x3 ∈ [√2 a, 3a]
            and a = F / sigma
    Returns:
        MOProblem: a problem object.
    """

    # Parameters
    F = 10.0
    sigma = 10.0
    E = 2.0 * 1e5
    L = 200.0
    a = F / sigma

    # Check the number of variables
    if (np.shape(np.atleast_2d(var_iv)[0]) != (4,)):
        raise RuntimeError("Number of variables must be four")

    # Lower bounds
    lb = np.array([a, np.sqrt(2) * a, np.sqrt(2) * a, a])
    
    # Upper bounds
    ub = np.array([3 * a, 3 * a, 3 * a, 3 * a])

    # Check the variable bounds
    if np.any(lb > var_iv) or np.any(ub < var_iv):
        raise ValueError("Initial variable values need to be between lower and upper bounds")

    def f_1(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return L * ((2 * x[:, 0]) + np.sqrt(2.0) * x[:, 1] + np.sqrt(x[:, 2]) + x[:, 3])

    def f_2(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return ((F * L) / E) * ((2.0 / x[:, 0]) + 
            (2.0 * np.sqrt(2.0) / x[:, 1]) - (2.0 * np.sqrt(2.0) / x[:, 2]) + (2.0 / x[:, 3]))

    objective_1 = ScalarObjective(name="minimize the structural volume", evaluator=f_1, maximize=[False])
    objective_2 = ScalarObjective(name="minimize the joint displacement", evaluator=f_2, maximize=[False])

    objectives = [objective_1, objective_2]

    # The four variables determine the length of four bars
    x_1 = Variable("x_1", 2 * a, a, 3 * a)
    x_2 = Variable("x_2", 2 * a, (np.sqrt(2.0) * a), 3 * a)
    x_3 = Variable("x_3", 2 * a, (np.sqrt(2.0) * a), 3 * a)
    x_4 = Variable("x_4", 2 * a, a, 3 * a)

    variables = [x_1, x_2, x_3, x_4]

    problem = MOProblem(variables=variables, objectives=objectives)

    return problem



def re22(var_iv: np.array = np.array([7.2, 10, 20])) -> MOProblem:
    """ Reinforced concrete beam design problem.
    2 objectives, 3 variables and 2 constraints.

    Arguments:
        var_iv (np.array): Optional, initial variable values.
            Defaults are [7.2, 10, 20]. x2 ∈ [0, 20] and x3 ∈ [0, 40].
            x1 has a pre-defined discrete value from 0.2 to 15.
    Returns:
        MOProblem: a problem object.
    """

    # Check the number of variables
    if (np.shape(np.atleast_2d(var_iv)[0]) != (3,)):
        raise RuntimeError("Number of variables must be three")

    # Lower bounds
    lb = np.array([0.2, 0, 0])

    # Upper bounds
    ub = np.array([15, 20, 40])

    # Check the variable bounds
    if np.any(lb > var_iv) or np.any(ub < var_iv):
        raise ValueError("Initial variable values need to be between lower and upper bounds")

    # x1 pre-defined discrete values
    feasible_vals = np.array([0.20, 0.31, 0.40, 0.44, 0.60, 0.62, 0.79, 0.80, 0.88, 0.93,
                              1.0, 1.20, 1.24, 1.32, 1.40, 1.55, 1.58, 1.60, 1.76, 1.80,
                              1.86, 2.0, 2.17, 2.20, 2.37, 2.40, 2.48, 2.60, 2.64, 2.79,
                              2.80, 3.0, 3.08, 3.10, 3.16, 3.41, 3.52, 3.60, 3.72, 3.95,
                              3.96, 4.0, 4.03, 4.20, 4.34, 4.40, 4.65, 4.74, 4.80, 4.84,
                              5.0, 5.28, 5.40, 5.53, 5.72, 6.0, 6.16, 6.32, 6.60, 7.11,
                              7.20, 7.80, 7.90, 8.0, 8.40, 8.69, 9.0, 9.48, 10.27, 11.0,
                              11.06, 11.85, 12.0, 13.0, 14.0, 15.0])

    # Returns discrete value for x1
    def feas_val(x: np.ndarray) -> np.array:
        fv_2d = np.repeat(np.atleast_2d(feasible_vals), x.shape[0], axis=0)
        idx = np.abs(fv_2d.T - x[:, 0]).argmin(axis=0)
        x[:, 0] = feasible_vals[idx]
        return x

    # Constrain functions
    def g_1(x: np.ndarray, _=None) -> np.ndarray:
        x = np.atleast_2d(x)
        x = feas_val(x)
        return x[:, 0] * x[:, 2] - 7.735 * (x[:, 0] ** 2 / x[:, 1]) - 180

    def g_2(x: np.ndarray, _=None) -> np.ndarray:
        x = np.atleast_2d(x)
        return 4 - x[:, 2] / x[:, 1]

    # Objective functions
    def f_1(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        x = feas_val(x)
        return (29.4 * x[:, 0]) + (0.6 * x[:, 1] * x[:, 2])

    def f_2(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        sum1 = g_1(x)
        sum2 = g_2(x)
        sum1 = np.where(sum1 < 0, -sum1, 0)
        sum2 = np.where(sum2 < 0, -sum2, 0)
        return sum1 + sum2

    objective_1 = ScalarObjective(name="minimize the total cost of concrete and reinforcing steel of the beam",
                                  evaluator=f_1, maximize=[False])

    objective_2 = ScalarObjective(name="the sum of the four constraint violations", evaluator=f_2, maximize=[False])

    objectives = [objective_1, objective_2]

    # cons_1 = ScalarConstraint("c_1", 3, 2, g_1)
    # cons_2 = ScalarConstraint("c_2", 3, 2, g_2)
    #
    # constraints = [cons_1, cons_2]

    x_1 = Variable("the area of the reinforcement", 7.2, 0.2, 15)
    x_2 = Variable("the width of the beam", 10, 0, 20)
    x_3 = Variable("the depth of the beam", 20, 0, 40)

    variables = [x_1, x_2, x_3]

    problem = MOProblem(variables=variables, objectives=objectives)

    return problem


def re23(var_iv: np.array = np.array([50, 50, 100, 120])) -> MOProblem:
    """ Pressure vesssel design problem.
    2 objectives, 4 variables and 3 constraints.

    Arguments:
        var_iv (np.array): Optional, initial variable values.
            Defaults are [50, 50, 100, 120]. x1 and x2 ∈ {1, ..., 100},
            x3 ∈ [10, 200] and x4 ∈ [10, 240].
            x1 and x2 are integer multiples of 0.0625.
    Returns:
        MOProblem: a problem object.
    """

    # Check the number of variables
    if (np.shape(np.atleast_2d(var_iv)[0]) != (4,)):
        raise RuntimeError("Number of variables must be four")

    # Lower bounds
    lb = np.array([1, 1, 10, 10])
    ub = np.array([100, 100, 200, 240])

    # Check the variable bounds
    if np.any(lb > var_iv) or np.any(ub < var_iv):
        raise ValueError("Initial variable values need to be between lower and upper bounds")

    # Constrain functions
    def g_1(x: np.ndarray, _=None) -> np.ndarray:
        x = np.atleast_2d(x)
        x = x.astype(float)
        x[:, 0] = 0.0625 * (np.round(x[:, 0]))
        return x[:, 0] - (0.0193 * x[:, 2])

    def g_2(x: np.ndarray, _=None) -> np.ndarray:
        x = np.atleast_2d(x)
        x = x.astype(float)
        x[:, 1] = 0.0625 * (np.round(x[:, 1]))
        return x[:, 1] - (0.00954 * x[:, 2])

    def g_3(x: np.ndarray, _=None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (np.pi * x[:, 2] ** 2 * x[:, 3]) + ((4 / 3) * np.pi * x[:, 2] ** 3) - 1296000

    # Objective functions
    def f_1(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        x = x.astype(float)
        x[:, 0] = 0.0625 * (np.round(x[:, 0]))
        x[:, 1] = 0.0625 * (np.round(x[:, 1]))
        return (
                (0.6224 * x[:, 0] * x[:, 2] * x[:, 3]) + (1.7781 * x[:, 1] * x[:, 2] ** 2) +
                (3.1661 * x[:, 0] ** 2 * x[:, 3]) + (19.84 * x[:, 0] ** 2 * x[:, 2])
        )

    def f_2(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        sum1 = g_1(x)
        sum2 = g_2(x)
        sum3 = g_3(x)
        sum1 = np.where(sum1 < 0, -sum1, 0)
        sum2 = np.where(sum2 < 0, -sum2, 0)
        sum3 = np.where(sum3 < 0, -sum3, 0)
        return sum1 + sum2 + sum3

    objective_1 = ScalarObjective(name="minimize to total cost of a clyndrical pressure vessel", evaluator=f_1,
                                  maximize=[False])
    objective_2 = ScalarObjective(name="the sum of the four constraint violations", evaluator=f_2, maximize=[False])

    objectives = [objective_1, objective_2]

    # cons_1 = ScalarConstraint("c_1", 4, 2, g_1)
    # cons_2 = ScalarConstraint("c_2", 4, 2, g_2)
    # cons_3 = ScalarConstraint("c_3", 4, 2, g_3)D
    #
    # constraints = [cons_1, cons_2, cons_3]

    x_1 = Variable("the thicknesses of the shell", 50, 1, 100)
    x_2 = Variable("the the head of pressure vessel", 50, 1, 100)
    x_3 = Variable("the inner radius", 100, 10, 200)
    x_4 = Variable("the length of the cylindrical section", 120, 10, 240)

    variables = [x_1, x_2, x_3, x_4]

    problem = MOProblem(variables=variables, objectives=objectives)

    return problem


def re24(var_iv: np.array = np.array([2, 25])) -> MOProblem:
    """ Hatch cover design problem.
    2 objectives, 2 variables and 4 constraints.

    Arguments:
        var_iv (np.array): Optional, initial variable values.
            Defaults are [2, 25]. x1 ∈ [0.5, 4] and
            x2 ∈ [4, 50].
    Returns:
        MOProblem: a problem object.
    """

    # Check the number of variables
    if (np.shape(np.atleast_2d(var_iv)[0]) != (2,)):
        raise RuntimeError("Number of variables must be two")

    # Lower bounds
    lb = np.array([0.5, 4])

    # Upper bounds
    ub = np.array([4, 50])

    # Check the variable bounds
    if np.any(lb > var_iv) or np.any(ub < var_iv):
        raise ValueError("Initial variable values need to be between lower and upper bounds")

    # Constrain functions
    def g_1(x: np.ndarray, _=None) -> np.ndarray:
        x = np.atleast_2d(x)
        return 1.0 - ((4500 / (x[:, 0] * x[:, 1])) / 700)

    def g_2(x: np.ndarray, _=None) -> np.ndarray:
        x = np.atleast_2d(x)
        return 1.0 - ((1800 / x[:, 1]) / 450)

    def g_3(x: np.ndarray, _=None) -> np.ndarray:
        x = np.atleast_2d(x)
        return 1.0 - (((56.2 * 10000) / (700000 * x[:, 0] * x[:, 1] ** 2)) / 1.5)

    def g_4(x: np.ndarray, _=None) -> np.ndarray:
        x = np.atleast_2d(x)
        return 1.0 - ((4500 / (x[:, 0] * x[:, 1])) / ((700000 * x[:, 0] ** 2) / 100))

    # Objective functions
    def f_1(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return x[:, 0] + 120 * x[:, 1]

    def f_2(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        sum1 = g_1(x)
        sum2 = g_2(x)
        sum3 = g_3(x)
        sum4 = g_4(x)
        sum1 = np.where(sum1 < 0, -sum1, 0)
        sum2 = np.where(sum2 < 0, -sum2, 0)
        sum3 = np.where(sum3 < 0, -sum3, 0)
        sum4 = np.where(sum4 < 0, -sum4, 0)
        return sum1 + sum2 + sum3 + sum4

    objective_1 = ScalarObjective(name="to minimize the weight of the hatch cover", evaluator=f_1, maximize=[False])
    objective_2 = ScalarObjective(name="the sum of the four constraint violations", evaluator=f_2, maximize=[False])

    objectives = [objective_1, objective_2]

    # cons_1 = ScalarConstraint("c_1", 2, 2, g_1)
    # cons_2 = ScalarConstraint("c_2", 2, 2, g_2)
    # cons_3 = ScalarConstraint("c_3", 2, 2, g_3)
    # cons_4 = ScalarConstraint("c_4", 2, 2, g_4)
    #
    # constraints = [cons_1, cons_2, cons_3, cons_4]

    x_1 = Variable("the flange thickness", 2, 0.5, 4)
    x_2 = Variable("the beam height", 25, 4, 50)

    variables = [x_1, x_2]

    problem = MOProblem(variables=variables, objectives=objectives)

    return problem


def re25(var_iv: np.array = np.array([35, 15, 0.207])) -> MOProblem:
    """ Coil compression spring design problem.
    2 objectives, 3 variables and 6 constraints.

    Arguments:
        var_iv (np.array): Optional, initial variable values.
            Defaults are [35, 15, 0.207]. x1 ∈ {1, ..., 70} and x2 ∈ [0.6, 30].
            x3 has a pre-defined discrete value from 0.009 to 0.5.
    Returns:
        MOProblem: a problem object.
    """

    # Check the number of variables
    if (np.shape(np.atleast_2d(var_iv)[0]) != (3,)):
        raise RuntimeError("Number of variables must be three")

    # Lower bounds
    lb = np.array([1, 0.6, 0.009])

    # Upper bounds
    ub = np.array([70, 30, 0.5])

    # Check the variable bounds
    if np.any(lb > var_iv) or np.any(ub < var_iv):
        raise ValueError("Initial variable values need to be between lower and upper bounds")

    # x3 pre-defined discrete values
    feasible_vals = np.array([0.009, 0.0095, 0.0104, 0.0118, 0.0128, 0.0132, 0.014, 0.015,
                              0.0162, 0.0173, 0.018, 0.02, 0.023, 0.025, 0.028, 0.032, 0.035,
                              0.041, 0.047, 0.054, 0.063, 0.072, 0.08, 0.092, 0.105, 0.12,
                              0.135, 0.148, 0.162, 0.177, 0.192, 0.207, 0.225, 0.244, 0.263,
                              0.283, 0.307, 0.331, 0.362, 0.394, 0.4375, 0.5])

    # Returns discrete value for x3
    def feas_val(x: np.ndarray) -> np.array:
        fv_2d = np.repeat(np.atleast_2d(feasible_vals), x.shape[0], axis=0)
        idx = np.abs(fv_2d.T - x[:, 2]).argmin(axis=0)
        x[:, 2] = feasible_vals[idx]
        return x

    # Constrain functions
    def g_1(x: np.ndarray, _=None) -> np.ndarray:
        x = np.atleast_2d(x)
        x = feas_val(x)
        return (
                -((8 * (((4.0 * (x[:, 1] / x[:, 2]) - 1) /
                         (4.0 * (x[:, 1] / x[:, 2]) - 4)) +
                        ((0.615 * x[:, 2]) / x[:, 1])) * 1000 * x[:, 1])
                  / (np.pi * x[:, 2] ** 3)) + 189000
        )

    def g_2(x: np.ndarray, _=None) -> np.ndarray:
        x = np.atleast_2d(x)
        x = feas_val(x)
        return (
                - (1000 / ((11.5 * 10 ** 6 * x[:, 2] ** 4) / (8 * np.round(x[:, 0]) * x[:, 1] ** 3))) + 1.05 * (
                    np.round(x[:, 0]) + 2) * x[:, 2] + 14
        )

    def g_3(x: np.ndarray, _=None) -> np.ndarray:
        x = np.atleast_2d(x)
        x = feas_val(x)
        return (
                -3 + (x[:, 1] / x[:, 2])
        )

    def g_4(x: np.ndarray, _=None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
                - (300 / ((11.5 * 10 ** 6 * x[:, 2] ** 4) / (8 * np.round(x[:, 0]) * x[:, 1] ** 3))) + 6
        )

    def g_5(x: np.ndarray, _=None) -> np.ndarray:
        x = np.atleast_2d(x)
        x = feas_val(x)
        return (
                -(300 / ((11.5 * 10 ** 6 * x[:, 2] ** 4) / (8 * np.round(x[:, 0]) * x[:, 1] ** 3))) - (
                    (1000 - 300) / ((11.5 * 10 ** 6 * x[:, 2] ** 4) / (8 * np.round(x[:, 0]) * x[:, 1] ** 3))) - (
                            1.05 * (np.round(x[:, 0]) + 2) * x[:, 2]) + (
                            (1000 / ((11.5 * 10 ** 6 * x[:, 2] ** 4) / (8 * np.round(x[:, 0]) * x[:, 1] ** 3))) + (
                                1.05 * (np.round(x[:, 0]) + 2) * x[:, 2]))
        )

    def g_6(x: np.ndarray, _=None) -> np.ndarray:
        x = np.atleast_2d(x)
        x = feas_val(x)
        return (
                -1.25 + ((1000 - 300) / ((11.5 * 10 ** 6 * x[:, 2] ** 4) / (8 * np.round(x[:, 0]) * x[:, 1] ** 3)))
        )

    # Objective functions
    def f_1(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        x = feas_val(x)
        return ((np.pi * np.pi * x[:, 1] * x[:, 2] ** 2 * ((np.round(x[:, 0])) + 2)) / 4.0)

    def f_2(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        sum1 = g_1(x)
        sum2 = g_2(x)
        sum3 = g_3(x)
        sum4 = g_4(x)
        sum5 = g_5(x)
        sum6 = g_6(x)
        sum1 = np.where(sum1 < 0, -sum1, 0)
        sum2 = np.where(sum2 < 0, -sum2, 0)
        sum3 = np.where(sum3 < 0, -sum3, 0)
        sum4 = np.where(sum4 < 0, -sum4, 0)
        sum5 = np.where(sum5 < 0, -sum5, 0)
        sum6 = np.where(sum6 < 0, -sum6, 0)
        return sum1 + sum2 + sum3 + sum4 + sum5 + sum6

    objective_1 = ScalarObjective(
        name="minimize the volume of spring steel wire which is used to manufacture the spring",
        evaluator=f_1, maximize=[False])

    objective_2 = ScalarObjective(name="the sum of the four constraint violations", evaluator=f_2, maximize=[False])

    objectives = [objective_1, objective_2]

    # cons_1 = ScalarConstraint("c_1", 3, 2, g_1)
    # cons_2 = ScalarConstraint("c_2", 3, 2, g_2)
    # cons_3 = ScalarConstraint("c_3", 3, 2, g_3)
    # cons_4 = ScalarConstraint("c_4", 3, 2, g_4)
    # cons_5 = ScalarConstraint("c_5", 3, 2, g_5)
    # cons_6 = ScalarConstraint("c_6", 3, 2, g_6)
    #
    # constraints = [cons_1, cons_2, cons_3, cons_4, cons_5, cons_6]

    x_1 = Variable("the number of spring coils", 35, 1, 70)
    x_2 = Variable("the outside diameter of the spring", 15, 0.6, 30)
    x_3 = Variable("the spring wire diameter", 0.207, 0.009, 0.5)

    variables = [x_1, x_2, x_3]

    problem = MOProblem(variables=variables, objectives=objectives)

    return problem


def re31(var_iv: np.array = np.array([50.0, 50.0, 2.0])) -> MOProblem:
    """ Two bar truss design problem.

    Arguments:
        var_iv (np.array): Optional, initial variable values.
            Defaults are [50.0, 50.0, 2.0]. x1 and x2 ∈ [0.00001, 100]
            and x3 ∈ [1.0, 3.0].
    Returns:
        MOProblem: a problem object.
    """

    # Check the number of variables
    if (np.shape(np.atleast_2d(var_iv)[0]) != (3,)):
        raise RuntimeError("Number of variables must be three")

    # Lower bounds
    lb = np.array([0.00001, 0.00001, 1.0])

    # Upper bounds
    ub = np.array([100.0, 100.0, 3.0])

    # Check the variable bounds
    if np.any(lb > var_iv) or np.any(ub < var_iv):
        raise ValueError("Initial variable values need to be between lower and upper bounds")

    # Objective functions
    def f_1(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
                x[:, 0] * np.sqrt(16 + x[:, 2] ** 2)
                + x[:, 1] * np.sqrt(1 + x[:, 2] ** 2)
        )

    def f_2(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            (20 * np.sqrt(16 + x[:, 2] ** 2)
             / (x[:, 2] * x[:, 0]))
        )

    # Constrain functions
    def g_1(x: np.ndarray, _=None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
                0.1 - f_1(x)
        )

    def g_2(x: np.ndarray, _=None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
                10 ** 5 - f_2(x)
        )

    def g_3(x: np.ndarray, _=None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
                10 ** 5
                - ((80 * np.sqrt(1 + x[:, 2] ** 2)
                    / (x[:, 2] * x[:, 1])))
        )

    # Third objective
    def f_3(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        sum1 = g_1(x)
        sum2 = g_2(x)
        sum3 = g_3(x)
        sum1 = np.where(sum1 < 0, -sum1, 0)
        sum2 = np.where(sum2 < 0, -sum2, 0)
        sum3 = np.where(sum3 < 0, -sum3, 0)
        return sum1 + sum2 + sum3

    objective_1 = ScalarObjective(name="minimize the structural weight", evaluator=f_1, maximize=[False])
    objective_2 = ScalarObjective(name="minimize the resultant displacement of join", evaluator=f_2, maximize=[False])
    objective_3 = ScalarObjective(name="the sum of the four constraint violations", evaluator=f_3, maximize=[False])

    objectives = [objective_1, objective_2, objective_3]

    # cons_1 = ScalarConstraint("c_1", 3, 3, g_1)
    # cons_2 = ScalarConstraint("c_2", 3, 3, g_2)
    # cons_3 = ScalarConstraint("c_3", 3, 3, g_3)
    #
    # constraints = [cons_1, cons_2, cons_3]

    x_1 = Variable("the length of the bar", 50.0, 0.00001, 100)
    x_2 = Variable("the length of the bar", 50.0, 0.00001, 100)
    x_3 = Variable("the spring wire diameter", 2.0, 1.0, 3.0)

    variables = [x_1, x_2, x_3]

    problem = MOProblem(variables=variables, objectives=objectives)

    return problem


def re32(var_iv: np.array = np.array([2.5, 5.0, 5.0, 2.5])) -> MOProblem:
    """ Welded beam design problem.

    Arguments:
        var_iv (np.array): Optional, initial variable values.
            Defaults are [2.5, 5.0, 5.0, 2.5]. x1, x4 ∈ [0.125, 5]
            and x2, x3 ∈ [0.1, 10.0].
    Returns:
        MOProblem: a problem object.
    """

    # Check the number of variables
    if (np.shape(np.atleast_2d(var_iv)[0]) != (4,)):
        raise RuntimeError("Number of variables must be four")

    # Lower bounds
    lb = np.array([0.125, 0.1, 0.1, 0.125])

    # Upper bounds
    ub = np.array([5, 10, 10, 5])

    # Check the variable bounds
    if np.any(lb > var_iv) or np.any(ub < var_iv):
        raise ValueError("Initial variable values need to be between lower and upper bounds")

    def tau(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            np.sqrt(
                (6000 / (np.sqrt(2) * x[:, 0] * x[:, 1])) ** 2 +
                ((2 * (6000 / (np.sqrt(2) * x[:, 0] * x[:, 1])) *
                  ((6000 * (14 + (x[:, 1] / 2))) *
                   (np.sqrt(((x[:, 1] ** 2) / 4.0) +
                            ((x[:, 0] + x[:, 2]) / 2) ** 2))) /
                  (2 * (np.sqrt(2) * x[:, 0] * x[:, 1] *
                        ((x[:, 1] ** 2) / 12 + ((x[:, 0] + x[:, 2]) / 2) ** 2))) * x[:, 1]) /
                 (2 * (np.sqrt(((x[:, 1] ** 2) / 4.0) + ((x[:, 0] + x[:, 2]) / 2) ** 2)))) +
                (((6000 * (14 + (x[:, 1] / 2))) *
                  (np.sqrt(((x[:, 1] ** 2) / 4.0) +
                           ((x[:, 0] + x[:, 2]) / 2) ** 2))) /
                 (2 * (np.sqrt(2) * x[:, 0] * x[:, 1] *
                       ((x[:, 1] ** 2) / 12 + ((x[:, 0] + x[:, 2]) / 2) ** 2)))) ** 2
            )
        )

    def sigma(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
                (6 * 6000 * 14) / (x[:, 3] * x[:, 2] ** 2)
        )

    def p_c(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
                ((4.013 * 30 * 10 ** 6 * np.sqrt((x[:, 2] ** 2 *
                                                  x[:, 3] ** 6) / 36)) / (14 ** 2)) *
                (1 - (x[:, 2] / (2 * 14)) * np.sqrt((30 * 10 ** 6) /
                                                    (4 * 12 * 10 ** 6)))
        )

    # Constrain functions
    def g_1(x: np.ndarray, _=None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
                13600 - tau(x)
        )

    def g_2(x: np.ndarray, _=None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
                30000 - sigma(x)
        )

    def g_3(x: np.ndarray, _=None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
                x[:, 3] - x[:, 0]
        )

    def g_4(x: np.ndarray, _=None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
                p_c(x) - 6000
        )

    # Objective functions
    def f_1(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
                1.10471 * x[:, 0] ** 2 * x[:, 1]
                + 0.04811 * x[:, 2] * x[:, 3] *
                (14 + x[:, 1])
        )

    def f_2(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
                (4 * 6000 * 14 ** 3) /
                (30 * 10 ** 6 * x[:, 3] * x[:, 2] ** 3)
        )

    def f_3(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        sum1 = g_1(x)
        sum2 = g_2(x)
        sum3 = g_3(x)
        sum4 = g_4(x)
        sum1 = np.where(sum1 < 0, -sum1, 0)
        sum2 = np.where(sum2 < 0, -sum2, 0)
        sum3 = np.where(sum3 < 0, -sum3, 0)
        sum4 = np.where(sum4 < 0, -sum4, 0)
        return sum1 + sum2 + sum3 + sum4

    objective_1 = ScalarObjective(name="minimize cost of a welded beam", evaluator=f_1, maximize=[False])
    objective_2 = ScalarObjective(name="minimize end deflection of a welded beam", evaluator=f_2, maximize=[False])
    objective_3 = ScalarObjective(name="the sum of the four constraint violations", evaluator=f_3, maximize=[False])

    objectives = [objective_1, objective_2, objective_3]

    # cons_1 = ScalarConstraint("c_1", 4, 3, g_1)
    # cons_2 = ScalarConstraint("c_2", 4, 3, g_2)
    # cons_3 = ScalarConstraint("c_3", 4, 3, g_3)
    # cons_4 = ScalarConstraint("c_4", 4, 3, g_4)
    #
    # constraints = [cons_1, cons_2, cons_3, cons_4]

    # Variables adjust the size of the beam
    x_1 = Variable("x_1", 2.5, 0.125, 5)
    x_2 = Variable("x_2", 5.0, 0.1, 10.0)
    x_3 = Variable("x_3", 5.0, 0.1, 10.0)
    x_4 = Variable("x_4", 2.5, 0.125, 5)

    variables = [x_1, x_2, x_3, x_4]

    # problem = MOProblem(variables=variables, objectives=objectives, constraints=constraints)
    problem = MOProblem(variables=variables, objectives=objectives)

    return problem


def re33(var_iv: np.array = np.array([67.5, 92.5, 2000, 15])) -> MOProblem:
    """ Disc brake design problem.

    Arguments:
        var_iv (np.array): Optional, initial variable values.
            Defaults are [67.5, 92.5, 2000, 15]. x1 ∈ [55, 80],
            x2 ∈ [75, 110], x3 ∈ [1000, 3000] and x4 ∈ [11, 20].

    Returns:
        MOProblem: a problem object.
    """

    # Check the number of variables
    if (np.shape(np.atleast_2d(var_iv)[0]) != (4,)):
        raise RuntimeError("Number of variables must be four")

    # Lower bounds
    lb = np.array([55, 75, 1000, 11])

    # Upper bounds
    ub = np.array([80, 110, 3000, 20])

    # Check the variable bounds
    if np.any(lb > var_iv) or np.any(ub < var_iv):
        raise ValueError("Initial variable values need to be between lower and upper bounds")

    # Constrain functions
    def g_1(x: np.ndarray, _=None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
                (x[:, 1] - x[:, 0]) - 20
        )

    def g_2(x: np.ndarray, _=None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
                0.4 - (x[:, 2] / (3.14 * (x[:, 1] ** 2 - x[:, 0] ** 2)))
        )

    def g_3(x: np.ndarray, _=None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
                1 - ((2.22 * 10 ** -3 * x[:, 2] * (x[:, 1] ** 3 - x[:, 0] ** 3)) /
                     (x[:, 1] ** 2 - x[:, 0] ** 2) ** 2)
        )

    def g_4(x: np.ndarray, _=None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
                ((2.66 * 10 ** -2 * x[:, 2] * x[:, 3] * (x[:, 1] ** 3 - x[:, 0] ** 3)) /
                 (x[:, 1] ** 2 - x[:, 0] ** 2)) - 900
        )

    # Objective functions
    def f_1(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
                4.9 * 10 ** -5 * (x[:, 1] ** 2 - x[:, 0] ** 2) * (x[:, 3] - 1)
        )

    def f_2(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
                9.82 * 10 ** 6 * ((x[:, 1] ** 2 - x[:, 0] ** 2) /
                                  (x[:, 2] * x[:, 3] * (x[:, 1] ** 3 - x[:, 0] ** 3)))
        )

    def f_3(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        sum1 = g_1(x)
        sum2 = g_2(x)
        sum3 = g_3(x)
        sum4 = g_4(x)
        sum1 = np.where(sum1 < 0, -sum1, 0)
        sum2 = np.where(sum2 < 0, -sum2, 0)
        sum3 = np.where(sum3 < 0, -sum3, 0)
        sum4 = np.where(sum4 < 0, -sum4, 0)
        return sum1 + sum2 + sum3 + sum4

    objective_1 = ScalarObjective(name="minimize the mass of the brake", evaluator=f_1, maximize=[False])
    objective_2 = ScalarObjective(name="the minimum stopping time", evaluator=f_2, maximize=[False])
    objective_3 = ScalarObjective(name="the sum of the four constraint violations", evaluator=f_3, maximize=[False])

    objectives = [objective_1, objective_2, objective_3]

    # cons_1 = ScalarConstraint("c_1", 4, 3, g_1)
    # cons_2 = ScalarConstraint("c_2", 4, 3, g_2)
    # cons_3 = ScalarConstraint("c_3", 4, 3, g_3)
    # cons_4 = ScalarConstraint("c_4", 4, 3, g_4)
    #
    # constraints = [cons_1, cons_2, cons_3, cons_4]

    x_1 = Variable("the inner radius of the discs", 67.5, 55, 80)
    x_2 = Variable("the outer radius of the discs", 92.5, 75, 110)
    x_3 = Variable("the engaging force", 2000, 1000, 3000)
    x_4 = Variable("the number of friction surfaces", 15, 11, 20)

    variables = [x_1, x_2, x_3, x_4]

    problem = MOProblem(variables=variables, objectives=objectives)

    return problem



def RE34(var_iv: np.array = np.array([2, 2, 2, 2, 2])) -> MOProblem:
    """The crash safety design problem with 3 objectives.

    Liao, X., Li, Q., Yang, X., Zhang, W. & Li, W. (2007).
    Multiobjective optimization for crash safety design of vehicles
    using stepwise regression model. Structural and multidisciplinary
    optimization, 35(6), 561-569. https://doi.org/10.1007/s00158-007-0163-x

    Arguments:
        var_iv (np.array): Optional, initial variable values. Must be between
            1 and 3. Defaults are [2, 2, 2, 2, 2].

    Returns:
        MOProblem: a problem object.
    """

    if np.any(3 < var_iv) or np.any(var_iv < 1):
        raise ValueError("Initial variable values need to be between lower and upper bounds")

    # Mass
    def f_1(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            1640.2823
            + 2.3573285 * x[:, 0]
            + 2.3220035 * x[:, 1]
            + 4.5688768 * x[:, 2]
            + 7.7213633 * x[:, 3]
            + 4.4559504 * x[:, 4]
        )

    # Ain
    def f_2(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            6.5856
            + 1.15 * x[:, 0]
            - 1.0427 * x[:, 1]
            + 0.9738 * x[:, 2]
            + 0.8364 * x[:, 3]
            - 0.3695 * x[:, 0] * x[:, 3]
            + 0.0861 * x[:, 0] * x[:, 4]
            + 0.3628 * x[:, 1] * x[:, 3]
            - 0.1106 * x[:, 0] ** 2
            - 0.3437 * x[:, 2] ** 2
            + 0.1764 * x[:, 3] ** 2
        )

    # Intrusion
    def f_3(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            -0.0551
            + 0.0181 * x[:, 0]
            + 0.1024 * x[:, 1]
            + 0.0421 * x[:, 2]
            - 0.0073 * x[:, 0] * x[:, 1]
            + 0.024 * x[:, 1] * x[:, 2]
            - 0.0118 * x[:, 1] * x[:, 3]
            - 0.0204 * x[:, 2] * x[:, 3]
            - 0.008 * x[:, 2] * x[:, 4]
            - 0.0241 * x[:, 1] ** 2
            + 0.0109 * x[:, 3] ** 2
        )



    objective_1 = ScalarObjective(name="the weight", evaluator=f_1, maximize=[False])
    objective_2 = ScalarObjective(name="acceleration characteristics", evaluator=f_2, maximize=[False])
    objective_3 = ScalarObjective(name="toe-board instruction", evaluator=f_3, maximize=[False])

    objectives = [objective_1, objective_2, objective_3]

    x_1 = Variable("x_1", var_iv[0], 1.0, 3.0)
    x_2 = Variable("x_2", var_iv[1], 1.0, 3.0)
    x_3 = Variable("x_3", var_iv[2], 1.0, 3.0)
    x_4 = Variable("x_4", var_iv[3], 1.0, 3.0)
    x_5 = Variable("x_5", var_iv[4], 1.0, 3.0)

    variables = [x_1, x_2, x_3, x_4, x_5]

    ideal_point = np.array([1661.7078225, 6.14280000608, 0.0394])
    nadir_point = np.array([1695.2002035, 10.7454, 0.26399999965])

    problem = MOProblem(variables=variables, objectives=objectives)

    return problem



def RE36(var_iv: np.array = np.array([30, 30, 30, 30])) -> MOProblem:
    """The gear train design problem with 3 objectives.

    Arguments:
        var_iv (np.array): Optional, initial variable values. Must be integers between
            12 and 60. Defaults are [30, 30, 30, 30].

    Returns:
        MOProblem: a problem object.
    """

    if np.any(60 < var_iv) or np.any(var_iv < 12):
        raise ValueError("Initial variable values need to be between lower and upper bounds")


    def f_1(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        x = np.round(x)
        # First original objective function
        f1 = np.abs(6.931 - ((x[:,2] / x[:,0]) * (x[:,3] / x[:,1])))

        return f1

    def f_2(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        x = np.round(x)
        # Second original objective function (the maximum value among the four variables)
        f2 = np.max(x, axis=1)

        return f2

    def f_3(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        x = np.round(x)

        g1= 0.5 - (f_1(var_iv) / 6.931)

        f3 = np.where(g1 < 0, -g1, 0)

        return f3

    objective_1 = ScalarObjective(name="the error between the realized gear ration and the given required gear ration", evaluator=f_1, maximize=[False])
    objective_2 = ScalarObjective(name="the maximum size of the four gears", evaluator=f_2, maximize=[False])
    objective_3 = ScalarObjective(name="the constraint violation of g1", evaluator=f_3, maximize=[False])

    objectives = [objective_1, objective_2, objective_3]

    x_1 = Variable("x_1", var_iv[0], 12, 60)
    x_2 = Variable("x_2", var_iv[1], 12, 60)
    x_3 = Variable("x_3", var_iv[2], 12, 60)
    x_4 = Variable("x_4", var_iv[3], 12, 60)

    variables = [x_1, x_2, x_3, x_4]

    ideal_point = np.array([7.89473684213e-05, 12.0, 0.0])
    nadir_point = np.array([5.931, 56.0, 0.355720675227])

    problem = MOProblem(variables=variables, objectives=objectives)

    return problem



def RE41(var_iv: np.array = np.array([1, 1, 1, 1, 1, 1, 1])) -> MOProblem:
    """The car side impact design problem with 4 objectives.

    Arguments:
        var_iv (np.array): Optional, initial variable values. Must be integers between
            lb and up. Defaults are [1, 1, 1, 1, 1, 1, 1].

    Returns:
        MOProblem: a problem object.
    """

    n_original_constraints = 10

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


    def f_1(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        # First original objective function
        f1 = 1.98 + 4.9 * x[:,0] + 6.67 * x[:,1] + 6.98 * x[:,2] + 4.01 * x[:,3] + 1.78 * x[:,4] + 0.00001 * x[:,5] + 2.73 * x[:,6]

        return f1

    def f_2(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        # Second original objective function
        f2 = 4.72 - 0.5 * x[:,3] - 0.19 * x[:,1] * x[:,2]

        return f2

    def f_3(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)

        Vmbp = 10.58 - 0.674 * x[:,0] * x[:,1] - 0.67275 * x[:,1]
        Vfd = 16.45 - 0.489 * x[:,2] * x[:,6] - 0.843 * x[:,4] * x[:,5]

        f3 = 0.5 * (Vmbp + Vfd)
        return f3

    def f_4(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)

        Vmbp = 10.58 - 0.674 * x[:, 0] * x[:, 1] - 0.67275 * x[:, 1]
        Vfd = 16.45 - 0.489 * x[:, 2] * x[:, 6] - 0.843 * x[:, 4] * x[:, 5]

        g = np.zeros((len(x), n_original_constraints))
        g[:,0] = 1 - (1.16 - 0.3717 * x[:, 1] * x[:, 3] - 0.0092928 * x[:, 2])
        g[:,1] = 0.32 - (0.261 - 0.0159 * x[:, 0] * x[:, 1] - 0.06486 * x[:, 0] - 0.019 * x[:, 1] * x[:, 6] + 0.0144 * x[:, 2] * x[:, 4] + 0.0154464 * x[:, 5])
        g[:,2] = 0.32 - (
                    0.214 + 0.00817 * x[:, 4] - 0.045195 * x[:, 0] - 0.0135168 * x[:, 0] + 0.03099 * x[:, 1] * x[:, 5] - 0.018 * x[:, 1] * x[:, 6] + 0.007176 * x[:, 2] + 0.023232 * x[:, 2] - 0.00364 * x[:, 4] * x[:, 5] - 0.018 * x[:, 1] * x[:, 1])
        g[:,3] = 0.32 - (0.74 - 0.61 * x[:, 1] - 0.031296 * x[:, 2] - 0.031872 * x[:, 6] + 0.227 * x[:, 1] * x[:, 1])
        g[:,4] = 32 - (28.98 + 3.818 * x[:, 2] - 4.2 * x[:, 0] * x[:, 1] + 1.27296 * x[:, 5] - 2.68065 * x[:, 6])
        g[:,5] = 32 - (33.86 + 2.95 * x[:, 2] - 5.057 * x[:, 0] * x[:, 1] - 3.795 * x[:, 1] - 3.4431 * x[:, 6] + 1.45728)
        g[:,6] = 32 - (46.36 - 9.9 * x[:, 1] - 4.4505 * x[:, 0])
        g[:,7] = 4 - f_2(var_iv)
        g[:,8] = 9.9 - Vmbp
        g[:,9] = 15.7 - Vfd

        g = np.where(g < 0, -g, 0)
        f4 = g[:,0] + g[:,1] + g[:,2] + g[:,3] + g[:,4] + g[:,5] + g[:,6] + g[:,7] + g[:,8] + g[:,9]

        return f4

    objective_1 = ScalarObjective(name="the weight of the car", evaluator=f_1, maximize=[False])
    objective_2 = ScalarObjective(name="the pubic force experienced by a passenger", evaluator=f_2, maximize=[False])
    objective_3 = ScalarObjective(name="the average velocity of the V-pillar responsible for withstanding the impact load", evaluator=f_3, maximize=[False])
    objective_4 = ScalarObjective(name="the sum of the 10 constraint violations", evaluator=f_4, maximize=[False])

    objectives = [objective_1, objective_2, objective_3, objective_4]

    x_1 = Variable("x_1", var_iv[0], 0.5, 1.5)
    x_2 = Variable("x_2", var_iv[1], 0.45, 1.35)
    x_3 = Variable("x_3", var_iv[2], 0.5, 1.5)
    x_4 = Variable("x_4", var_iv[3], 0.5, 1.5)
    x_5 = Variable("x_5", var_iv[4], 0.875, 2.625)
    x_6 = Variable("x_6", var_iv[5], 0.4, 1.2)
    x_7 = Variable("x_7", var_iv[6], 0.4, 1.2)

    variables = [x_1, x_2, x_3, x_4, x_5, x_6, x_7]

    ideal_point = np.array([15.576004, 3.58525, 10.61064375, 0.0])
    nadir_point = np.array([39.2905121788, 4.42725, 13.09138125, 9.49401929991])

    problem = MOProblem(variables=variables, objectives=objectives)

    return problem

def re61(var_iv: np.array = np.array([0.02, 0.02, 0.02])) -> MOProblem:
    """ Water resource planning problem.
    Six objectives and three variables.

    RAY, T., TAI, K., & SEOW, K. C. (2001).
    MULTIOBJECTIVE DESIGN OPTIMIZATION BY AN EVOLUTIONARY ALGORITHM. Engineering Optimization, 33(4), 399–424.
    https://doi.org/10.1080/03052150108940926

    Arguments:
        var_iv (np.array): Optional, initial variable values.
            Defaults are [0.02, 0.02, 0.02]. x1 ∈ [0.01, 0.45], x2,x3 ∈ [0.01, 0.1]
    Returns:
        MOProblem: a problem object.
    """

    # Parameters
    n_original_constraints = 7
    # n_objectives = 6

    # Check the number of variables
    if (np.shape(np.atleast_2d(var_iv)[0]) != (3,)):
        raise RuntimeError("Number of variables must be three")

    # Lower bounds
    lb = np.array([0.01, 0.01, 0.01])

    # Upper bounds
    ub = np.array([0.45, 0.1, 0.1])

    # Check the variable bounds
    if np.any(lb > var_iv) or np.any(ub < var_iv):
        raise ValueError("Initial variable values need to be between lower and upper bounds")

    def f_1(x: np.ndarray) -> np.ndarray:
        f1 = 106780.37 * (x[:,1] + x[:,2]) + 61704.67
        return f1

    def f_2(x: np.ndarray) -> np.ndarray:
        f2 = 3000 * x[:,0]
        return f2

    def f_3(x: np.ndarray) -> np.ndarray:
        # a = np.power(0.06 * 2289, 0.65)
        # b = x[:,1] / np.power(0.06 * 2289, 0.65)
        f3 = 305700 * 2289 * x[:,1] / np.power(0.06 * 2289, 0.65)
        return f3

    def f_4(x: np.ndarray) -> np.ndarray:
        # a = np.exp(-39.75 * x[:,1] + 9.9 * x[:,2] + 2.74)
        f4 = 250 * 2289 * np.exp(-39.75 * x[:,1] + 9.9 * x[:,2] + 2.74)
        return f4

    def f_5(x: np.ndarray) -> np.ndarray:
        f5 = 25 * (1.39 / (x[:,0] * x[:,1]) + 4940 * x[:,2] - 80)
        return f5

    def f_6(x: np.ndarray) -> np.ndarray:
        # Constraint functions
        g = np.zeros((len(x), n_original_constraints))
        g[:,0] = 1 - 0.00139 / (x[:,0] * x[:,1]) - 4.94 * x[:,2] + 0.08
        g[:,1] = 1 - 0.000306 / (x[:,0] * x[:,1]) - 1.082 * x[:,2] + 0.0986
        g[:,2] = 50000 - 12.307 / (x[:,0] * x[:,1]) - 49408.24 * x[:,2] - 4051.02
        g[:,3] = 16000 - 2.098 / (x[:,0] * x[:,1]) - 8046.33 * x[:,2] + 696.71
        g[:,4] = 10000 - 2.138 / (x[:,0] * x[:,1]) - 7883.39 * x[:,2] + 705.04
        g[:,5] = 2000 - 0.417 * x[:,0] * x[:,1] - 1721.26 * x[:,2] + 136.54
        g[:,6] = 550 - 0.164 / (x[:,0] * x[:,1]) - 631.13 * x[:,2] + 54.48

        g = np.where(g < 0, -g, 0)
        f6 = g[:,0] + g[:,1] + g[:,2] + g[:,3] + g[:,4] + g[:,5] + g[:,6]
        return f6

    objective_1 = ScalarObjective(name="the drainage network cost", evaluator=f_1, maximize=[False])
    objective_2 = ScalarObjective(name="the storage facility cost", evaluator=f_2, maximize=[False])
    objective_3 = ScalarObjective(name="the treatment facility cost", evaluator=f_3, maximize=[False])
    objective_4 = ScalarObjective(name="the expected flflood damage cost", evaluator=f_4, maximize=[False])
    objective_5 = ScalarObjective(name="the expected economic loss due to flood", evaluator=f_5, maximize=[False])
    objective_6 = ScalarObjective(name="the sum of the seven constraint violations", evaluator=f_6, maximize=[False])

    objectives = [objective_1, objective_2, objective_3, objective_4, objective_5, objective_6]

    # The four variables determine the length of four bars
    x_1 = Variable("x_1", 0.02, lb[0], ub[0])
    x_2 = Variable("x_2", 0.02, lb[1], ub[1])
    x_3 = Variable("x_3", 0.02, lb[2], ub[2])

    variables = [x_1, x_2, x_3]

    nadir_point = np.array([80896.9128355, 1350.0, 2853468.96494, 7076861.67064, 87748.6339553, 2.50994535821])
    ideal_point = np.array([63840.2774, 30.0, 285346.896494, 183749.967061, 7.22222222222, 0.0])

    problem = MOProblem(variables=variables, objectives=objectives)

    return problem
    return problem
