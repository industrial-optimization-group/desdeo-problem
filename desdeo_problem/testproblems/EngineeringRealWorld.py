"""
A real-world multi-objective problem suite (the RE benchmark set)

Tanabe, R. & Ishibuchi, H. (2020). An easy-to-use real-world multi-objective 
optimization problem suite. Applied soft computing, 89, 106078. 
https://doi.org/10.1016/j.asoc.2020.106078 

https://github.com/ryojitanabe/reproblems/blob/master/reproblem_python_ver/reproblem.py

"""

from desdeo_problem.problem.Variable import Variable
from desdeo_problem.problem.Objective import ScalarObjective
from desdeo_problem.problem.Problem import MOProblem, ProblemBase
from desdeo_problem import ScalarConstraint, problem

import numpy as np

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
    def g_1(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        x = feas_val(x)
        return -(x[:, 0] * x[:, 2] - 7.735 * (x[:, 0]**2 / x[:, 1]) - 180 )

    def g_2(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return -(4 - x[:, 2] / x[:, 1])

    # Objective functions
    def f_1(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        x = feas_val(x)
        return (29.4 * x[:, 0]) + (0.6 * x[:, 1] * x[:,2])

    def f_2(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        sum1 = g_1(x)
        sum2 = g_2(x)
        sum1 = np.where(sum1 > 0, sum1, 0)
        sum2 = np.where(sum2 > 0, sum2, 0)
        return sum1 + sum2

    objective_1 = ScalarObjective(name="minimize the total cost of concrete and reinforcing steel of the beam",
        evaluator=f_1, maximize=[False])

    objective_2 = ScalarObjective(name="the sum of the four constraint violations", evaluator=f_2, maximize=[False])

    objectives = [objective_1, objective_2]

    cons_1 = ScalarConstraint("c_1", 3, 2, g_1)
    cons_2 = ScalarConstraint("c_2", 3, 2, g_2)

    constraints = [cons_1, cons_2]

    x_1 = Variable("the area of the reinforcement", 7.2, 0.2, 15)
    x_2 = Variable("the width of the beam", 10, 0, 20)
    x_3 = Variable("the depth of the beam", 20, 0, 40)

    variables = [x_1, x_2, x_3]

    problem = MOProblem(variables=variables, objectives=objectives, constraints=constraints)

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
    def g_1(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        x = x.astype(float)
        x[:, 0] = 0.0625 * (np.round(x[:,0]))
        return -(x[:, 0] - (0.0193 * x[:, 2]))

    def g_2(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        x = x.astype(float)
        x[:, 1] = 0.0625 * (np.round(x[:,1]))
        return -(x[:, 1] - (0.00954 * x[:, 2]))

    def g_3(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return -( (np.pi * x[:, 2]**2 * x[:, 3]) + ((4/3) * np.pi * x[:, 2]**3) - 1296000 )

    # Objective functions
    def f_1(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        x = x.astype(float)
        x[:, 0] = 0.0625 * (np.round(x[:,0]))
        x[:, 1] = 0.0625 * (np.round(x[:,1]))
        return (
            (0.6224 * x[:, 0] * x[:, 2] * x[:, 3]) + (1.7781 * x[:, 1] * x[:, 2]**2) +
            (3.1661 * x[:, 0]**2 * x[:, 3]) + (19.84 * x[:, 0]**2 * x[:, 2])
        )

    def f_2(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        sum1 = g_1(x)
        sum2 = g_2(x)
        sum3 = g_3(x)
        sum1 = np.where(sum1 > 0, sum1, 0)
        sum2 = np.where(sum2 > 0, sum2, 0)
        sum3 = np.where(sum3 > 0, sum3, 0)
        return sum1 + sum2 + sum3
    
    objective_1 = ScalarObjective(name="minimize to total cost of a clyndrical pressure vessel", evaluator=f_1, maximize=[False])
    objective_2 = ScalarObjective(name="the sum of the four constraint violations", evaluator=f_2, maximize=[False])

    objectives = [objective_1, objective_2]

    cons_1 = ScalarConstraint("c_1", 4, 2, g_1)
    cons_2 = ScalarConstraint("c_2", 4, 2, g_2)
    cons_3 = ScalarConstraint("c_3", 4, 2, g_3)

    constraints = [cons_1, cons_2, cons_3]

    x_1 = Variable("the thicknesses of the shell", 50, 1, 100)
    x_2 = Variable("the the head of pressure vessel", 50, 1, 100)
    x_3 = Variable("the inner radius", 100, 10, 200)
    x_4 = Variable("the length of the cylindrical section", 120, 10, 240)

    variables = [x_1, x_2, x_3, x_4]

    problem = MOProblem(variables=variables, objectives=objectives, constraints=constraints)

    return problem

def re24(var_iv : np.array = np.array([2, 25])) -> MOProblem:
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
    def g_1(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return -(1.0 - ((4500 / (x[:, 0] * x[:, 1])) / 700))

    def g_2(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return -(1.0 - ((1800 / x[:, 1]) / 450))

    def g_3(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return -(1.0 - (((56.2 * 10000) / (700000 * x[:, 0] * x[:, 1]**2)) / 1.5) )

    def g_4(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return -(1.0 - ((4500 / (x[:,0] * x[:, 1])) / ((700000 * x[:, 0]**2) / 100)) )

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
        sum1 = np.where(sum1 > 0, sum1, 0)
        sum2 = np.where(sum2 > 0, sum2, 0)
        sum3 = np.where(sum3 > 0, sum3, 0)
        sum4 = np.where(sum4 > 0, sum4, 0)
        return sum1 + sum2 + sum3 + sum4
    
    objective_1 = ScalarObjective(name="to minimize the weight of the hatch cover", evaluator=f_1, maximize=[False])
    objective_2 = ScalarObjective(name="the sum of the four constraint violations", evaluator=f_2, maximize=[False])

    objectives = [objective_1, objective_2]

    cons_1 = ScalarConstraint("c_1", 2, 2, g_1)
    cons_2 = ScalarConstraint("c_2", 2, 2, g_2)
    cons_3 = ScalarConstraint("c_3", 2, 2, g_3)
    cons_4 = ScalarConstraint("c_4", 2, 2, g_4)

    constraints = [cons_1, cons_2, cons_3, cons_4]

    x_1 = Variable("the flange thickness", 2, 0.5, 4)
    x_2 = Variable("the beam height", 25, 4, 50)

    variables = [x_1, x_2]

    problem = MOProblem(variables=variables, objectives=objectives, constraints=constraints)

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
    def g_1(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        x = feas_val(x)
        return -(
            -((8 * (((4.0 * (x[:, 1] / x[:, 2]) - 1) / 
            (4.0 * (x[:, 1] / x[:, 2]) - 4)) + 
            ((0.615 * x[:, 2]) / x[:, 1])) * 1000 * x[:, 1]) 
            / (np.pi * x[:, 2]**3 )) + 189000
        )

    def g_2(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        x = feas_val(x)
        return -(
            - ( 1000 / ( ( 11.5 * 10**6 * x[:, 2]**4) / ( 8 * np.round(x[:, 0]) * x[:, 1]**3 ) ) ) + 1.05 * ( np.round( x[:, 0]) + 2) * x[:, 2] + 14
        )

    def g_3(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        x = feas_val(x)
        return -(
            -3 + (x[:, 1] / x[:, 2])
        )

    def g_4(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return -(
            - (300 / ((11.5 * 10**6 * x[:, 2]**4) / (8 * np.round(x[:, 0]) * x[:, 1]**3))) + 6
        )

    def g_5(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        x = feas_val(x)
        return -(
            -(300 / ((11.5 * 10**6 * x[:, 2]**4) / (8 * np.round(x[:, 0]) * x[:, 1]**3))) - ((1000 - 300) / ((11.5 * 10**6 * x[:, 2]**4) / (8 * np.round(x[:, 0]) * x[:, 1]**3))) - (1.05 * (np.round(x[:, 0]) + 2) * x[:, 2]) + ((1000 / ((11.5 * 10**6 * x[:, 2]**4) / (8 * np.round(x[:, 0]) * x[:, 1]**3))) + (1.05 * ( np.round(x[:, 0]) + 2) * x[:, 2]))
        )

    def g_6(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        x = feas_val(x)
        return -(
            -1.25 + ((1000 - 300) / ((11.5 * 10**6 * x[:, 2]**4) / (8 * np.round(x[:, 0]) * x[:, 1]**3)))
        )

    # Objective functions
    def f_1(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        x = feas_val(x)
        return ((np.pi * np.pi * x[:,1] * x[:,2]**2 * ((np.round(x[:,0])) + 2)) / 4.0)

    def f_2(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        sum1 = g_1(x)
        sum2 = g_2(x)
        sum3 = g_3(x)
        sum4 = g_4(x)
        sum5 = g_5(x)
        sum6 = g_6(x)
        sum1 = np.where(sum1 > 0, sum1, 0)
        sum2 = np.where(sum2 > 0, sum2, 0)
        sum3 = np.where(sum3 > 0, sum3, 0)
        sum4 = np.where(sum4 > 0, sum4, 0)
        sum5 = np.where(sum5 > 0, sum5, 0)
        sum6 = np.where(sum6 > 0, sum6, 0)
        return sum1 + sum2 + sum3 + sum4 + sum5 + sum6

    objective_1 = ScalarObjective(name="minimize the volume of spring steel wire which is used to manufacture the spring",
        evaluator=f_1, maximize=[False])

    objective_2 = ScalarObjective(name="the sum of the four constraint violations", evaluator=f_2, maximize=[False])

    objectives = [objective_1, objective_2]

    cons_1 = ScalarConstraint("c_1", 3, 2, g_1)
    cons_2 = ScalarConstraint("c_2", 3, 2, g_2)
    cons_3 = ScalarConstraint("c_3", 3, 2, g_3)
    cons_4 = ScalarConstraint("c_4", 3, 2, g_4)
    cons_5 = ScalarConstraint("c_5", 3, 2, g_5)
    cons_6 = ScalarConstraint("c_6", 3, 2, g_6)


    constraints = [cons_1, cons_2, cons_3, cons_4, cons_5, cons_6]

    x_1 = Variable("the number of spring coils", 35, 1, 70)
    x_2 = Variable("the outside diameter of the spring", 15, 0.6, 30)
    x_3 = Variable("the spring wire diameter", 0.207, 0.009, 0.5)

    variables = [x_1, x_2, x_3]

    problem = MOProblem(variables=variables, objectives=objectives, constraints=constraints)

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
            x[:, 0] * np.sqrt(16 + x[:, 2]**2 ) 
            + x[:, 1] * np.sqrt(1 + x[:, 2]**2 )
        )

    def f_2(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            (20 * np.sqrt(16 + x[:, 2]**2 ) 
            / ( x[:, 2] * x[:, 0] ))
        )

    # Constrain functions
    def g_1(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return -(
            0.1 - f_1(x)
        )

    def g_2(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return -(
            10**5 - f_2(x)
            )

    def g_3(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return -(
            10**5 
            - ((80 * np.sqrt(1 + x[:, 2]**2 ) 
            / (x[:, 2] * x[:, 1])))
        )

    # Third objective
    def f_3(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        sum1 = g_1(x)
        sum2 = g_2(x)
        sum3 = g_3(x)
        sum1 = np.where(sum1 > 0, sum1, 0)
        sum2 = np.where(sum2 > 0, sum2, 0)
        sum3 = np.where(sum3 > 0, sum3, 0)
        return sum1 + sum2 + sum3

    objective_1 = ScalarObjective(name="minimize the structural weight", evaluator=f_1, maximize=[False])
    objective_2 = ScalarObjective(name="minimize the resultant displacement of join", evaluator=f_2, maximize=[False])
    objective_3 = ScalarObjective(name="the sum of the four constraint violations", evaluator=f_3, maximize=[False])

    objectives = [objective_1, objective_2, objective_3]

    cons_1 = ScalarConstraint("c_1", 3, 3, g_1)
    cons_2 = ScalarConstraint("c_2", 3, 3, g_2)
    cons_3 = ScalarConstraint("c_3", 3, 3, g_3)

    constraints = [cons_1, cons_2, cons_3]

    x_1 = Variable("the length of the bar", 50.0, 0.00001, 100)
    x_2 = Variable("the length of the bar", 50.0, 0.00001, 100)
    x_3 = Variable("the spring wire diameter", 2.0, 1.0, 3.0)

    variables = [x_1, x_2, x_3]

    problem = MOProblem(variables=variables, objectives=objectives, constraints=constraints)

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
                (6000 / (np.sqrt(2) * x[:, 0] * x[:, 1]))**2 +
                ((2 * (6000 / (np.sqrt(2) * x[:, 0] * x[:, 1])) * 
                ((6000 * (14 + (x[:, 1] / 2))) * 
                (np.sqrt( ( (x[:,1]**2) / 4.0) + 
                ((x[:, 0] + x[:, 2]) / 2)**2 ))) / 
                (2 * (np.sqrt(2) * x[:, 0] * x[:, 1] * 
                ((x[:,1]**2)/12 + ((x[:,0] + x[:,2]) / 2)**2) )) * x[:, 1]) / 
                (2 * (np.sqrt( ( (x[:,1]**2) / 4.0) + ((x[:, 0] + x[:, 2]) / 2)**2 )))) +
                (((6000 * (14 + (x[:, 1] / 2))) * 
                (np.sqrt( ( (x[:,1]**2) / 4.0) + 
                ((x[:, 0] + x[:, 2]) / 2)**2 ))) / 
                (2 * (np.sqrt(2) * x[:, 0] * x[:, 1] * 
                ((x[:,1]**2)/12 + ((x[:,0] + x[:,2]) / 2)**2) )))**2
            )
        )

    def sigma(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            (6 * 6000 * 14) / (x[:, 3] * x[:, 2]**2)
        )

    def p_c(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            ((4.013 * 30 * 10**6 * np.sqrt((x[:, 2]**2 * 
            x[:, 3]**6) / 36)) / (14**2) ) * 
            (1 - (x[:, 2] / (2 * 14)) * np.sqrt((30 * 10**6) / 
            (4 * 12 * 10**6)))
        )

    # Constrain functions
    def g_1(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return -(
            13600 - tau(x)
        )

    def g_2(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return -(
            30000 - sigma(x)
        )

    def g_3(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return -(
            x[:, 3] - x[:, 0]
        )
            

    def g_4(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return -(
            p_c(x) - 6000
        )

    # Objective functions
    def f_1(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            1.10471 * x[:, 0]**2 * x[:, 1]
            + 0.04811 * x[:, 2] * x[:, 3] *
            (14 + x[:, 1])
        )

    def f_2(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            (4 * 6000 * 14**3) /
            (30 * 10**6 * x[:, 3] * x[:, 2]**3)
        )

    def f_3(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        sum1 = g_1(x)
        sum2 = g_2(x)
        sum3 = g_3(x)
        sum4 = g_4(x)
        sum1 = np.where(sum1 > 0, sum1, 0)
        sum2 = np.where(sum2 > 0, sum2, 0)
        sum3 = np.where(sum3 > 0, sum3, 0)
        sum4 = np.where(sum4 > 0, sum4, 0)
        return sum1 + sum2 + sum3 + sum4

    objective_1 = ScalarObjective(name="minimize cost of a welded beam", evaluator=f_1, maximize=[False])
    objective_2 = ScalarObjective(name="minimize end deflection of a welded beam", evaluator=f_2, maximize=[False])
    objective_3 = ScalarObjective(name="the sum of the four constraint violations", evaluator=f_3, maximize=[False])

    objectives = [objective_1, objective_2, objective_3]

    cons_1 = ScalarConstraint("c_1", 4, 3, g_1)
    cons_2 = ScalarConstraint("c_2", 4, 3, g_2)
    cons_3 = ScalarConstraint("c_3", 4, 3, g_3)
    cons_4 = ScalarConstraint("c_4", 4, 3, g_4)

    constraints = [cons_1, cons_2, cons_3, cons_4]

    # Variables adjust the size of the beam
    x_1 = Variable("x_1", 2.5, 0.125, 5)
    x_2 = Variable("x_2", 5.0, 0.1, 10.0)
    x_3 = Variable("x_3", 5.0, 0.1, 10.0)
    x_4 = Variable("x_4", 2.5, 0.125, 5)

    variables = [x_1, x_2, x_3, x_4]

    problem = MOProblem(variables=variables, objectives=objectives, constraints=constraints)

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
    def g_1(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return -(
            (x[:, 1] - x[:, 0]) - 20
        )

    def g_2(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return -(
            0.4 - (x[:, 2] / (3.14 * (x[:, 1]**2 - x[:, 0]**2)))
        )

    def g_3(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return -(
            1 - ((2.22 * 10**-3 * x[:, 2] * (x[:, 1]**3 - x[:, 0]**3)) / 
            (x[:, 1]**2 - x[:, 0]**2)**2)
        )
            

    def g_4(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return -(
            ((2.66 * 10**-2 * x[:, 2] * x[:, 3] * (x[:, 1]**3 - x[:, 0]**3)) / 
            (x[:, 1]**2 - x[:, 0]**2)) - 900
        )

    # Objective functions
    def f_1(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            4.9 * 10**-5 * (x[:, 1]**2 - x[:, 0]**2) * (x[:, 3] -1)
        )

    def f_2(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            9.82 * 10**6 * ((x[:, 1]**2 - x[:, 0]**2) / 
            (x[:, 2] * x[:, 3] * (x[:, 1]**3 - x[:, 0]**3)))
        )

    def f_3(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        sum1 = g_1(x)
        sum2 = g_2(x)
        sum3 = g_3(x)
        sum4 = g_4(x)
        sum1 = np.where(sum1 > 0, sum1, 0)
        sum2 = np.where(sum2 > 0, sum2, 0)
        sum3 = np.where(sum3 > 0, sum3, 0)
        sum4 = np.where(sum4 > 0, sum4, 0)
        return sum1 + sum2 + sum3 + sum4

    objective_1 = ScalarObjective(name="minimize the mass of the brake", evaluator=f_1, maximize=[False])
    objective_2 = ScalarObjective(name="the minimum stopping time", evaluator=f_2, maximize=[False])
    objective_3 = ScalarObjective(name="the sum of the four constraint violations", evaluator=f_3, maximize=[False])

    objectives = [objective_1, objective_2, objective_3]

    cons_1 = ScalarConstraint("c_1", 4, 3, g_1)
    cons_2 = ScalarConstraint("c_2", 4, 3, g_2)
    cons_3 = ScalarConstraint("c_3", 4, 3, g_3)
    cons_4 = ScalarConstraint("c_4", 4, 3, g_4)

    constraints = [cons_1, cons_2, cons_3, cons_4]

    x_1 = Variable("the inner radius of the discs", 67.5, 55, 80)
    x_2 = Variable("the outer radius of the discs", 92.5, 75, 110)
    x_3 = Variable("the engaging force", 2000, 1000, 3000)
    x_4 = Variable("the number of friction surfaces", 15, 11, 20)

    variables = [x_1, x_2, x_3, x_4]

    problem = MOProblem(variables=variables, objectives=objectives, constraints=constraints)

    return problem


def re34():
    re34_json = {
    "constants":[],
    "variables":[
        {
            "longname":"Decision variable 1",
            "shortname":"x_1",
            "lowerbound":1,
            "upperbound":3,
            "type":"RealNumber",
            "initialvalue":None
        },
        {
            "longname":"Decision variable 2",
            "shortname":"x_2",
            "lowerbound":1,
            "upperbound":3,
            "type":"RealNumber",
            "initialvalue":None
        },
        {
            "shortname":"x_3",
            "lowerbound":1,
            "upperbound":3,
            "type":"RealNumber",
            "initialvalue":None
        },
        {
            "shortname":"x_4",
            "lowerbound":1,
            "upperbound":3,
            "type":"RealNumber",
            "initialvalue":None
        },
        {
            "shortname":"x_5",
            "lowerbound":1,
            "upperbound":3,
            "type":"RealNumber",
            "initialvalue":None
        }
    ],
    "extra_func":[
    ],
    "objectives":[  
        {
            "longname":"minimize structural volume",
            "shortname":"f1",
            "func":
                [
                    "Add",
                    ["Multiply", 2.3573285, "x_1"],
                    ["Multiply", 2.3220035, "x_2"],
                    ["Multiply", 4.5688768, "x_3"],
                    ["Multiply", 7.7213633, "x_4"],
                    ["Multiply", 4.4559504, "x_5"],
                    1640.2823
                ],
            "max": False,
            "lowerbound":None,
            "upperbound":None
        },
        {
            "longname":"minimize the joint displacement",
            "shortname":"f2",
            "func":[
                    "Add", 
                    ["Multiply", -0.1106, ["Square", "x_1"]], 
                    ["Multiply", -0.3437, ["Square", "x_3"]], 
                    ["Multiply", 0.1764, ["Square", "x_4"]], 
                    ["Multiply", 1.15, "x_1"], 
                    ["Multiply", -0.3695, "x_1", "x_4"], 
                    ["Multiply", 0.0861, "x_1", "x_5"], 
                    ["Multiply", -1.0427, "x_2"], 
                    ["Multiply", 0.3628, "x_2", "x_4"], 
                    ["Multiply", 0.9738, "x_3"], 
                    ["Multiply", 0.8364, "x_4"], 
                    6.5856
                    ],
            "max": False,
            "lowerbound":None,
            "upperbound":None
        },
        {
            "longname":"minimize the joint displacement",
            "shortname":"f3",
            "func":[
                "Add", 
                ["Multiply", -0.0241, ["Square", "x_2"]], 
                ["Multiply", 0.0109, ["Square", "x_4"]], 
                ["Multiply", 0.0181, "x_1"], 
                ["Multiply", -0.0073, "x_1", "x_2"], 
                ["Multiply", 0.1024, "x_2"], 
                ["Multiply", 0.024, "x_2", "x_3"], 
                ["Multiply", -0.0118, "x_2", "x_4"], 
                ["Multiply", 0.0421, "x_3"], 
                ["Multiply", -0.0204, "x_3", "x_4"], 
                ["Multiply", -0.008, "x_3", "x_5"], 
                -0.0551
                ],
            "max": False,
            "lowerbound":None,
            "upperbound":None
        }
    ],
    "constraints":[],
    "__problemName":"RE"
    }
    p = MOProblem(json=re34_json)
    return p

def re37():
    re37_json = {
    "constants":[],
    "variables":[
        {
            "longname":"Decision variable 1",
            "shortname":"x_1",
            "lowerbound":0,
            "upperbound":1,
            "type":"RealNumber",
            "initialvalue":None
        },
        {
            "longname":"Decision variable 2",
            "shortname":"x_2",
            "lowerbound":0,
            "upperbound":1,
            "type":"RealNumber",
            "initialvalue":None
        },
        {
            "shortname":"x_3",
            "lowerbound":0,
            "upperbound":1,
            "type":"RealNumber",
            "initialvalue":None
        },
        {
            "shortname":"x_4",
            "lowerbound":0,
            "upperbound":1,
            "type":"RealNumber",
            "initialvalue":None
        }
    ],
    "extra_func":[
    ],
    "objectives":[  
        {
            "longname":"minimize structural volume",
            "shortname":"f1",
            "func":
                [
                "Add", 
                ["Multiply", -0.167, ["Square", "x_1"]], 
                ["Multiply", 0.0796, ["Square", "x_2"]], 
                ["Multiply", 0.0877, ["Square", "x_3"]], 
                ["Multiply", 0.0184, ["Square", "x_4"]], 
                ["Multiply", 0.477, "x_1"], 
                ["Multiply", -0.0129, "x_1", "x_2"], 
                ["Multiply", -0.0634, "x_1", "x_3"], 
                ["Multiply", -0.0521, "x_1", "x_4"], 
                ["Multiply", -0.687, "x_2"], 
                ["Multiply", -0.0257, "x_2", "x_3"], 
                ["Multiply", 0.00156, "x_2", "x_4"], 
                ["Multiply", -0.08, "x_3"], 
                ["Multiply", 0.00198, "x_3", "x_4"], 
                ["Multiply", -0.065, "x_4"], 
                0.692
            ],
            "max": False,
            "lowerbound":None,
            "upperbound":None
        },
        {
            "longname":"minimize the joint displacement",
            "shortname":"f2",
            "func":[
                    "Add", 
                    ["Multiply", 0.175, ["Square", "x_1"]], 
                    ["Multiply", -0.0701, ["Square", "x_2"]], 
                    ["Multiply", 0.015, ["Square", "x_3"]], 
                    ["Multiply", 0.0192, ["Square", "x_4"]], 
                    ["Multiply", -0.322, "x_1"], 
                    ["Multiply", 0.0185, "x_1", "x_2"], 
                    ["Multiply", -0.251, "x_1", "x_3"], 
                    ["Multiply", 0.0134, "x_1", "x_4"], 
                    ["Multiply", 0.396, "x_2"], 
                    ["Multiply", 0.179, "x_2", "x_3"], 
                    ["Multiply", 0.0296, "x_2", "x_4"], 
                    ["Multiply", 0.424, "x_3"], 
                    ["Multiply", 0.0752, "x_3", "x_4"], 
                    ["Multiply", 0.0226, "x_4"], 
                    0.153
                    ],
            "max": False,
            "lowerbound":None,
            "upperbound":None
        },
        {
            "longname":"minimize the joint displacement",
            "shortname":"f3",
            "func":[
                    "Add", 
                    ["Multiply", -0.135, ["Square", "x_1"]], 
                    ["Multiply", 0.0998, ["Square", "x_2"]], 
                    ["Multiply", -0.226, ["Square", "x_3"]], 
                    ["Multiply", -0.423, ["Square", "x_4"]], 
                    ["Multiply", -0.205, "x_1"], 
                    ["Multiply", -0.342, "x_1", ["Square", "x_2"]], 
                    ["Multiply", -0.184, "x_1", ["Square", "x_4"]], 
                    ["Multiply", 0.0141, "x_1", "x_2"], 
                    ["Multiply", -0.281, "x_1", "x_2", "x_3"], 
                    ["Multiply", 0.208, "x_1", "x_3"], 
                    ["Multiply", 0.353, "x_1", "x_4"], 
                    ["Multiply", 0.0307, "x_2"], 
                    ["Multiply", 0.202, "x_2", ["Square", "x_1"]], 
                    ["Multiply", 0.281, "x_2", ["Square", "x_3"]], 
                    ["Multiply", -0.0301, "x_2", "x_3"], 
                    ["Multiply", 0.108, "x_3"], 
                    ["Multiply", -0.281, "x_3", ["Square", "x_1"]], 
                    ["Multiply", -0.245, "x_3", ["Square", "x_2"]], 
                    ["Multiply", -0.0497, "x_3", "x_4"], 
                    ["Multiply", 1.019, "x_4"], 
                    0.37
                    ],
            "max": False,
            "lowerbound":None,
            "upperbound":None
        }
    ],
    "constraints":[],
    "__problemName":"RE"
    }
    p = MOProblem(json=re37_json)
    return p

def re41():
    re41_json = {
    "constants":[],
    "variables":[
        {
            "longname":"Decision variable 1",
            "shortname":"x_1",
            "lowerbound":0.5,
            "upperbound":1.5,
            "type":"RealNumber",
            "initialvalue":None
        },
        {
            "longname":"Decision variable 2",
            "shortname":"x_2",
            "lowerbound":0.45,
            "upperbound":1.35,
            "type":"RealNumber",
            "initialvalue":None
        },
        {
            "shortname":"x_3",
            "lowerbound":0.5,
            "upperbound":1.5,
            "type":"RealNumber",
            "initialvalue":None
        },
        {
            "shortname":"x_4",
            "lowerbound":0.5,
            "upperbound":1.5,
            "type":"RealNumber",
            "initialvalue":None
        },
                {
            "shortname":"x_5",
            "lowerbound":0.875,
            "upperbound":2.625,
            "type":"RealNumber",
            "initialvalue":None
        },
        {
            "shortname":"x_6",
            "lowerbound":0.4,
            "upperbound":1.2,
            "type":"RealNumber",
            "initialvalue":None
        },
                {
            "shortname":"x_7",
            "lowerbound":0.4,
            "upperbound":1.2,
            "type":"RealNumber",
            "initialvalue":None
        }
    ],
    "extra_func":[
        {
            "longname":"minimize structural volume",
            "shortname":"g_1",
            "func":["Negate",[
                "Add", 
                ["Multiply", 0.3717, "x_2", "x_4"], 
                ["Multiply", 0.0092928, "x_3"], 
                -1.16, 
                1
                ]],
        },
        {
            "longname":"minimize structural volume",
            "shortname":"g_2",
            "func":["Negate",[
                "Add", 
                ["Multiply", 0.06486, "x_1"], 
                ["Multiply", 0.0159, "x_1", "x_2"], 
                ["Multiply", 0.019, "x_2", "x_7"], 
                ["Multiply", -0.0144, "x_3", "x_5"], 
                ["Multiply", -0.0154464, "x_6"], 
                -0.261, 
                0.32
                ]],
        },
        {
            "longname":"minimize structural volume",
            "shortname":"g_3",
            "func":["Negate",[
                "Add", 
                ["Multiply", 0.018, ["Square", "x_2"]], 
                ["Multiply", 0.045195, "x_1"], 
                ["Multiply", 0.0135168, "x_1"], 
                ["Multiply", -0.03099, "x_2", "x_6"], 
                ["Multiply", 0.018, "x_2", "x_7"], 
                ["Multiply", -0.007176, "x_3"], 
                ["Multiply", -0.023232, "x_3"], 
                ["Multiply", -0.00817, "x_5"], 
                ["Multiply", 0.00364, "x_5", "x_6"], 
                -0.214, 
                0.32
                ]],
        },
        {
            "longname":"minimize structural volume",
            "shortname":"g_4",
            "func":["Negate",[
                "Add", 
                ["Multiply", -0.227, ["Square", "x_2"]], 
                ["Multiply", 0.61, "x_2"], 
                ["Multiply", 0.031296, "x_3"], 
                ["Multiply", 0.031872, "x_7"], 
                -0.74, 
                0.32
                ]],
        },
        {
            "longname":"minimize structural volume",
            "shortname":"g_5",
            "func":["Negate",[
  "Add", 
  ["Multiply", 4.2, "x_1", "x_2"], 
  ["Multiply", -3.818, "x_3"], 
  ["Multiply", -1.27296, "x_6"], 
  ["Multiply", 2.68065, "x_7"], 
  -28.98, 
  32
]],
        },
        {
            "longname":"minimize structural volume",
            "shortname":"g_6",
            "func":["Negate",[
  "Add", 
  ["Multiply", 5.057, "x_1", "x_2"], 
  ["Multiply", 3.795, "x_2"], 
  ["Multiply", -2.95, "x_3"], 
  ["Multiply", 3.4431, "x_7"], 
  -33.86, 
  -1.45728, 
  32
]],
        },
        {
            "longname":"minimize structural volume",
            "shortname":"g_7",
            "func":["Negate",[
  "Add", 
  ["Multiply", 4.4505, "x_1"], 
  ["Multiply", 9.9, "x_2"], 
  -46.36, 
  32
]],
        },
        {
            "longname":"minimize structural volume",
            "shortname":"g_8",
            "func":["Negate",[
                "Add", 
                ["Multiply", 0.19, "x_2", "x_3"], 
                ["Multiply", 0.5, "x_4"], 
                -4.72, 
                4
                ]],
        },
        {
            "longname":"minimize structural volume",
            "shortname":"g_9",
            "func":["Negate",[
    "Add", 
    ["Multiply", 0.674, "x_1", "x_2"], 
    ["Multiply", 0.67275, "x_2"], 
    -10.58, 
    9.9
    ]],
        },
        {
            "longname":"minimize structural volume",
            "shortname":"g_10",
            "func":["Negate",[
                "Add", 
                ["Multiply", 0.489, "x_3", "x_7"], 
                ["Multiply", 0.843, "x_5", "x_6"], 
                -16.45, 
                15.7
                ]],
        }
    ],
    "objectives":[  
        {
            "longname":"minimize structural volume",
            "shortname":"f1",
            "func":
                [
                    "Add", 
                    ["Multiply", 4.9, "x_1"], 
                    ["Multiply", 6.67, "x_2"], 
                    ["Multiply", 6.98, "x_3"], 
                    ["Multiply", 4.01, "x_4"], 
                    ["Multiply", 1.78, "x_5"], 
                    ["Multiply", 0.00001, "x_6"], 
                    ["Multiply", 2.73, "x_7"], 
                    1.98

            ],
            "max": False,
            "lowerbound":None,
            "upperbound":None
        },
        {
            "longname":"minimize the joint displacement",
            "shortname":"f2",
            "func":[
                    "Add", 
                    ["Multiply", -0.19, "x_2", "x_3"], 
                    ["Multiply", -0.5, "x_4"], 
                    4.72
                    ],
            "max": False,
            "lowerbound":None,
            "upperbound":None
        },
        {
            "longname":"minimize the joint displacement",
            "shortname":"f3",
            "func":[
                    "Multiply", 
                    0.5, 
                    [
                        "Add", 
                        ["Multiply", -0.674, "x_1", "x_2"], 
                            ["Multiply", -0.67275, "x_2"], 
                        ["Multiply", -0.489, "x_3", "x_7"], 
                        ["Multiply", -0.843, "x_5", "x_6"], 
                        10.58, 
                        16.45
                    ]
                    ],
            "max": False,
            "lowerbound":None,
            "upperbound":None
        },
                {
            "longname":"minimize the joint displacement",
            "shortname":"f_3",
            "func":["Sum", ["Max","g_i",0], ["Triple", ["Hold", "i"], 1, 10]],
            "max": False,   
            "lowerbound":None,
            "upperbound":None
        }
    ],
    "constraints":[],
    "__problemName":"RE"
    }
    p = MOProblem(json=re41_json)
    return p

def re42():
    re42_json = {
    "constants":[
        {
            "shortname":"g",
            "value":9.8065
        },
        {
            "shortname":"round_trip_miles",
            "value":5000.0
        },
        {
            "shortname":"handling_rate",
            "value":8000.0
        },
        {
            "shortname":"fuel_price",
            "value":100.0
        }
    ],
    "variables":[
        {
            "longname":"Decision variable 1",
            "shortname":"x_1",
            "lowerbound":150,
            "upperbound":274.32,
            "type":"RealNumber",
            "initialvalue":None
        },
        {
            "longname":"Decision variable 2",
            "shortname":"x_2",
            "lowerbound":20,
            "upperbound":32.31,
            "type":"RealNumber",
            "initialvalue":None
        },
        {
            "shortname":"x_3",
            "lowerbound":13,
            "upperbound":25.0,
            "type":"RealNumber",
            "initialvalue":None
        },
        {
            "shortname":"x_4",
            "lowerbound":10.0,
            "upperbound":11.71,
            "type":"RealNumber",
            "initialvalue":None
        },
                {
            "shortname":"x_5",
            "lowerbound":14.0,
            "upperbound":18.0,
            "type":"RealNumber",
            "initialvalue":None
        },
        {
            "shortname":"x_6",
            "lowerbound":0.63,
            "upperbound":0.75,
            "type":"RealNumber",
            "initialvalue":None
        },
    ],
    "extra_func":[
        {
            "shortname":"displacement",
            "func":[
                    "Multiply", 
                    1.025, 
                    "x_1", 
                    "x_2", 
                    "x_4", 
                    "x_6"
                    ],
        },
        {
            "shortname":"V",
            "func":[
                    "Multiply", 
                    0.5144, 
                    "x_5"
                    ],
        },
        {
            "shortname":"Fn",
            "func":[
                    "Divide", 
                    "V", 
                    ["Sqrt", ["Multiply", "g", "x_1"]]
                    ],
        },
        {
            "shortname":"a",
            "func":[
                    "Add", 
                    ["Multiply", 4977.06, ["Square", "x_6"]], 
                    ["Multiply", -8105.61, "x_6"], 
                    4456.51
                    ],
        },
        {
            "shortname":"b",
            "func":[
                    "Add", 
                    ["Multiply", -10847.2, ["Square", "x_6"]], 
                    ["Multiply", 12817, "x_6"], 
                    -6960.32
                    ],
        },
        {
            "shortname":"power",
            "func":[
                    "Divide", 
                    [
                        "Multiply", 
                        ["Power", "x_5", 3], 
                        ["Power", "displacement", ["Rational", 2, 3]]
                    ], 
                    ["Add", ["Multiply", "b", "Fn"], "a"]
                    ],
        },
        {
            "shortname":"outfit_weight",
            "func":[
                    "Multiply", 
                    ["Power", "x_1", 0.8], 
                    ["Power", "x_2", 0.6], 
                    ["Power", "x_3", 0.3], 
                    ["Power", "x_6", 0.1]
                    ],
        },
        {
            "shortname":"steel_weight",
            "func":[
                    "Multiply", 
                    0.034, 
                    ["Power", "x_1", 1.7], 
                    ["Power", "x_2", 0.7], 
                    ["Power", "x_3", 0.4], 
                    ["Sqrt", "x_6"]
                    ],
        },
        {
            "shortname":"machinery_weight",
            "func":["Multiply", 0.17, ["Power", "power", 0.9]],
        },
        {
            "shortname":"light_ship_weight",
            "func":[
                "Add", 
                "steel_weight", 
                "outfit_weight", 
                "machinery_weight"
                ],
        },

        {
            "shortname":"ship_cost",
            "func":[
                "Multiply", 
                1.3, 
                [
                    "Add", 
                    ["Multiply", 2000, ["Power", "steel_weight", 0.85]], 
                    ["Multiply", 2400, ["Power", "power", 0.8]], 
                    ["Multiply", 3500, "outfit_weight"]
                ]
                ],
        },
        {
            "shortname":"capital_costs",
            "func":[
                "Multiply", 
                0.2, 
                "ship_cost"
                ],
        },
        {
            "shortname":"DWT",
            "func":["Subtract", "displacement", "light_ship_weight"],
        },
        {
            "shortname":"running_costs",
            "func":["Multiply", 40000, ["Power", "DWT", 0.3]],
        },
        {
            "shortname":"sea_days",
            "func":["Multiply", ["Divide", 5000.0, 24], "x_5",],
        },
        {
            "shortname":"daily_consumption",
            "func":["Add", ["Multiply", ["Rational", 3, 125], 0.19, "power"], 0.2 ],
        },
        {
            "shortname":"fuel_cost",
            "func":[
                "Multiply", 
                "daily_consumption", 
                "sea_days", 
                "fuel_price",
                1.05
                ],
        },
        {
            "shortname":"port_cost",
            "func":["Multiply", 6.3, ["Power", "DWT", 0.8]],
        },
        {
            "shortname":"fuel_carried",
            "func":["Multiply", "daily_consumption", ["Add","sea_days",5] ],
        },
        {
            "shortname":"miscellaneous_DWT",
            "func":["Multiply", 2, ["Power", "DWT", 0.5]],
        },
        {
            "shortname":"cargo_DWT",
            "func":["Subtract", "DWT", "fuel_carried", "miscellaneous_DWT"],
        },
        {
            "shortname":"port_days",
            "func":["Multiply", 2, ["Add", ["Divide", "cargo_DWT", "handling_rate"], 0.5]],
        },
        {
            "shortname":"RTPA",
            "func":["Divide", 350, ["Add", "sea_days", "port_days"]],
        },
        {
            "shortname":"voyage_costs",
            "func":["Multiply", "RTPA", ["Add", "fuel_cost", "port_cost"]],
        },
        {
            "shortname":"annual_costs",
            "func":["Add", "capital_costs", "running_costs","voyage_costs"],
        },
        {
            "shortname":"annual_cargo",
            "func":["Multiply", "cargo_DWT", "RTPA"],
        },

        {
            "shortname":"g_1",
            "func":["Negate",["Subtract", ["Divide", "x_1", "x_2"], 6] ],
        },
        {
            "shortname":"g_2",
            "func":["Negate", ["Subtract", 15, ["Divide", "x_1", "x_3"]]],
        },
        {
            "shortname":"g_3",
            "func":["Negate", ["Subtract", 19,["Divide", "x_1", "x_4"]]],
        },
        {
            "shortname":"g_4",
            "func":["Negate", ["Subtract", ["Multiply", 0.45, ["Power", "DWT", 0.31]], "x_4"] ],
        },
        {
            "shortname":"g_5",
            "func":["Negate",["Add", ["Multiply", 0.7, "x_3"], ["Negate", "x_4"], 0.7] ],
        },
        {
            "shortname":"g_6",
            "func":["Negate",["Subtract", 500000.0, "DWT"] ],
        },
        {
            "shortname":"g_7",
            "func":["Negate",["Subtract","DWT",3000.0] ],
        },
        {
            "shortname":"g_8",
            "func":["Negate",["Subtract",3000.0,"Fn"] ],
        },
        {
            "shortname":"g_9",
            "func":["Negate",[
                "Add", 
                [
                    "Divide", 
                    [
                    "Multiply", 
                    ["Add", ["Multiply", 0.085, "x_6"], -0.002], 
                    ["Square", "x_2"]
                    ], 
                    ["Multiply", "x_4", "x_6"]
                ], 
                ["Multiply", -0.07, "x_2"], 
                ["Multiply", -0.52, "x_3"], 
                ["Multiply", 0.53, "x_4"], 
                -1
                ]],
        },

    ],
    "objectives":[  
        {
            "longname":"minimize structural volume",
            "shortname":"f1",
            "func":["Divide", "annual_costs", "annual_cargo"],
            "max": False,
            "lowerbound":None,
            "upperbound":None
        },
        {
            "longname":"minimize the joint displacement",
            "shortname":"f2",
            "func":["Add", "steel_weight", "outfit_weight", "machinery_weight"],
            "max": False,
            "lowerbound":None,
            "upperbound":None
        },
        {
            "longname":"minimize the joint displacement",
            "shortname":"f3",
            "func":["Negate",[ "Multiply", "cargo_DWT", "RTPA"]],
            "max": False,
            "lowerbound":None,
            "upperbound":None
        },
        {
            "longname":"minimize the joint displacement",
            "shortname":"f4",
            "func":["Sum", ["Max","g_i",0], ["Triple", ["Hold", "i"], 1, 9]],
            "max": False,   
            "lowerbound":None,
            "upperbound":None
        }
    ],
    "constraints":[],
    "__problemName":"RE"
    }
    p = MOProblem(json=re42_json)
    return p

def re61():
    re61_json = {
    "constants":[],
    "variables":[
        {
            "longname":"Decision variable 1",
            "shortname":"x_1",
            "lowerbound":0.01,
            "upperbound":0.45,
            "type":"RealNumber",
            "initialvalue":None
        },
        {
            "longname":"Decision variable 2",
            "shortname":"x_2",
            "lowerbound":0.01,
            "upperbound":0.1,
            "type":"RealNumber",
            "initialvalue":None
        },
        {
            "shortname":"x_3",
            "lowerbound":0.01,
            "upperbound":0.1,
            "type":"RealNumber",
            "initialvalue":None
        },
    ],
    "extra_func":[
        {
            "shortname":"g_1",
            "func":["Negate",[
                    "Add", 
                    ["Divide", -0.00139, ["Multiply", "x_1", "x_2"]], 
                    ["Multiply", -4.94, "x_3"], 
                    0.08, 
                    1
                    ]],
        },
        {
            "shortname":"g_2",
            "func":["Negate",[
                "Add", 
                ["Divide", -0.000306, ["Multiply", "x_1", "x_2"]], 
                ["Multiply", -1.082, "x_3"], 
                0.0986, 
                1
                ]],
        },
        {
            "shortname":"g_3",
            "func":["Negate",[
                    "Add", 
                    ["Divide", -12.307, ["Multiply", "x_1", "x_2"]], 
                    ["Multiply", -49408.24, "x_3"], 
                    -4051.02, 
                    50000
                    ]],
        },
        {
            "shortname":"g_4",
            "func":["Negate",[
                    "Add", 
                    ["Divide", -2.098, ["Multiply", "x_1", "x_2"]], 
                    ["Multiply", -8046.33, "x_3"], 
                    696.71, 
                    16000
                    ]],
        },
        {
            "shortname":"g_5",
            "func":["Negate",[
                    "Add", 
                    ["Divide", -2.138, ["Multiply", "x_1", "x_2"]], 
                    ["Multiply", -7883.39, "x_3"], 
                    705.04, 
                    10000
                    ]],
        },
        {
            "shortname":"g_6",
            "func":["Negate",[
                    "Add", 
                    ["Multiply", -0.417, "x_1", "x_2"], 
                    ["Multiply", -1721.26, "x_3"], 
                    136.54, 
                    2000
                    ]],
        },
                {
            "shortname":"g_7",
            "func":["Negate", [
                    "Add", 
                    ["Divide", -0.164, ["Multiply", "x_1", "x_2"]], 
                    ["Multiply", -631.13, "x_3"], 
                    54.48, 
                    550
                    ] ],
        },
    ],
    "objectives":[  
        {
            "longname":"minimize structural volume",
            "shortname":"f1",
            "func":["Add", ["Multiply", 106780.37, ["Add", "x_2", "x_3"]], 61704.67],
            "max": False,
            "lowerbound":None,
            "upperbound":None
        },
        {
            "longname":"minimize the joint displacement",
            "shortname":"f2",
            "func":["Multiply", 3000,"x_1"],
            "max": False,
            "lowerbound":None,
            "upperbound":None
        },
        {
            "longname":"minimize the joint displacement",
            "shortname":"f3",
            "func":["Multiply", 699747300, "x_2", ["Power", ["Multiply", 2289, 0.06], -0.65]],
            "max": False,
            "lowerbound":None,
            "upperbound":None
        },
        {
            "longname":"minimize the joint displacement",
            "shortname":"f4",
            "func":[
                    "Multiply", 
                    572250, 
                    [
                        "Exp", 
                        [
                        "Add", 
                        ["Multiply", -39.75, "x_2"], 
                        ["Multiply", 9.9, "x_3"], 
                        2.74
                        ]
                    ]
                    ],
            "max": False,
            "lowerbound":None,
            "upperbound":None
        },
        {
            "longname":"minimize the joint displacement",
            "shortname":"f5",
            "func":[
                    "Multiply", 
                    25, 
                    [
                        "Add", 
                        ["Divide", 1.39, ["Multiply", "x_1", "x_2"]], 
                        ["Multiply", 4940, "x_3"], 
                        -80
                    ]
                    ],
            "max": False,
            "lowerbound":None,
            "upperbound":None
        },
        {
            "longname":"minimize the joint displacement",
            "shortname":"f6",
            "func":["Sum", ["Max","g_i",0], ["Triple", ["Hold", "i"], 1, 7]],
            "max": False,
            "lowerbound":None,
            "upperbound":None
        }
    ],
    "constraints":[],
    "__problemName":"RE"
    }
    p = MOProblem(json=re61_json)
    return p

def cre21():
    cre21_json = {
    "constants":[],
    "variables":[
        {
            "longname":"Decision variable 1",
            "shortname":"x_1",
            "lowerbound":0.00001,
            "upperbound":100.0,
            "type":"RealNumber",
            "initialvalue":None
        },
        {
            "longname":"Decision variable 2",
            "shortname":"x_2",
            "lowerbound":0.00001,
            "upperbound":100.0,
            "type":"RealNumber",
            "initialvalue":None
        },
        {
            "shortname":"x_3",
            "lowerbound":1,
            "upperbound":3.0,
            "type":"RealNumber",
            "initialvalue":None
        },
    ],
    "extra_func":[
        {
            "shortname":"f_1",
            "func":[
                    "Add", 
                    [
                        "Multiply", 
                        "x_1", 
                        ["Sqrt", ["Add", ["Square", "x_3"], 16]]
                    ], 
                    [
                        "Multiply", 
                        "x_2", 
                        ["Sqrt", ["Add", ["Square", "x_3"], 1]]
                    ]
                    ],
        },
        {
            "shortname":"f_2",
            "func":[
                "Divide", 
                [
                    "Sqrt", 
                    [
                    "Multiply", 
                    20, 
                    ["Add", ["Square", "x_3"], 16]
                    ]
                ], 
                ["Multiply", "x_1", "x_3"]
                ],
        },
    ],
    "objectives":[  
        {
            "shortname":"f1",
            "func":[
                    "Add", 
                    [
                        "Multiply", 
                        "x_1", 
                        ["Sqrt", ["Add", ["Square", "x_3"], 16]]
                    ], 
                    [
                        "Multiply", 
                        "x_2", 
                        ["Sqrt", ["Add", ["Square", "x_3"], 1]]
                    ]
                    ],
            "max": False,
            "lowerbound":None,
            "upperbound":None
        },
        {
            "shortname":"f2",
            "func":["Divide", 
                        ["Multiply", 
                            20,
                            ["Sqrt", ["Add", ["Square", "x_3"], 16]]
                        ],
                        ["Multiply", "x_1", "x_3"]
                    ],
            "max": False,
            "lowerbound":None,
            "upperbound":None
        },
    ],
    "constraints":[
        {
            "shortname":"g1",
            "func":["Max",["Negate",["Subtract", 0.1, "f_1",]],0],
        },
        {
            "shortname":"g2",
            "func":["Max",["Negate",["Subtract", 100000, "f_2",]],0],
        },
        {
            "shortname":"g_3",
            "func":["Max",[
                "Subtract", 
                [
                    "Divide", 
                    [
                    "Sqrt", 
                    [
                        "Multiply", 
                        80, 
                        ["Add", ["Square", "x_3"], 1]
                    ]
                    ], 
                    ["Multiply", "x_2", "x_3"]
                ], 
                100000
                ],0],
        },
    ],
    "__problemName":"RE"
    }
    p = MOProblem(json=cre21_json)
    return p

def cre22():
    cre22_json = {
    "constants":[
        {"shortname":"P","value":6000},
        {"shortname":"L","value":14},
        {"shortname":"E","value":30e6},
        {"shortname":"G","value":12e6},
        {"shortname":"tauMax","value":13600},
        {"shortname":"sigmaMax","value":30000},
    ],
    "variables":[
        {
            "longname":"Decision variable 1",
            "shortname":"x_1",
            "lowerbound":0.125,
            "upperbound":5.0,
            "type":"RealNumber",
            "initialvalue":None
        },
        {
            "longname":"Decision variable 2",
            "shortname":"x_2",
            "lowerbound":0.1,
            "upperbound":10.0,
            "type":"RealNumber",
            "initialvalue":None
        },
        {
            "shortname":"x_3",
            "lowerbound":0.1,
            "upperbound":10.0,
            "type":"RealNumber",
            "initialvalue":None
        },
        {
            "shortname":"x_4",
            "lowerbound":0.125,
            "upperbound":5.0,
            "type":"RealNumber",
            "initialvalue":None
        },
    ],
    "extra_func":[
        {
            "shortname":"M",
            "func":["Multiply", "P", ["Add", "L", 
                    ["Multiply", ["Rational", 1, 2], "x_2"]]
                ],
        },
        {
            "shortname":"tmpVar1",
            "func":[
                    "Add", 
                    [
                        "Multiply", 
                        ["Square", ["Rational", 1, 2]], 
                        ["Square", ["Add", "x_1", "x_3"]]
                    ], 
                    [
                        "Multiply", 
                        ["Rational", 1, 4], 
                        ["Square", "x_2"]
                    ]
                    ],
        },
        {
            "shortname":"R",
            "func":["Sqrt", "tmpVar1"],
        },
        {
            "shortname":"tmpVar2",
            "func":[
                    "Add", 
                    [
                        "Multiply", 
                        ["Square", ["Rational", 1, 2]], 
                        ["Square", ["Add", "x_1", "x_3"]]
                    ], 
                    [
                        "Multiply", 
                        ["Rational", 1, 12], 
                        ["Square", "x_2"]
                    ]
                    ],
        },
        { 
            "shortname":"J",
            "func":["Multiply",2,["Sqrt",2],"x_1","x_2","tmpVar2"]
        },
        {
            "shortname":"tauDashDash",
            "func":["Divide", ["Multiply","M","R"],"J"  ],
        },
        {
            "shortname":"tauDash",
            "func":["Divide", "P", ["Multiply", "x_1", "x_2", ["Sqrt", 2]]],
        },
        {
            "shortname":"tmpVar3",
            "func":["Add", 
                    ["Divide", ["Multiply", "tauDash", "x_2", "tauDashDash"], "R"], 
                    ["Square", "tauDash"], 
                    ["Square", "tauDashDash"]
                    ],
        },
        {
            "shortname":"tau",
            "func":["Sqrt", "tmpVar3" ],
        },
        {
            "shortname":"sigma",
            "func":["Divide", 
                    ["Multiply", 6, "L", "P"], 
                    ["Multiply", "x_4", ["Square", "x_3"]]],
        },
        {
            "shortname":"tmpVar4",
            "func":[
                        "Divide", 
                        [
                            "Multiply", 
                            4.013, 
                            "E", 
                            [
                            "Sqrt", 
                            [
                                "Multiply", 
                                ["Rational", 1, 36], 
                                ["Square", "x_3"], 
                                ["Power", "x_4", 6]
                            ]
                            ]
                        ], 
                        ["Square", "L"]
                        ],
        },
        {
            "shortname":"tmpVar5",
            "func":[
                    "Divide", 
                    [
                        "Multiply", 
                        ["Rational", 1, 2], 
                        "x_3", 
                        [
                        "Sqrt", 
                        [
                            "Divide", 
                            ["Multiply", ["Rational", 1, 4], "E"], 
                            "G"
                        ]
                        ]
                    ], 
                    "L"
                    ],
        },
        {
            "shortname":"PC",
            "func":["Multiply", 
                    "tmpVar4",
                    ["Subtract", 1, "tmpVar5"]],
        },
    ],
    "objectives":[  
        {
            "shortname":"f1",
            "func":["Add", ["Multiply", 1.10471, "x_2", ["Square", "x_1"]], 
                    [
                        "Multiply", 
                        0.04811, 
                        "x_3", 
                        "x_4", 
                        ["Add", "x_2", 14]
                    ]
                    ],
            "max": False,
            "lowerbound":None,
            "upperbound":None
        },
        {
            "shortname":"f2",
            "func":[
                    "Divide", 
                    [
                        "Multiply", 
                        4, 
                        "P", 
                        ["Power", "L", 3]
                    ], 
                    [
                        "Multiply", 
                        "E", 
                        "x_4", 
                        ["Power", "x_3", 3]
                    ]
                    ],
            "max": False,
            "lowerbound":None,
            "upperbound":None
        },
    ],
    "constraints":[
        {
            "shortname":"g1",
            "func":["Max",["Negate",["Subtract", "tauMax", "tau",]],0],
        },
        {
            "shortname":"g2",
            "func":["Max",["Negate",["Subtract", "sigmaMax", "sigma",]],0],
        },
        {
            "shortname":"g3",
            "func":["Max",["Negate",["Subtract", "x_4", "x_1",]],0],
        },
        {
            "shortname":"g4",
            "func":["Max",["Negate",["Subtract", "PC", "P",]],0],
        },
    ],
    "__problemName":"RE"
    }
    p = MOProblem(json=cre22_json)
    return p

def cre23():
    cre23_json = {
    "constants":[],
    "variables":[
        {
            "longname":"Decision variable 1",
            "shortname":"x_1",
            "lowerbound":55,
            "upperbound":80,
            "type":"RealNumber",
            "initialvalue":None
        },
        {
            "longname":"Decision variable 2",
            "shortname":"x_2",
            "lowerbound":75,
            "upperbound":110,
            "type":"RealNumber",
            "initialvalue":None
        },
        {
            "shortname":"x_3",
            "lowerbound":1000,
            "upperbound":3000,
            "type":"RealNumber",
            "initialvalue":None
        },
        {
            "shortname":"x_4",
            "lowerbound":11,
            "upperbound":20,
            "type":"RealNumber",
            "initialvalue":None
        },
    ],
    "extra_func":[
    ],
    "objectives":[  
        {
            "shortname":"f1",
            "func":["Multiply", 
                    1e-5, 
                    ["Subtract", "x_4", 1], 
                    ["Subtract", ["Square", "x_2"], 
                    ["Square", "x_1"]]],
            "max": False,
            "lowerbound":None,
            "upperbound":None
        },
        {
            "shortname":"f2",
            "func":[
                    "Divide", 
                    [
                        "Multiply", 
                        9.82, 
                        1e6, 
                        ["Subtract", ["Square", "x_2"], ["Square", "x_1"]]
                    ], 
                    [
                        "Multiply", 
                        "x_3", 
                        "x_4", 
                        [
                        "Subtract", 
                        ["Power", "x_2", 3], 
                        ["Power", "x_1", 3]
                        ]
                    ]
                    ],
            "max": False,
            "lowerbound":None,
            "upperbound":None
        },
    ],
    "constraints":[
        {
            "shortname":"g1",
            "func":["Max",["Negate",["Add", ["Negate", "x_1"], "x_2", -20] ],0],
        },
        {
            "shortname":"g2",
            "func":["Max",["Negate",
                          [
                            "Add", 
                            [
                                "Divide", 
                                ["Negate", "x_3"], 
                                [
                                "Multiply", 
                                3.14, 
                                ["Subtract", ["Square", "x_2"], ["Square", "x_1"]]
                                ]
                            ], 
                            0.4
                            ]         
                           ],0],
        },
        {
            "shortname":"g3",
            "func":["Max",["Negate",
                            [
                            "Add", 
                            [
                                "Divide", 
                                [
                                "Multiply", 
                                -2.22, 
                                1e-3, 
                                "x_3", 
                                [
                                    "Subtract", 
                                    ["Power", "x_2", 3], 
                                    ["Power", "x_1", 3]
                                ]
                                ], 
                                [
                                "Square", 
                                ["Subtract", ["Square", "x_2"], ["Square", "x_1"]]
                                ]
                            ], 
                            1
                            ]                          
                                    
                           ],0],
        },
        {
            "shortname":"g4",
            "func":["Max",["Negate",
                           [
  "Subtract", 
  [
    "Divide", 
    [
      "Multiply", 
      2.66, 
      1e-2, 
      "x_3", 
      "x_4", 
      [
        "Subtract", 
        ["Power", "x_2", 3], 
        ["Power", "x_1", 3]
      ]
    ], 
    ["Subtract", ["Square", "x_2"], ["Square", "x_1"]]
  ], 
  900
]
                           ],0],
        },
    ],
    "__problemName":"RE"
    }
    p = MOProblem(json=cre23_json)
    return p

import plotly.graph_objects as go
from desdeo_emo.EAs.NSGAIII import NSGAIII
#["Negate",
def test():
    p: MOProblem = re23()
    # Variable values
    re22_test = np.array([[6.37192567e+00, 1.44064899e+01, 4.57499269e-03]])
    re23_test = np.array([[42.28517847, 72.31212485, 10.02173122, 79.53649171]])
    re24_test = np.array([[ 1.95957702, 36.15606243]])
    re25_test = np.array([[29.77451832,  2.32877878,  0.09004689]])
    
    objective_vectors = p.evaluate(re23_test).objectives
    print("objectve= ", objective_vectors)

# test()

def test3():
    p = re61()
    re34_test = np.array([[1.83404401, 2.44064899, 1.00022875, 1.60466515, 1.29351178]])
    re37_test = np.array([[4.17022005e-01, 7.20324493e-01, 1.14374817e-04, 3.02332573e-01]])
    re41_test = np.array([[0.917022,   1.09829204, 0.50011437, 0.80233257,
                    1.13182281, 0.47387088,0.54900817]])
    
    re42_test = np.array([[201.84417562,  28.86719451,  13.0013725,
                              10.5169887,   14.58702356, 0.64108063]])
    re61_test = np.array([[0.19348968, 0.0748292,  0.01001029]])
    objective_vectors = p.evaluate(re61_test).objectives
    print("objectve= ", objective_vectors)
    #Testing EMO NSGAII
    # evolver = NSGAIII(p,
    #                 n_iterations=10,
    #                 n_gen_per_iter=100,
    #                 population_size=100)  

    # while evolver.continue_evolution():
    #     evolver.iterate()

    # individuals, solutions, _ = evolver.end()

    # fig1 = go.Figure(
    #     data=go.Scatter(
    #         x=individuals[:,0],
    #         y=individuals[:,1],
    #         mode="markers"))
    # fig1.show()

# test3()
def test4():
    p = cre23()
    cre21_test = np.array([[41.7022063,  72.03245214,  1.00022875]])
    cre22_test = np.array([[2.15798227, 7.23121249, 0.10113231, 1.59887129]])
    cre23_test = np.array([ [  65.42555012,  100.21135727, 1000.22874963,13.72099315]])
    r = p.evaluate(cre23_test)
    print(r)

test4()