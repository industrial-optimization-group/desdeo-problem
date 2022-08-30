"""
Optimal design of multiple clutch brakes.

Osyczka, A. (1992). Computer aided multicriterion optimization 
system (CAMOS) : software package in FORTRAN.

"""

from desdeo_problem.problem.Variable import Variable
from desdeo_problem.problem.Objective import ScalarObjective
from desdeo_problem.problem.Problem import MOProblem, ProblemBase
from desdeo_problem import ScalarConstraint, problem

import numpy as np

def multiple_clutch_brakes(var_iv: np.array = np.array([67.5, 92.5, 2.25, 650, 6])) -> MOProblem:
    """ Multiple clutch brake problem.
    
    Arguments:
        var_iv (np.array): Optional, initial variable values.
            Defaults are [67.5, 92.5, 2.25, 650, 6]. 
            x1 ∈ [55, 80], x2 ∈ [75, 110], x3 ∈ [1.5, 3], 
            x4 ∈ [300, 1000] and x5 ∈ {2, ..., 10}

    Returns:
        MOProblem: a problem object.
    """

    # Check the number of variables
    if (np.shape(np.atleast_2d(var_iv)[0]) != (5,)):
        raise RuntimeError("Number of variables must be seven")

    # Lower bounds
    lb = np.array([55, 75, 1.5, 300, 2])
    
    # Upper bounds
    ub = np.array([80, 110, 3, 1000, 10])

    # Check the variable bounds
    if np.any(lb > var_iv) or np.any(ub < var_iv):
        raise ValueError("Initial variable values need to be between lower and upper bounds")

    def m_h(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            (2/3) * 0.5 * x[:,3] * (np.round(x[:,4])) * 
            ((x[:,1]**3 - x[:,0]**3) / (x[:,1]**2 - x[:,0]**2))
            * 0.001 # This line is from Jussi's Matlab code
            
            # Mh = (2/3) * mu* x(5-1) * x(6-1) * ((x(3-1)^3-x(2-1)^3)/(x(3-1)^2-x(2-1)^2)) * 0.001; %convert to Nm
        )

    def p_rz(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            x[:, 3] / (np.pi * x[:, 1]**2 - np.pi * x[:, 0]**2)
        )
    
    def v_sr(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            (np.pi * (2/3 * ((x[:, 1]**3 - x[:, 0]**3) / 
            (x[:, 1]**2 - x[:, 0]**2))) * 250) / 30
        )

    # Objective functions
    def f_1(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            np.pi * ((x[:, 1]**2 - x[:, 0]**2)) * (x[:, 2] * (np.round(x[:, 4]) + 1)) * 0.0000078
        )

    def f_2(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            (55 * ((np.pi * 250) / 30)) / ((m_h(x)) + 3)
        )
    
    def f_3(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            np.round(x[:, 4])
        )

    def f_4(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            x[:, 1]
        )

    def f_5(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            x[:, 3]
        )

    # Constraint functions
    def g_1(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            x[:, 0] - 55
        )

    def g_2(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            110 - x[:, 1]
        )

    def g_3(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            x[:, 1] - x[:, 0] - 20 
        )

    def g_4(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            x[:, 2] - 1.5
        )

    def g_5(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            3 - x[:, 2]
        )

    def g_6(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            30 - (np.round(x[:, 4]) + 1) * (x[:, 2] + 0.5)
        )

    def g_7(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            10 - (np.round(x[:, 4]) + 1)
        )

    def g_8(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            np.round(x[:, 4]) - 1
        )
    
    def g_9(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            10 - p_rz(x)
        )

    def g_10(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            10 * 2000 - (p_rz(x) * v_sr(x))
        )

    def g_11(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            2000 - v_sr(x)
        )

    def g_12(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            15 - f_2(x)
        )

    def g_13(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            m_h(x) - (1.5 * 40)
        )
            
    def g_14(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            f_2(x)
        )

    def g_15(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            x[:, 3]
        )

    def g_16(x: np.ndarray, _ = None) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            1000 - x[:, 3]
        )

    objective_1 = ScalarObjective(name="mass of the brake", evaluator=f_1, maximize=[False])
    objective_2 = ScalarObjective(name="stopping time", evaluator=f_2, maximize=[False])
    objective_3 = ScalarObjective(name="number of friction surfaces", evaluator=f_3, maximize=[False])
    objective_4 = ScalarObjective(name="outer radius", evaluator=f_4, maximize=[False])
    objective_5 = ScalarObjective(name="actuating force", evaluator=f_5, maximize=[False])

    objectives = [objective_1, objective_2, objective_3, objective_4, objective_5]

    cons_1 = ScalarConstraint("c_1", 5, 5, g_1)
    cons_2 = ScalarConstraint("c_2", 5, 5, g_2)
    cons_3 = ScalarConstraint("c_3", 5, 5, g_3)
    cons_4 = ScalarConstraint("c_4", 5, 5, g_4)
    cons_5 = ScalarConstraint("c_5", 5, 5, g_5)
    cons_6 = ScalarConstraint("c_6", 5, 5, g_6)
    cons_7 = ScalarConstraint("c_7", 5, 5, g_7)
    cons_8 = ScalarConstraint("c_8", 5, 5, g_8)
    cons_9 = ScalarConstraint("c_9", 5, 5, g_9)
    cons_10 = ScalarConstraint("c_10", 5, 5, g_10)
    cons_11 = ScalarConstraint("c_11", 5, 5, g_11)
    cons_12 = ScalarConstraint("c_12", 5, 5, g_12)
    cons_13 = ScalarConstraint("c_13", 5, 5, g_13)
    cons_14 = ScalarConstraint("c_14", 5, 5, g_14)
    cons_15 = ScalarConstraint("c_15", 5, 5, g_15)
    cons_16 = ScalarConstraint("c_16", 5, 5, g_16)


    constraints = [cons_1, cons_2, cons_3, cons_4, cons_5, cons_6, cons_7, cons_8, 
        cons_9, cons_10, cons_11, cons_12, cons_13, cons_14, cons_15, cons_16]

    x_1 = Variable("inner radius", 67.5, 55, 80)
    x_2 = Variable("outer radius", 92.5, 75, 110)
    x_3 = Variable("thickness of the disc", 2.25, 1.5, 3)
    x_4 = Variable("actuating force", 650, 300, 1000)
    x_5 = Variable("number of friction surfaces", 6, 2, 10)

    variables = [x_1, x_2, x_3, x_4, x_5]

    problem = MOProblem(variables=variables, objectives=objectives, constraints=constraints)

    return problem