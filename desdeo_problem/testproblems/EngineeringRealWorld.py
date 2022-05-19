from desdeo_problem.problem.Variable import Variable
from desdeo_problem.problem.Objective import ScalarObjective
from desdeo_problem.problem.Problem import MOProblem, ProblemBase

import numpy as np

def re21(var_iv: np.array = ([2, 2, 2, 2])) -> MOProblem:
    """ Four bar truss design problem
    
    Arguments:
        var_iv (np.array): Optional, initial variable values.
            Defaults are [2, 2, 2, 2]. x1 and x4 lb is 1, x2 and x3 lb is sqrt(2),
            ub for all xs is 3.
    Returns:
        MOProblem: a problem object.
    """

    F = 10.0
    sigma = 10.0
    E = 2.0 * 1e5
    L = 200.0
    a = F / sigma

    def f_1(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return L * ((2 * x[:, 0]) + np.sqrt(2.0) * x[:, 1] + np.sqrt(x[:, 2]) + x[:, 3])

    def f_2(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return ((F * L) / E) * ((2.0 / x[:, 0]) + 
            (2.0 * np.sqrt(2.0) / x[:, 1]) - (2.0 * np.sqrt(2.0) / x[:, 2]) + (2.0 / x[:, 3]))

    objective_1 = ScalarObjective(name="name", evaluator=f_1, maximize=[False])
    objective_2 = ScalarObjective(name="name", evaluator=f_2, maximize=[False])

    objectives = [objective_1, objective_2]

    x_1 = Variable("x_1", 2 * a, a, 3 * a)
    x_2 = Variable("x_2", 2 * a, (np.sqrt(2.0) * a), 3 * a)
    x_3 = Variable("x_3", 2 * a, (np.sqrt(2.0) * a), 3 * a)
    x_4 = Variable("x_4", 2 * a, a, 3 * a)

    variables = [x_1, x_2, x_3, x_4]

    problem = MOProblem(variables=variables, objectives=objectives)

    return problem