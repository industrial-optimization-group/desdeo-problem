import numpy as np

from desdeo_problem.Objective import ScalarObjective, VectorObjective
from desdeo_problem.Problem import MOProblem
from desdeo_problem.Variable import variable_builder

var_names = ["a", "b", "c"]
initial_values = [1, 1, 1]
lower_bounds = [0, -1, 0]
upper_bounds = [5, 10, 3]
variables = variable_builder(var_names, initial_values, lower_bounds, upper_bounds)


def obj1_2(x):
    y1 = x[0] + x[1] + x[2]
    y2 = x[0] * x[1] * x[2]
    return (y1, y2)


def obj3(x):
    y3 = x[0] * x[1] + x[2]
    return y3


f1_2= VectorObjective(['y1', 'y2'], obj1_2)
f3 = ScalarObjective("f3", obj3)
prob = MOProblem([f1_2, f3], variables, None)
data = np.asarray([[1, 1, 1], [1, -1, 0]])
print(prob.evaluate(data))
