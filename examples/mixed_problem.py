
# %%
from desdeo_problem.Objective import ScalarObjective, VectorObjective
from desdeo_problem.Problem import MOProblem
from desdeo_problem.Variable import variable_builder
from desdeo_problem.Constraint import ScalarConstraint
import numpy as np

# %%
var_names = ["a", "b", "c"]
initial_values = [1, 1, 1]
lower_bounds = [-2, -1, 0]
upper_bounds = [5, 10, 3]
variables = variable_builder(var_names, initial_values, lower_bounds, upper_bounds)
print("Type of \"variables\": ", type(variables))
print("Length of \"variables\": ", len(variables))
print("Type of the contents of \"variables\": ", type(variables[0]))


# %%
def obj1_2(x):
    y1 = x[0] + x[1] + x[2]
    y2 = x[0] * x[1] * x[2]
    return (y1, y2)


def obj3(x):
    y3 = x[0] * x[1] + x[2]
    return y3


# %%
f1_2 = VectorObjective(["y1", "y2"], obj1_2)
f3 = ScalarObjective("f3", obj3)

cons1 = ScalarConstraint("c_1", 3, 3, lambda x, _: 10 - (x[0] + x[1] + x[2]))

# %%
prob = MOProblem(objectives=[f1_2, f3], variables=variables, constraints=[cons1])

# %%
data = np.asarray([[1, -1, 0], [5, 5, 2]])
res = prob.evaluate(data)

print(res)
