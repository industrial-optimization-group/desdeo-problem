import plotly.graph_objects as go
from desdeo_emo.EAs.NSGAIII import NSGAIII
from desdeo_emo.utilities.plotlyanimate import animate_init_, animate_next_
from desdeo_problem import MOProblem, ScalarObjective, variable_builder
import json
import numpy as np
f = open('desdeo_problem/problem/mop7.json')  
data = json.load(f)
p = MOProblem(json=data)

# Testing EMO NSGAII
evolver = NSGAIII(p,
                  n_iterations=10,
                  n_gen_per_iter=100,
                  population_size=100)  

while evolver.continue_evolution():
    evolver.iterate()

individuals, solutions, _ = evolver.end()

fig1 = go.Figure(
    data=go.Scatter(
        x=individuals[:,0],
        y=individuals[:,1],
        mode="markers"))
fig1.show()
# def f_4(x):
#     term = x[:, 0]**2 + x[:, 1]**2
#     return 0.5*term + np.sin(term)

# def f_5(x):
#     term1 = ((3*x[:, 0] - 2*x[:,1] + 4)**2)/8
#     term2 = ((x[:, 0] - x[:,1] + 1)**2)/27
#     return term1 + term2 + 15

# def f_6(x):
#     term = x[:, 0]**2 + x[:, 1]**2
#     return (1/(term + 1)) - 1.1 * np.exp(-term)
# list_vars = variable_builder(['x', 'y'],
#                              initial_values = [0,0],
#                              lower_bounds=[-30, -30],
#                              upper_bounds=[30, 30])
# f1 = ScalarObjective(name='f1', evaluator=f_4)
# f2 = ScalarObjective(name='f2', evaluator=f_5)
# f3 = ScalarObjective(name='f3', evaluator=f_6)
# problem = MOProblem(variables=list_vars, objectives=[f1, f2, f3])
# # evolver = NSGAIII(p, keep_archive=True)
# individual, solutions, archive = evolver.end()
# figure = animate_init_(solutions, filename="MOP5.html")

# from desdeo_mcdm.interactive.NIMBUS import NIMBUS

# method = NIMBUS(problem, "scipy_de")

# classification_request, plot_request = method.start()
