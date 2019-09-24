from desdeo_problem.testproblems.TestProblems import test_problem_builder
import numpy as np


zdt1 = test_problem_builder("ZDT1")


dtlz3 = test_problem_builder("DTLZ3", n_of_objectives=3, n_of_variables=20)


number_of_samples = 3
zdt_data = np.random.random(
    (number_of_samples, 30)
)  # 30 is the number of variables in the ZDT1 problem

dtlz_data = np.random.random(
    (number_of_samples, 20)
)  # We put the number of variables earlier as 20

zdt_obj_val = zdt1.evaluate(zdt_data)
zdt_obj_val

dtlz_obj_val = dtlz3.evaluate(dtlz_data)
dtlz_obj_val