from desdeo_problem import MOProblem
from desdeo_problem.testproblems.TestProblems import test_problem_builder


problem_name = "ZDT1"
problem = test_problem_builder(problem_name)

problem_name = "DTLZ1" 
problem = test_problem_builder(problem_name, n_of_variables=7, n_of_objectives=3)
    
problem_name = "DTLZ2" 
problem = test_problem_builder(problem_name, n_of_variables=12, n_of_objectives=3)

problem_name = "DTLZ4"
problem = test_problem_builder(problem_name, n_of_variables=12, n_of_objectives=3)

problem_name = "DTLZ6"
problem = test_problem_builder(problem_name, n_of_variables=12, n_of_objectives=3)

problem_name = "DTLZ7"
problem = test_problem_builder(problem_name, n_of_variables=22, n_of_objectives=3)
print("success")
