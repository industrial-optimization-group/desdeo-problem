from desdeo_problem.testproblems.DBMOPP.utilities import get_2D_version, euclidean_distance, \
    get_random_angles, between_lines_rooted_at_pivot, assign_design_dimension_projection
from typing import Dict, Tuple
import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from desdeo_problem.problem import MOProblem, ScalarObjective, variable_builder, ScalarConstraint, VectorObjective, EvaluationResults
from matplotlib import cm
from desdeo_problem.testproblems.DBMOPP.Region import AttractorRegion, Attractor, Region
from desdeo_problem.testproblems.DBMOPP.DBMOPP_generator import DBMOPP_generator


import random

#import cProfile
#import re
#cProfile.run('re.compile("DBMOPP_generator")', 'stats')

n_objectives = 5 
n_variables = 4 
n_local_pareto_regions = 2 
n_dominance_res_regions = 1 
n_global_pareto_regions = 3 
const_space = 0.1
pareto_set_type = 2 
constraint_type = 4 
ndo = 1 #numberOfdiscontinousObjectiveFunctionRegions
neutral_space = 0.1


# 0: No constraint, 1-4: Hard vertex, centre, moat, extended checker, 
# 5-8: soft vertex, centre, moat, extended checker.

problem = DBMOPP_generator(
    n_objectives,
    n_variables,
    n_local_pareto_regions,
    n_dominance_res_regions,
    n_global_pareto_regions,
    const_space,
    pareto_set_type,
    constraint_type, 
    ndo, False, False, neutral_space, 10000
)
print("Initializing works!")
