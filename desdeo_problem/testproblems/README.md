# DBMOPP Generator

This DBMOPP generator is based on the original Distance-Based Multi-Objective Point Problem (DBMOPP) instance generator https://github.com/fieldsend/DBMOPP_generator.
Currently the usage and code is very similar to the original Matlab version.

More detailed tutorial at desdeo_problem/docs/notebooks/DBMOPP_tutorial

## What works 
Combining DBMOPP with DESDEO's MOProblem instances for solving the generated
testproblems with DESDEO's methods is work in progress.

- Original DBMOPP generator's functionality is done.
- MOProblem can be called to evaluate objective function values.
- Constraints have been redefined, currently vertex soft constraints will return
correct values when called with
DBMOPP.evaluate_soft_constraints(x), where x is vector containing n number of
decision vectors. As with DESDEO's way, if returned value is positive, it means
constraint is not violated and negative means it violated. The value itself is
the distance from the point x_i to the vertex border.

## What does not work yet

- MOProblem cannot be called to get constraint violation values.
- Constraints are in the middle of rewriting to fit better to use with DESDEO.

## Example usage 

Define the wanted problem
