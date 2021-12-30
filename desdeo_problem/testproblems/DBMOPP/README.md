# DBMOPP Generator

This DBMOPP generator is based on the original Distance-Based Multi-Objective Point Problem (DBMOPP) instance generator https://github.com/fieldsend/DBMOPP_generator.
Currently the usage and code is very similar to the original Matlab version.


## Example usage 

Example usage as notebook-tutorial at desdeo_problem/docs/notebooks/DBMOPP_tutorial

## What works 
Combining DBMOPP with DESDEO's MOProblem instances for solving the generated
testproblems with DESDEO's methods is work in progress.

- Original DBMOPP generator's functionality is done.
- MOProblem can be called to evaluate objective function values.
- MOProblem can be called to evaluate constraints.
- Constraints include:
    - Hard / Soft vertex constraints
    - Hard / Soft center constraints
    - Hard / Soft moat constraints
    - Hard / Soft extended checker constraints

## What does not work yet

- Disconnected Pareto penalty regions are not yet plotted.

