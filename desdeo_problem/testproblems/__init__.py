"""A sub-package for desdeo-problem for generating default testproblems.

Current implementation has DTLZ1-7 and ZDT1-6 problems from optproblems
package
"""

__all__ = ["test_problem_builder"]

from desdeo_problem.testproblems.TestProblems import test_problem_builder
