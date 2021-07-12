"""Desdeo-problem package

This package is for creating a problem for desdeo to solve.
It includes modules for Variables, Objectives, Constraints, and actual Problem.
It also has subpackage for surrogate models for creating computationally
simpler surrogate models for problems.
Package also includes a sub package with test problems.
"""


__all__ = [
    "ScalarConstraint",
    "ConstraintError",
    "constraint_function_factory",
    "ConstraintError",
    "supported_operators",
    "ObjectiveBase",
    "ObjectiveError",
    "ObjectiveEvaluationResults",
    "VectorObjectiveBase",
    "_ScalarObjective",
    "VectorObjective",
    "_ScalarDataObjective",
    "VectorDataObjective",
    "ProblemError",
    "ProblemBase",
    "EvaluationResults",
    "ScalarMOProblem",
    "ScalarDataProblem",
    "MOProblem",
    "DataProblem",
    "ExperimentalProblem",
    "VariableError",
    "VariableBuilderError",
    "Variable",
    "variable_builder",
    "BaseRegressor",
    "GaussianProcessRegressor",
    "LipschitzianRegressor",
    "ModelError",
    "DiscreteDataProblem",
    "classificationPISProblem",
]

from desdeo_problem.Constraint import (
    ConstraintBase,
    ConstraintError,
    ScalarConstraint,
    constraint_function_factory,
    supported_operators,
)
from desdeo_problem.Objective import (
    ObjectiveBase,
    ObjectiveError,
    ObjectiveEvaluationResults,
    VectorDataObjective,
    VectorObjective,
    VectorObjectiveBase,
    _ScalarDataObjective,
    _ScalarObjective,
)
from desdeo_problem.Problem import (
    DataProblem,
    EvaluationResults,
    ExperimentalProblem,
    MOProblem,
    ProblemBase,
    ProblemError,
    ScalarDataProblem,
    ScalarMOProblem,
    DiscreteDataProblem,
    classificationPISProblem,
)
from desdeo_problem.surrogatemodels import (
    BaseRegressor,
    GaussianProcessRegressor,
    LipschitzianRegressor,
    ModelError,
)
from desdeo_problem.testproblems import TestProblems
from desdeo_problem.Variable import (
    Variable,
    VariableBuilderError,
    VariableError,
    variable_builder,
)
