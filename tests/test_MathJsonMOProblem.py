import pytest
import numpy as np
from desdeo_problem.problem import MathJsonMOProblem
from desdeo_problem.problem.Problem import ProblemError
import numpy.testing as npt
#DEFINE TESTCASES

real_example21 = {
    "constants":[
        {
            "longname":"Force",
            "shortname":"F",
            "value":10
        },
        {
            "longname":"Force",
            "shortname":"E",
            "value":2e5
        },
        {
            "shortname":"L",
            "value":200,
            "__description":"Length, unit: cm"
        },
        {
            "shortname":"sigma",
            "value":10,
            "__description":"Length, unit: kN/cm^2"
        },
        {
            "shortname":"a",
            "value":1.0,
            "__description":"use for Variable bounds "
        }
    ],
    "variables":[
        {
            "longname":"Decision variable 1",
            "shortname":"x1",
            "lowerbound":"a",
            "upperbound":["Multiply",3,"a"],
            "initialvalue":None
        },
        {
            "longname":"Decision variable 2",
            "shortname":"x2",
            "lowerbound":["Multiply",["Sqrt",2],"a"],
            "upperbound":["Multiply",3,"a"],
            "initialvalue":None
        },
        {
            "shortname":"x3",
            "lowerbound":["Multiply",["Sqrt",2],"a"],
            "upperbound":["Multiply",3,"a"],
            "initialvalue":None
        },
        {
            "shortname":"x4",
            "lowerbound":"a",
            "upperbound":["Multiply",3,"a"],
            "initialvalue":None
        }
    ],
    "extra_func":[

    ],
    "objectives":[  
        {
            "longname":"minimize structural volume",
            "shortname":"f1",
            "func":
                [
                  "Multiply",
                  "L",
                  [
                    "Add",
                    ["Multiply", ["Sqrt", 2], "x2"],
                    ["Multiply", 2, "x1"],
                    ["Sqrt", "x3"],
                    "x4"
                  ]
                ],
            "max": False,
            "lowerbound":None,
            "upperbound":None
        },
        {
            "longname":"minimize the joint displacement",
            "shortname":"f2",
            "func":[
                "Divide",
                [
                  "Multiply",
                  "F",
                  "L",
                  [
                    "Add",
                    ["Divide", 2, "x1"],
                    ["Divide", 2, "x4"],
                    ["Divide", ["Multiply", 2, ["Sqrt", 2]], "x2"],
                    ["Divide", ["Multiply", -2, ["Sqrt", 2]], "x2"]
                  ]
                ],
                "E"
              ],
            "max": False,
            "lowerbound":None,
            "upperbound":None
        }
    ],
    "constraints":[],
    "__problemName":"Four bar truss design problem",
    "__problemDescription":"This problem is from DESDEO example Engineering real-world test problems on https://desdeo-problem.readthedocs.io/en/latest/problems/engineering_real_world.html#re-21-four-bar-truss-design-problem"
}

real_example22 = {
    "constants":[],
    "variables":[
        {
            "longname":"Decision variable 1",
            "shortname":"x_1",
            "lowerbound":0.2,
            "upperbound":15,
            "initialvalue":None
        },
        {
            "longname":"Decision variable 2",
            "shortname":"x_2",
            "lowerbound":0,
            "upperbound":20,
            "initialvalue":None
        },
        {
            "shortname":"x_3",
            "lowerbound":0,
            "upperbound":40,
            "initialvalue":None
        }
    ],
    "extra_func":[
        {
            "longname":"minimize structural volume",
            "shortname":"g_1",
            "func":
            [
                "Add", 
                [
                  "Divide", 
                  ["Multiply", -7.735, ["Square", "x_1"]], 
                  ["Square", "x_2"]
                ], 
                ["Multiply", "x_1", "x_3"], 
                -180
            ]
        },
        {
            "longname":"minimize structural volume",
            "shortname":"g_2",
            "func":
            [
                "Add", 
                ["Divide", ["Negate", "x_3"], "x_2"], 
                4
            ]
        }
    ],
    "objectives":[  
        {
            "longname":"minimize structural volume",
            "shortname":"f_1",
            "func":
            [
                "Add", 
                ["Multiply", 29.4, "x_1"], 
                ["Multiply", 0.6, "x_2", "x_3"]
            ],
            "max": False,
            "lowerbound":None,
            "upperbound":None
        },
        {
            "longname":"minimize the joint displacement",
            "shortname":"f_2",
            "func":["Sum", ["Max", "g_i", 0], ["Triple", ["Hold", "i"], 1, 2]],
            "max": False,   
            "lowerbound":None,
            "upperbound":None
        }
    ],
    "constraints":[
        {
            "longname":"minimize structural volume",
            "shortname":"g_1",
            "func":
            [
                "Add", 
                [
                  "Divide", 
                  ["Multiply", -7.735, ["Square", "x_1"]], 
                  "x_2"
                ], 
                ["Multiply", "x_1", "x_3"], 
                -180
            ]
        },
        {
            "longname":"minimize structural volume",
            "shortname":"g_2",
            "func":
            [
                "Add", 
                ["Divide", ["Negate", "x_3"], "x_2"], 
                4
            ]
        }

    ],
    "__problemName":"Pressure vessel design problem",
    "__problemDescription":"This problem is from DESDEO example Engineering real-world test problems on https://desdeo-problem.readthedocs.io/en/latest/problems/engineering_real_world.html#re-21-four-bar-truss-design-problem"
}

real_example23 = {
  "constants":[],
  "variables":[
      {
          "longname":"Decision variable 1",
          "shortname":"x_1",
          "lowerbound":1,
          "upperbound":100,
          "initialvalue":None
      },
      {
          "longname":"Decision variable 2",
          "shortname":"x_2",
          "lowerbound":1,
          "upperbound":100,
          "initialvalue":None
      },
      {
          "shortname":"x_3",
          "lowerbound":10,
          "upperbound":200,
          "initialvalue":None
      },
      {
          "shortname":"x_4",
          "lowerbound":10,
          "upperbound":240,
          "initialvalue":None
      }
  ],
  "extra_func":[
      {
          "longname":"minimize structural volume",
          "shortname":"g_1",
          "func":
          ["Add", ["Multiply", -0.0193, "x_3"], "x_1"]
      },
      {
          "longname":"minimize structural volume",
          "shortname":"g_2",
          "func":
          ["Add", ["Multiply", -0.00954, "x_3"], "x_2"]
      },
      {
          "longname":"minimize structural volume",
          "shortname":"g_3",
          "func":
          [
              "Add", 
              ["Multiply", 2.51327, ["Power", "x_3", 3]], 
              ["Multiply", 3.14159, "x_4", ["Square", "x_3"]], 
              -1296000
            ]
      }
  ],
  "objectives":[  
      {
          "longname":"minimize structural volume",
          "shortname":"f_1",
          "func":
          [
              "Add", 
              ["Multiply", 0.6224, "x_1", "x_3", "x_4"], 
              ["Multiply", 1.7781, "x_2", ["Square", "x_3"]], 
              ["Multiply", 19.84, "x_3", ["Square", "x_1"]], 
              ["Multiply", 3.1661, "x_4", ["Square", "x_1"]]
            ],
          "max": False,
          "lowerbound":None,
          "upperbound":None
      },
      {
          "longname":"minimize the joint displacement",
          "shortname":"f_2",
          "func":["Sum", ["Max", "g_i", 0], ["Triple", ["Hold", "i"], 1, 3]],
          "max": False,   
          "lowerbound":None,
          "upperbound":None
      }
  ],
  "constraints":[
      {
          "longname":"minimize structural volume",
          "shortname":"g_1",
          "func":
          ["Add", ["Multiply", -0.0193, "x_3"], "x_1"]
      },
      {
          "longname":"minimize structural volume",
          "shortname":"g_2",
          "func":
          ["Add", ["Multiply", -0.00954, "x_3"], "x_2"]
      },
      {
          "longname":"minimize structural volume",
          "shortname":"g_3",
          "func":
          [
              "Add", 
              ["Multiply", 2.51327, ["Power", "x_3", 3]], 
              ["Multiply", 3.14159, "x_4", ["Square", "x_3"]], 
              -1296000
          ]
      }

  ],
  "__problemName":"Pressure vessel design problem",
  "__problemDescription":"This problem is from DESDEO example Engineering real-world test problems on https://desdeo-problem.readthedocs.io/en/latest/problems/engineering_real_world.html#re-21-four-bar-truss-design-problem"
}

real_example24 = {
    "constants":[
        {
            "longname":"sigma_b_max",
            "shortname":"sigma_b_max",
            "value":700
        },
        {
            "longname":"omega_max",
            "shortname":"omega_max",
            "value":1.5
        },
        {
            "shortname":"tau_max",
            "value":450
        },
        {
            "shortname":"omega",
            "value":56.2e4
        },
        {
            "shortname":"E",
            "value":7e5
        }
    ],
    "variables":[
        {
            "longname":"Decision variable 1",
            "shortname":"x_1",
            "lowerbound":0.5,
            "upperbound":4,
            "initialvalue":None
        },
        {
            "longname":"Decision variable 2",
            "shortname":"x_2",
            "lowerbound":4,
            "upperbound":40,
            "initialvalue":None
        }
    ],
    "extra_func":[
        {
            "longname":"minimize structural volume",
            "shortname":"g_1",
            "func":
            [
                "Add", 
                [
                  "Divide", 
                  ["Rational", -45, 7], 
                  ["Multiply", "x_1", "x_2"]
                ], 
                1
              ]
        },
        {
            "longname":"minimize structural volume",
            "shortname":"g_2",
            "func":
            [
                "Add", 
                ["Divide", ["Rational", -1, 4], "x_2"], 
                1
              ]
        },
        {
            "longname":"minimize structural volume",
            "shortname":"g_3",
            "func":
            [
                "Add", 
                [
                  "Divide", 
                  ["Rational", -281, 350], 
                  ["Multiply", 1.5, "x_1", ["Square", "x_2"]]
                ], 
                1
              ]
        },
        {
            "shortname":"g_4",
            "func":
            [
                "Add", 
                [
                  "Divide", 
                  ["Rational", -9, 14], 
                  ["Multiply", "x_2", ["Power", "x_1", 3]]
                ], 
                1
              ]
        }
    ],
    "objectives":[  
        {
            "longname":"minimize structural volume",
            "shortname":"f_1",
            "func":["Add", ["Multiply", 120, "x_2"], "x_1"],
            "max": False,
            "lowerbound":None,
            "upperbound":None
        },
        {
            "longname":"minimize the joint displacement",
            "shortname":"f_2",
            "func":["Sum", ["Max", "g_i", 0], ["Triple", ["Hold", "i"], 1, 4]],
            "max": False,   
            "lowerbound":None,
            "upperbound":None
        }
    ],
    "constraints":[
        {
            "longname":"minimize structural volume",
            "shortname":"g_1",
            "func":
            [
                "Add", 
                [
                  "Divide", 
                  ["Rational", -45, 7], 
                  ["Multiply", "x_1", "x_2"]
                ], 
                1
              ]
        },
        {
            "longname":"minimize structural volume",
            "shortname":"g_2",
            "func":
            [
                "Add", 
                ["Divide", ["Rational", -1, 4], "x_2"], 
                1
              ]
        },
        {
            "longname":"minimize structural volume",
            "shortname":"g_3",
            "func":
            [
                "Add", 
                [
                  "Divide", 
                  ["Rational", -281, 350], 
                  ["Multiply", 1.5, "x_1", ["Square", "x_2"]]
                ], 
                1
              ]
        },
        {
            "shortname":"g_4",
            "func":
            [
                "Add", 
                [
                  "Divide", 
                  ["Rational", -9, 14], 
                  ["Multiply", "x_2", ["Power", "x_1", 3]]
                ], 
                1
              ]
        }

    ],
    "__problemName":"Pressure vessel design problem",
    "__problemDescription":"This problem is from DESDEO example Engineering real-world test problems on https://desdeo-problem.readthedocs.io/en/latest/problems/engineering_real_world.html#re-21-four-bar-truss-design-problem"
}

analytical_problem = {
    "constants":[],
    "variables":[  
        {
            "longname":"Constants 1",
            "shortname":"x1",
            "lowerbound":-0.5,
            "upperbound":0.5,
            "initialvalue":0
        },
        {
            "longname":"Constants 2",
            "shortname":"x2",
            "lowerbound":-0.5,
            "upperbound":0.5,
            "initialvalue":0
        }
    ],
    "extra_func":[],
    "objectives":[
        { 
            "longname":"objectives 1",
            "shortname":"f_1",
            "func":["Subtract",["Square","x1"],"x2"],
            "max": False,
            "lowerbound":0,
            "upperbound":10
        },
        {
            "longname":"objectives 2",
            "shortname":"f_2",
            "func":["Subtract",["Square","x2"],["Multiply",3,"x1"]],
            "max": False,
            "lowerbound":0,
            "upperbound":10
        }
    ],
    "constraints":[ 
        {
            "longname":"constraints 2",
            "shortname":"g1",
            "func":["Subtract",10,["Add","x1","x2"]]
        }
    ],
    "__ProblemName":"name",
    "__ProblemDescription":"This problem is from DESDEO example Analytical problem on https://desdeo-problem.readthedocs.io/en/latest/notebooks/analytical_problem.html"
}


#START TESTING
#python -m pytest tests/test_MathJsonMOProblem.py
def test_real_example21():
    p = MathJsonMOProblem(real_example21)
    # Variable values
    xs = np.array([2, 2, 2, 2])
    objective_vectors = p.evaluate(xs).objectives
    assert objective_vectors.shape[0] == 1
    expected_result = np.array([[2048.528137, 0.02]])
    npt.assert_allclose(objective_vectors, expected_result)

def test_real_example22():
    p = MathJsonMOProblem(real_example22)
    # Variable values
    xs = np.array([[10., 10., 20.], [12., 10., 20.], [11.5, 15, 35]])
    objective_vectors = p.evaluate(xs).objectives
    assert objective_vectors.shape[0] == 3
    expected_result = np.array([[414, 14.265     ],
                                [472.8,50.8616    ],
                                [653.1,219.62020556]])
    npt.assert_allclose(objective_vectors, expected_result)