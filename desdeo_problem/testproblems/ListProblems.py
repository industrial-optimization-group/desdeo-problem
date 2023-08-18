from desdeo_problem.testproblems.CarSideImpact import car_side_impact

from desdeo_problem.testproblems.DBMOPP.DBMOPP_generator import DBMOPP_generator
from desdeo_problem.testproblems import dummy_problem
from desdeo_problem.testproblems.EngineeringRealWorld import *
from desdeo_problem.testproblems.GAA import gaa
from desdeo_problem.testproblems.MultipleClutchBrakes import multiple_clutch_brakes
from desdeo_problem.testproblems.RiverPollution import river_pollution_problem
from desdeo_problem.testproblems.TestProblems import test_problem_builder
from desdeo_problem.testproblems.VehicleCrashworthiness import vehicle_crashworthiness

class ProblemDict:
    def __init__(self):
        self.dic = {
            #CAR SIDE IMPACT
            "car_side_impact":car_side_impact,
            
            #DBMOOP GENERATOR
            "dummy_problem":dummy_problem,

            #REALWOLRD ENINEERING PROBLEM
            "re21":re21,
            "re22":re22,
            "re23":re23,
            "re24":re24,
            "re25":re25,

            "re31":re31,
            "re32":re32,
            "re33":re33,
            "re34":re34,
            "re35":re35,
            "re36":re36,
            "re37":re37,

            "re41":re41,
            "re42":re42,

            "re61":re61,
            "re91":re91,

            "cre21":cre21,
            "cre22":cre22,
            "cre23":cre23,
            "cre24":cre24,
            "cre25":cre25,

            "cre31":cre31,
            "cre32":cre32,

            "cre51":cre51,

            "gaa":gaa,
            "multiple_clutch_brakes":multiple_clutch_brakes,
            "river_pollution_problem":river_pollution_problem,

            "ZDT1": self.zdt1,
            "ZDT2": self.zdt2,
            "ZDT3": self.zdt3,
            "ZDT4": self.zdt4,
            "ZDT5": self.zdt5,
            "ZDT6": self.zdt6,
            "DTLZ1":self.dtlz1,
            "DTLZ2":self.dtlz2, 
            "DTLZ3":self.dtlz3,
            "DTLZ4":self.dtlz4, 
            "DTLZ5":self.dtlz5,
            "DTLZ6":self.dtlz6, 
            "DTLZ7":self.dtlz7, 

            "vehicle_crashworthiness":vehicle_crashworthiness,
        }
    def zdt1():
        return test_problem_builder("ZDT1")
    def zdt2():
        return test_problem_builder("ZDT2")
    def zdt3():
        return test_problem_builder("ZDT3")
    def zdt4():
        return test_problem_builder("ZDT4")
    def zdt5():
        return test_problem_builder("ZDT5")
    def zdt6():
        return test_problem_builder("ZDT6")
    def dtlz1():
        return test_problem_builder("DTLZ1")
    def dtlz2():
        return test_problem_builder("DTLZ2")
    def dtlz3():
        return test_problem_builder("DTLZ3")
    def dtlz4():
        return test_problem_builder("DTLZ4")
    def dtlz5():
        return test_problem_builder("DTLZ5")
    def dtlz6():
        return test_problem_builder("DTLZ6")
    def dtlz7():
        return test_problem_builder("DTLZ7")

    def get_problem(self,name:str):
        return self.dic[name]
    
    def list_problems(self):
        for key, value in self.dic.items() :
            print (key, value)

