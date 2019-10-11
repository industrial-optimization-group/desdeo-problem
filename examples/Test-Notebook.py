# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'examples'))
	print(os.getcwd())
except:
	pass
#%%
import pandas as pd
import numpy as np
#from sklearn.gaussian_process import GaussianProcessRegressor as GPR


#%%
from desdeo_problem.Objective import ScalarDataObjective as sdo
from desdeo_problem.Objective import VectorDataObjective as vdo
from desdeo_problem.surrogatemodels.SurrogateModels import GaussianProcessRegressor
from desdeo_problem.Problem import DataProblem


#%%
data = np.random.rand(100,4)
x = ['a','b']
y = ['f1','f2']
datapd = pd.DataFrame(data, columns=x+y)


#%%
obj = vdo(data=datapd, name=y)


#%%
obj.train(models=GaussianProcessRegressor)


#%%
print(obj.evaluate(np.asarray([[10,1]]), use_surrogate=True))


#%%
obj._model_trained


#%%
obj._model_trained.values()


#%%
prob = DataProblem(data=datapd, objective_names=y, variable_names=x)


#%%
prob.train(GaussianProcessRegressor())


#%%
print(prob.evaluate(np.asarray([[0.1,0.8], [0.5,0.3]])))


#%%



