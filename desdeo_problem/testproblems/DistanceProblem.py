# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 00:49:27 2021

@author: Golman Rahmanifar
"""

def distance_problem_generator(num_objectives: int, num_dimensions: int=2):
        """
    

    Parameters
    ----------
    num_objectives : int
        DESCRIPTION.
    num_dimensions : int, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    problem
        DESCRIPTION.

    """
        import math
        import random
        import numpy as np
        distance_problem_parameters={"distance_vectors":None,
                                     "rotations":None,
                                     "centre_list":None,
                                     "radii":None,
                                     "base_radius":None,}
        number_of_centres=1
        radius = 1.0/(math.ceil(math.sqrt(number_of_centres))+2)
        distance_problem_parameters['rotations'] = random.random()*2*math.pi
        radius = radius/2;
        #r = (rand(1,2)*2)-1;
        r=(np.random.rand(1, 2)*2)-1
        distance_problem_parameters['center_list']=r
        distance_problem_parameters['radii'] = radius
        distance_problem_parameters['base_radius'] = radius
        angles = np.random.rand(num_objectives,1)*2*math.pi #arbitrary angles for points on circulference of region
        distance_problem_parameters["distance_vectors"]=list(range(len(angles)))
        for i in range(len(angles)):
            distance_problem_parameters['distance_vectors'][i] = np.add(distance_problem_parameters['center_list'],np.dot(distance_problem_parameters['radii'],[math.cos(np.add(distance_problem_parameters['rotations'],angles[i])),math.sin(np.add(distance_problem_parameters['rotations'],angles[i]))]))
                                                            

        return distance_problem_parameters
        
   
    
