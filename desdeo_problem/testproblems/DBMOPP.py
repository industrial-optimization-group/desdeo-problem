from desdeo_problem.testproblems.utilities import get_2D_version, euclidean_distance, convhull, in_hull, get_random_angles, between_lines_rooted_at_pivot, assign_design_dimension_projection
from typing import Dict, Tuple
import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from numpy import matlib # i guess we could implement repmat ourselves
from desdeo_problem.problem import MOProblem, ScalarObjective, variable_builder, ScalarConstraint, VectorObjective
from matplotlib import cm
from desdeo_problem.testproblems.Region import AttractorRegion, Attractor, Region



class DBMOPPobject:
    """
       Object that holds the problem state and information 
    """
    def __init__(self):
        self.rescaleConstant = 0 
        self.rescaleMultiplier = 1 
        self.pi1 = None
        self.pi2 = None
       
        self.pareto_set_indices = 0
        self.pareto_angles = None
        self.rotations = None

        self.attractors = []  # Array of attractors
        self.attractor_regions = [] # array of attractorRegions 
        self.centre_regions = None
        self.neutral_regions = None
        self.neutral_region_objective_values = np.sqrt(8)

        self.hard_constraint_regions = None
        self.soft_constraint_regions = None
        self.discontinuous_regions = None
        self.discontinuous_region_objective_value_offset = None

        self.pivot_locations = None
        self.bracketing_locations_lower = None
        self.bracketing_locations_upper = None


class DBMOPP:
    """
        DBMOPP-class has all the necessary functions and methods to create different problems.

    Args:
        k (int): Number of objectives
        n (int): Number of variables
        nlp (int): Number of local pareto sets
        ndr (int): Number of dominance resistance regions
        ngp: (int): Number of global Pareto sets
        prop_constraint_checker (float): Proportion of constrained 2D space if checker type is used
        pareto_set_type (int): A set type for global Pareto set. Should be one of these
            0: duplicate performance, 1: partially overlapping performance,
            or 2: non-intersecting performance
        constraint_type (int): A constraint type. Should be one of these
            0: No constraint, 1-4: Hard vertex, centre, moat, extended checker, 
            5-8: soft vertex, centre, moat, extended checker.
        ndo (int): Number of regions to apply whose cause discontinuities in objective functions. Defaults to 0
        vary_sol_density (bool): Should solution density vary in maping down to each of the two visualized dimensions.
            Default to False
        vary_objective_scales (bool): Are objective scale varied. Defaults to False
        prop_neutral (float): Proportion of neutral space. Defaults to 0
        nm (int): Number of samples used for approximation checker and neutral space coverage. Defaults to 10000

    Raises:
        Argument was invalid
    """
    def __init__(
        self,
        k: int,
        n: int,
        nlp: int,
        ndr: int,
        ngp: int,
        prop_constraint_checker: float,
        pareto_set_type: int,
        constraint_type: int,
        ndo: int = 0,
        vary_sol_density: bool = False,
        vary_objective_scales: bool = False,
        prop_neutral: float = 0,
        nm: int = 10000,
    ) -> None:
        msg = self._validate_args(k, n, nlp, ndr, ngp, prop_constraint_checker, pareto_set_type, constraint_type,
            ndo, prop_neutral, nm)
        if msg != "":
            raise Exception(msg)
        self.k = k
        self.n = n
        self.nlp = nlp
        self.ndr = ndr
        self.ngp = ngp
        self.prop_contraint_checker = prop_constraint_checker
        self.pareto_set_type = pareto_set_type
        self.constraint_type = constraint_type
        self.ndo = ndo
        self.vary_sol_density = vary_sol_density
        self.vary_objective_scales = vary_objective_scales
        self.prop_neutral = prop_neutral
        self.nm = nm

        self.obj = DBMOPPobject() # The obj in the matlab implementation

        self.initialize()


    def _print_params(self):
        print("n_obj: ", self.k)
        print("n_var: ", self.n)
        print("n_nlp: ", self.nlp)
        print("n_ndr: ", self.ndr)
        print("n_ngp: ", self.ngp)
        print("potype: ", self.pareto_set_type)
        print("const type: ", self.constraint_type)


    def _validate_args(
        self,
        k: int,
        n: int,
        nlp: int,
        ndr: int,
        ngp: int,
        prop_constraint_checker: float,
        pareto_set_type: str,
        constraint_type: str,
        ndo: int,
        prop_neutral: float,
        nm: int
    ) -> None:
        """
        Validate arguments given to the constructor of the class. 

        Args:
            See __init__
        
        Returns:
            str: A error message which contains everything wrong with the arguments. Empty string if arguments are valid
        """
        msg = ""
        if k < 1:
            msg += f"Number of objectives should be greater than zero, was {k}.\n"
        if n < 2:
            msg += f"Number of variables should be greater than two, was {n}.\n"
        if nlp < 0: 
            msg += f"Number of local Pareto sets should be greater than or equal to zero, was {nlp}.\n"
        if ndr < 0: 
            msg += f"Number of dominance resistance regions should be greater than or equal to zero, was {ndr}.\n"
        if ngp < 1:
            msg += f"Number of global Pareto sets should be greater than one, was {ngp}.\n"
        if not 0 <= prop_constraint_checker <= 1:
            msg += f"Proportion of constrained 2D space should be between zero and one, was {prop_constraint_checker}.\n"
        if pareto_set_type not in np.arange(3):
            msg += f"Global pareto set type should be a integer number between 0 and 2, was {pareto_set_type}.\n"
        if pareto_set_type == 1 and ngp <= 1:
            msg += f"Number of global pareto sets needs to be more than one, if using disconnected pareto set type"
        if constraint_type not in np.arange(9):
            msg += f"Constraint type should be a integer number between 0 and 8, was {constraint_type}.\n"
        if constraint_type not in [4,8] and prop_constraint_checker != 0: 
            msg += f"Proporortion of constrained space checker should be 0 if constraint type is not 4 or 8, was constraint type {constraint_type} and prop {prop_constraint_checker}"
        if ndo < 0:
            msg += f"Number of discontinuous objective function regions should be greater than or equal to zero, was {ndo}.\n"
        if not 0 <= prop_neutral <= 1:
            msg += f"Proportion of neutral space should be between zero and one, was {prop_neutral}.\n"
        if nm < 1000:
            msg += f"Number of samples should be at least 1000, was {nm}.\n"
        return msg



    def initialize(self):
        #place attractor centres for regions defining attractor points
        self.set_up_attractor_centres()
        #set up angles for attractors on regin cicumferences and arbitrary rotations for regions
        self.obj.pareto_angles = get_random_angles(self.k) # arbitrary angles for Pareto set
        self.obj.rotations = get_random_angles(len(self.obj.centre_regions))
        # now place attractors
        self.place_attractors()
        
        if self.pareto_set_type != 0:
            self.place_disconnected_pareto_elements()
        self.place_discontinunities_neutral_and_checker_constraints()

        # set the neutral value to be the same in all neutral locations
        self.obj.neutral_region_objective_values = np.ones((1,self.k))*self.obj.neutral_region_objective_values; # CHECK
        self.place_vertex_constraint_locations()
        self.place_centre_constraint_locations()
        self.place_moat_constraint_locations()
        self.obj.pi1, self.obj.pi2 = assign_design_dimension_projection(self.n, self.vary_sol_density)


    def generate_problem(self):
        """
        Generate the test problem to use in DESDEO.

        Returns:
            MOProblem: A test problem
        """

        # objectives = [ScalarObjective(f"objective{i}", lambda x: self.evaluate(x)['obj_vector'][i]) for i in range(self.k)] # this is probably the problem.
        obj_names = ["f" + str(i + 1) for i in range(self.k)]
        #var_names = [f'x{i}' for i in range(self.n)]
        var_names = ["x" + str(i + 1) for i in range(self.n)]
        initial_values = (np.random.rand(self.n,1) * 2) - 1
        lower_bounds = np.ones(self.n) * -1
        upper_bounds = np.ones(self.n)
        variables = variable_builder(var_names, initial_values, lower_bounds, upper_bounds)

        cs = lambda x, _y: self.evaluate(x)['soft_constr_viol'] * -1
        ch = lambda x, _y: self.evaluate(x)['hard_constr_viol'] * -1

        constraints = [
            ScalarConstraint("hard constraint", self.n, self.k, ch),
            ScalarConstraint("soft constraint", self.n, self.k, cs)
        ]
        #objective = VectorObjective(name=)

        def modified_obj_func(x):
            if isinstance(x, list):
                if len(x) == n_of_variables:
                    return [obj_func(x)]
                elif len(x[0]) == n_of_variables:
                    return list(map(obj_func, x))
            else:
                if x.ndim == 1:
                    return [obj_func(x)]
                elif x.ndim == 2:
                    return list(map(obj_func, x))
            raise TypeError("Unforseen problem, contact developer")


        objective = VectorObjective(name=obj_names, evaluator=modified_obj_func)
        return MOProblem([objective], variables, None)  


    def is_pareto_set_member(self, z):
        self.check_valid_length(z)
        x = get_2D_version(z, self.obj.pi1, self.obj.pi2)
        return self.is_pareto_2D(x)


    def evaluate(self, x):
        x = np.atleast_2d(x)
        self.check_valid_length(x)
        z = get_2D_version(x, self.obj.pi1, self.obj.pi2)
        return self.evaluate_2D(z)
    

    def evaluate_2D(self, x) -> Dict:
        """
        Evaluate x in problem instance in 2 dimensions
        
        Args:
            x (np.ndarray): The decision vector to be evaluated
        
        Returns:
            Dict: A dictionary object with the following entries:
                'obj_vector' : np.ndarray, the objective vector
                'soft_constr_viol' : boolean, soft constraint violation
                'hard_constr_viol' : boolean, hard constraint violation
        """

        ans = {
            "obj_vector": np.array([None] * self.k),
            "soft_constr_viol": False,
            "hard_constr_viol": self.get_hard_constraint_violation(x),
        }
        if ans["hard_constr_viol"]:
            if self.constraint_type == 3:
                if self.in_convex_hull_of_attractor_region(x):
                    ans["hard_constr_viol"] = False
                    ans["obj_vector"] = self.get_objectives(x)
            return ans
        
        ans["soft_constr_viol"] =  self.get_soft_constraint_violation(x)
        if ans["soft_constr_viol"]:
            if (self.constraint_type == 7):
                if (self.in_convex_hull_of_attractor_region(x)):
                    ans["soft_constr_viol"] = False
                    ans["obj_vector"] = self.get_objectives(x)
            return ans
        
        # Neither constraint breached
        if self.check_neutral_regions(x):
            ans["obj_vector"] = self.obj.neutral_region_objective_values
        else: 
            ans["obj_vector"] = self.get_objectives(x)
        return ans


    def is_pareto_2D(self, x: np.ndarray):
        """
        
        """
        if self.get_hard_constraint_violation(x):
            return False
        if self.get_soft_constraint_violation(x):
            return False
        return self.is_in_limited_region(x)["in_pareto_region"]


    def in_convex_hull_of_attractor_region(self, y: np.ndarray):
        """
            # Attractor region method? 
        """
        self.check_valid_length(y)
        x = get_2D_version(y, self.obj.pi1, self.obj.pi2)
        
        for i, centre_region in enumerate(self.obj.centre_regions):
            if centre_region.is_inside(x):
                return self.obj.attractor_regions[i].in_hull(x)

        return False
    
    
    def check_valid_length(self, x):
        x = np.atleast_2d(x)
        if (x.shape[1] != self.n): 
            msg = f"Number of design variables in the argument does not match that required in the problem instance, was {x.shape[1]}, should be {self.n}"
            raise Exception(msg)


    def set_up_attractor_centres(self):
        """
        Calculate max maximum region radius given problem properties
        """
        # number of local PO sets, global PO sets, dominance resistance regions
        n = self.nlp + self.ngp + self.ndr 

        #Create the attractor objects
        self.obj.centre_regions = np.array([Region() for _ in range(n)]) # Different objects

        max_radius = 1/(2*np.sqrt(n)+1) * (1 - (self.prop_neutral + self.prop_contraint_checker)) # prop 0 and 0.
        
        # Assign centres
        radius = self.place_region_centres(n, max_radius)

        # Assign radius
        self.place_region_radius(n, radius)
        
        # save indices of PO set locations
        self.obj.pareto_set_indices = np.arange(self.ngp+1, self.nlp + self.ngp + 1)
    

    def place_region_radius(self, n, r):
        for i in range(n):
            self.obj.centre_regions[i].radius = r

        # reduce raddius if local fronts used
        if self.nlp > 0:
            for i in range(self.nlp + 1, n):
                self.obj.centre_regions[i].radius = r / 2

            w = np.linspace(1, 0.5, self.nlp+1)

            # linearly decrease local front radius
            for i in range(self.nlp+1):
                self.obj.centre_regions[i].radius = self.obj.centre_regions[i].radius * w[i]


    def place_region_centres(self, n: int, r: float):
        effective_bound = 1 - r
        threshold = 4*r

        time_start = time()
        too_long = False
        max_elapsed = 1 
        rand_coord = (np.random.rand(2)*2*effective_bound) - effective_bound
        self.obj.centre_regions[0].centre = rand_coord

        for i in np.arange(1, n): # looping the objects would be nicer
            while True:
                rand_coord = (np.random.rand(2)*2*effective_bound) - effective_bound
                distances = np.array([self.obj.centre_regions[i].get_distance(rand_coord) for i in range(i)])
                t = np.min(distances)
                if t > threshold:
                    print("assigned centre", i)
                    break
                too_long = (time() - time_start) > max_elapsed
                if (too_long): break
            self.obj.centre_regions[i].centre = rand_coord

        if (too_long): # Took longer than max_elapsed. 
            print('restarting attractor region placement with smaller radius...\n')
            return self.place_region_centres(n, r*0.95)

        return r


    def place_attractors(self):
        """
            Randomly place attractor regions in 2D space
        """
        l = self.nlp + self.ngp
        ini_locs = np.zeros((l, 2, self.k))

        self.obj.attractor_regions = np.array([None] * (l + self.ndr))

        for i in np.arange(l):
            B = np.hstack((
                np.cos(self.obj.pareto_angles + self.obj.rotations[i]),
                np.sin(self.obj.pareto_angles + self.obj.rotations[i])
            ))

            locs = (
                matlib.repmat(self.obj.centre_regions[i].centre, self.k, 1) + 
                (matlib.repmat(self.obj.centre_regions[i].radius, self.k, 2) * B)
            )

            # create attractor region
            self.obj.attractor_regions[i] = AttractorRegion(
                locations = locs, 
                indices = np.arange(self.k),
                centre = self.obj.centre_regions[i].centre,
                radius = self.obj.centre_regions[i].radius,
                convhull = convhull(locs)
            )

            for k in np.arange(self.k):
                ini_locs[i,:,k] = locs[k,:]
            
        self.obj.attractors = np.array([Attractor() for _ in range(self.k)])

        for i in range(self.k):
            self.obj.attractors[i].locations = ini_locs[:,:,i]

        for i in range(l, l + self.ndr):
            locs = (
                matlib.repmat(self.obj.centre_regions[i].centre, self.k,1) 
                + (matlib.repmat(self.obj.centre_regions[i].radius, self.k, 2)
                    * np.hstack((
                        np.cos(self.obj.pareto_angles + self.obj.rotations[i]),
                        np.sin(self.obj.pareto_angles + self.obj.rotations[i])
                    ))
                )
            )

            n_include = np.random.permutation(self.k - 1) + 1 # Plus one as we want to include at least one?
            n_include = n_include[0] # Take the first one
            I = np.argsort(np.random.rand(self.k))
            j = I[:n_include]
            self.obj.attractor_regions[i] = AttractorRegion(
                locations = locs[j, :], 
                indices = j,
                centre = None, 
                radius = self.obj.centre_regions[i].radius,
                convhull = convhull(locs[j, :])
            )
   
            for k in range(n_include):
                attractor_loc = self.obj.attractors[k].locations
                self.obj.attractors[k].locations = np.vstack((attractor_loc,  locs[I[k], :]))
        
     

    def place_disconnected_pareto_elements(self):
        n = self.ngp - 1
        pivot_index = np.random.randint(self.k)

        # sort from smallest to largest and get the indices
        indices = np.argsort(self.obj.pareto_angles, axis = 0)

        offset_angle_1 = (self.obj.pareto_angles[indices[self.k - 1]] if pivot_index == 0
            else self.obj.pareto_angles[indices[pivot_index-1]]) # check this minus
        
        offset_angle_2 = (self.obj.pareto_angles[indices[0]] if pivot_index == self.k-1
            else self.obj.pareto_angles[indices[pivot_index + 1]]) # check plus
        
        pivot_angle = self.obj.pareto_angles[indices[pivot_index]]

        if pivot_angle == (offset_angle_1 or offset_angle_2):
            raise Exception("Angle should not be duplicated!")
        
        if offset_angle_1 < offset_angle_2:
            range_covered = offset_angle_1 + 2 * np.pi - offset_angle_2
            p1 = offset_angle_1 / range_covered
            r = np.random.rand(n)
            p1 = np.sum(r < p1)
            r[:p1] = 2*np.pi + np.random.rand(p1) * offset_angle_1
            r[p1:n] = np.random.rand(n-p1) * (2*np.pi - offset_angle_2) + offset_angle_2
            r = np.sort(r)
            r_angles = np.zeros(n+2)
            r_angles[0] = offset_angle_2
            r_angles[n+1] = offset_angle_1
            r_angles[1:n+1] = r
        else:
            r = r = np.random.rand(n)
            r = np.sort(r)
            r_angles = np.zeros(n+2)
            r_angles[0] = offset_angle_2 
            r_angles[n+1] = offset_angle_1
            r_angles[1:n+1] = r 

        k = self.nlp + self.ngp
        self.obj.pivot_locations = np.zeros((k, 2)) 
        self.obj.bracketing_locations_lower = np.zeros((k,2))
        self.obj.bracketing_locations_upper = np.zeros((k,2))

        def calc_location(ind, a):
            return self.obj.centre_regions[ind].calc_location(a, self.obj.rotations[ind])

        index = 0
        for i in range(self.nlp, self.nlp + self.ngp): # verify indexing
            self.obj.pivot_locations[i,:] = calc_location(i, pivot_angle)
            
            self.obj.bracketing_locations_lower[i,:] = calc_location(i, r_angles[index])

            if self.pareto_set_type == 0:
                raise Exception('should not be calling this method with an instance with identical Pareto set regions')
            
            elif self.pareto_set_type == 2:
                self.obj.bracketing_locations_upper[i,:] = calc_location(i, r_angles[index+1])

            elif self.pareto_set_type == 1:
                if index == self.ngp - 1:
                    self.obj.bracketing_locations_lower[i,:] = calc_location(i, r_angles[2]) # with some input this: IndexError: index 2 is out of bounds for axis 0 with size 2
                    self.obj.bracketing_locations_upper[i,:] = calc_location(i, r_angles[n])
                else:
                    self.obj.bracketing_locations_upper[i,:] = calc_location(i, r_angles[index+2])
            index += 1
                    

    def place_vertex_constraint_locations(self):
        """
        Place constraints located at attractor points
        """
        print('Assigning any vertex soft/hard constraint regions\n')
        if (self.constraint_type in [1,4]):
            to_place = 0
            for i in range(len(self.obj.attractors)): # or self.k as that should be the same...
                to_place += len(self.obj.attractors[i].locations)
            
            centres = np.zeros((to_place, 2))
            radii = np.zeros(to_place)
            k = 0

            penalty_radius = np.random.rand(1) / 2
            for i, attractor_region in enumerate(self.obj.attractor_regions):
                for j in range(len(attractor_region.objective_indices)):
                    centres[k,:] = attractor_region.locations[j,:] # Could make an object here...
                    radii[k] = attractor_region.radius * penalty_radius
                    k += 1
            
            if self.constraint_type == 1:
                self.obj.hard_constraint_regions = np.array([Region() for _ in range(to_place)])
                for i, hard_constraint_region in enumerate(self.obj.hard_constraint_regions):
                    hard_constraint_region.centre = centres[i,:]
                    hard_constraint_region.radius = radii[i]
            else:
                self.obj.soft_constraint_regions = np.array([Region() for _ in range(to_place)])
                for i, soft_constraint_region in enumerate(self.obj.soft_constraint_regions):
                    soft_constraint_region.centre = centres[i,:]
                    soft_constraint_region.radius = radii[i]


    def place_centre_constraint_locations(self):
        """
        Place center constraint regions
        """
        print("Assigning any centre soft/hard constraint regions.\n")
        if self.constraint_type == 2:
            self.obj.hard_constraint_regions = self.obj.centre_regions
        elif self.constraint_type == 5:
            self.obj.soft_constraint_regions = self.obj.centre_regions


    def place_moat_constraint_locations(self):
        """
        Place moat constraint regions
        """
        print('Assigning any moat soft/hard constraint regions\n')
        r = np.random.rand() + 1
        if self.constraint_type == 3:
            self.obj.hard_regions = self.obj.centre_regions
            for i in range(len(self.obj.hard_regions)):
                self.obj.hard_regions[i].radius = self.obj.hard_regions[i].radius * r
        elif self.constraint_type == 6:
            self.obj.soft_regions = self.obj.centre_regions
            for i in range(len(self.obj.soft_regions)):
                self.obj.soft_regions[i].radius = self.obj.soft_regions[i].radius * r


    def place_discontinunities_neutral_and_checker_constraints(self):
        print('Assigning any checker soft/hard constraint regions and neutral regions\n')
        S = (np.random.rand(self.nm, 2) * 2) - 1
        for _i, centre_region in enumerate(self.obj.centre_regions):
            to_remove = centre_region.is_inside(S, True)
            not_to_remove = np.logical_not(to_remove)
            S = S[not_to_remove, :]
        
        if S.shape[0] < self.nm * (self.prop_contraint_checker + self.prop_neutral):
            msg = 'Not enough space outside of attractor regions to match requirement of constrained+neural space'
            raise Exception(msg)
        
        if self.prop_contraint_checker > 0:
            regions, S = self.set_not_attractor_regions_as_proportion_of_space(S, self.prop_contraint_checker, [])
            if self.constraint_type == 4:
                self.obj.hard_constraint_regions = regions
            elif self.constraint_type == 8:
                self.obj.soft_constraint_regions = regions
            else:
                raise Exception(f"constraintType should be 8 or 4 to reach here is {self.constraint_type}")
        
        # Neutral space
        if self.prop_neutral > 0:
            regions, _ = self.set_not_attractor_regions_as_proportion_of_space(S, self.prop_neutral, regions)
            self.obj.neutral_regions = regions

        print("TODO check discontinuity, not done in matlab")


    def set_not_attractor_regions_as_proportion_of_space(self, S, proportion_to_attain, other_regions):
        allocation = 0
        regions = []
        while allocation < proportion_to_attain:
            region = Region()
            region.centre = S[-1, :]
            
            centre_list = np.zeros((len(self.obj.centre_regions), 2))
            centre_radii = np.zeros(len(self.obj.centre_regions))
            for i, centre_region in enumerate(self.obj.centre_regions):
                centre_list[i] = centre_region.centre
                centre_radii[i] = centre_region.radius
            
            other_centres = np.zeros((len(other_regions), 2))
            other_radii = np.zeros(len(other_regions))
    
            for i, other_region in enumerate(other_regions):
                other_centres[i] = other_region.centre
                other_radii[i] = other_region.radius

            both_centres = np.vstack((centre_list, other_centres)) if other_centres.shape[0] > 0 else centre_list

            d = euclidean_distance(both_centres, S[-1, :])
            d = d - np.hstack((centre_radii, other_radii))
            d = np.min(d)

            if d <= 0:
                raise Exception("Should not get here")
            
            c_r = np.sqrt((proportion_to_attain - allocation)/ np.pi)
            r = np.random.rand(1)*np.minimum(d, c_r)
            region.radius = r 
            regions.append(region)
            S = S[:-1, :] # remove last row

            d = euclidean_distance(S, region.centre)
            I = d > r
            covered_count = (I == False).sum() + 1
            S = S[I,:] # Remove covered points

            allocation += covered_count/self.nm

        return np.array(regions) , S

    
    def check_region(self, regions, x, include_boundary):
        if regions is None: return False
        for region in regions:
            if region.is_inside(x, include_boundary):
                return True
        return False

    def check_neutral_regions(self, x):
        return self.check_region(self.obj.neutral_regions, x, True)


    def get_hard_constraint_violation(self, x):
        return self.check_region(self.obj.hard_constraint_regions, x, False)

    def get_soft_constraint_violation(self, x):
        in_soft_constraint_region = self.check_region(self.obj.soft_constraint_regions, x, True)
        # return in_soft_constraint_region
        if in_soft_constraint_region:
            d = np.zeros(len(self.obj.soft_constraint_regions))
            radiis = np.zeros(len(self.obj.soft_constraint_regions))
            for i, soft_constraint_region in enumerate(self.obj.soft_constraint_regions):
                d[i] = soft_constraint_region.get_distance(x)
                radiis[i] = soft_constraint_region.radius
            k = np.sum(d < self.obj.soft_constraint_radii)
            print(k)
            if k > 0:
                c = d - radiis
                c = c * k
                return np.max(c)
        return False

    def get_minimun_distance_to_attractors(self, x: np.ndarray):
        """
        
        """
        y = np.zeros(self.k)
        for i, attractor in enumerate(self.obj.attractors):
            y[i] = attractor.get_minimum_distance(x)
        y *= self.obj.rescaleMultiplier
        y += self.obj.rescaleConstant
        return y

    def get_minimum_distances_to_attractors_overlap_or_discontinuous_form(self, x):
        y = self.get_minimun_distance_to_attractors(x)
        in_pareto_region, in_hull, index  = self.is_in_limited_region(x).values()
        if in_hull:
            if not in_pareto_region:
                y += self.obj.centre_regions[index].radius
        return y
    
    def get_objectives(self, x):
        if (self.pareto_set_type == 0):
            y = self.get_minimun_distance_to_attractors(x)
        else:
            y = self.get_minimum_distances_to_attractors_overlap_or_discontinuous_form(x)
        
        y = self.update_with_discontinuity(x,y)
        y = self.update_with_neutrality(x,y)
        return y

    def is_in_limited_region(self, x, eps = 1e-06):
        """
        
        """
        ans = {
            "in_pareto_region": False,
            "in_hull": False,
            "index": -1
        }

        # can still be improved?
        I = np.array([i for i in range(len(self.obj.centre_regions)) if self.obj.centre_regions[i].is_close(x, eps)])
        
        if len(I) > 0: # is not empty 
            i = I[0]
            if self.nlp <= i < self.nlp + self.ngp:
                if self.constraint_type in [2,6]: 
                    # Smaller of dist
                    dist = self.obj.centre_regions[i].get_distance(x)
                    radius = self.obj.centre_regions[i].radius
                    r = np.min(np.abs(dist), np.abs(radius))
                    if np.abs(dist) - radius < 1e4 * eps * r:
                        ans["in_hull"] = True
                elif in_hull(x, self.obj.attractor_regions[i].locations[self.obj.attractor_regions[i].convhull.simplices]):
                    ans["in_hull"] = True 
        
        if self.pareto_set_type == 0 or self.constraint_type in [2,6]:
            ans["in_pareto_region"] = ans["in_hull"]
            ans["in_hull"] = False
        else:
            if ans["in_hull"]:
                ans["index"] = I[0]
                ans["in_pareto_region"] = between_lines_rooted_at_pivot(
                    x,
                    self.obj.pivot_locations[I[0], :],
                    self.obj.bracketing_locations_lower[I[0],:],
                    self.obj.bracketing_locations_upper[I[0],:],
                )
                if self.pareto_set_type == 1:
                    if I[0] == self.nlp + self.ngp: # should maybe be -1
                        ans["in_pareto_region"] = not ans["in_pareto_region"] # special case where last region is split at the two sides, should not get here everytime

        return ans


    def update_with_discontinuity(self, x, y):
        return self._update(
            self.obj.discontinuous_regions,
            self.obj.discontinuous_region_objective_value_offset,
            x,
            y,    
        )

    def update_with_neutrality(self, x, y):
        return self._update(
            self.obj.neutral_regions,
            self.obj.neutral_region_objective_values,
            x,
            y,    
        )

    def _update(self, regions, offsets, x, y):
        if regions is None: return y
        distances = np.zeros(len(regions))
        for i, region in enumerate(regions):
            distances[i] = region.get_distance(x) if region.is_inside(x, include_boundary = True) else 0
        if np.any(distances > 0):
            index = np.argmin(distances) # molst likely will return the index of the first 0
            y = y + offsets[index,:]
        return y




    # PLOTTING

    def plot_problem_instance(self):
        """
        """
        fig, ax = plt.subplots()
        # Plot local Pareto regions

        plt.xlim([-1, 1])
        plt.ylim([-1,1])
        for i in range(self.nlp):
            self.obj.attractor_regions[i].plot(ax, 'g') # Green
        
        # global pareto regions

        for i in range(self.nlp, self.nlp + self.ngp):
            self.obj.attractor_regions[i].plot(ax, 'r')
            print("the fill here is different than above")
        
        # dominance resistance set regions
        for i in range(self.nlp + self.ngp, self.nlp + self.ngp + self.ndr):
            # attractor regions should take care of different cases
            self.obj.attractor_regions[i].plot(ax, 'b') 

        
        def plot_constraint_regions(constraint_regions, color):
            if constraint_regions is None: return
            for constraint_region in constraint_regions:
                constraint_region.plot(color, ax)

        plot_constraint_regions(self.obj.hard_constraint_regions, 'black')
        plot_constraint_regions(self.obj.soft_constraint_regions, 'grey')
        plot_constraint_regions(self.obj.neutral_regions, 'c')


        # PLOT DISCONNECTED PENALTY
        print("disconnected Pareto penalty regions not yet plotted. THIS IS NOT IMPLEMENTED IN MATLAB")

        #plt.show()

    def plot_landscape_for_single_objective(self, index, res = 500):
        if res < 1:
            raise Exception("Cannot grid the space with a resolution less than 1")
        if index not in np.arange(self.k):
            raise Exception(f"Index should be between 0 and {self.k-1}, was {index}.")

        xy = np.linspace(-1,1, res)
        x, y = np.meshgrid(xy, xy)

        z = np.zeros((res, res))
        for i in range(res):
            for j in range(res):
                decision_vector = np.hstack((xy[i], xy[j]))
                obj_vector = self.evaluate_2D(decision_vector)["obj_vector"]
                obj_vector = np.atleast_2d(obj_vector)
                z[i, j] = obj_vector[0, index]
    
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.view_init(elev=90, azim=-90)

        surf = ax.plot_surface(x, y, z.T, cmap=cm.plasma, linewidth=0, antialiased = False, vmin =  np.nanmin(z), vmax = np.nanmax(z))

        fig.colorbar(surf, shrink=0.5, aspect=5)
        #plt.show()

    def plot_pareto_set_members(self, resolution = 500):
        if resolution < 1: 
            raise Exception("Cannot grid the space with a resolution less than 1")
        fig, ax = plt.subplots()

        plt.xlim([-1,1])
        plt.ylim([-1,1])
        
        xy = np.linspace(-1, 1, resolution)

        for x in xy:
            for y in xy:
                z = np.array([x,y])
                if self.is_pareto_2D(z):
                    ax.scatter(x,y, color='black', s=1)

        #plt.show()
    
    def plot_dominance_landscape(self, res = 500, moore_neighbourhood = True):
        if res < 1: 
            raise Exception("Cannot grid the space with a resolution less than 1")
        
        xy = np.linspace(-1, 1, res)
        y = np.zeros((self.k, res, res))
        for i in range(res):
            for j in range(res):
                decision_vector = np.hstack((xy[i], xy[j]))
                obj_vector = self.evaluate_2D(decision_vector)
                y[:, i, j] = obj_vector

        return self.plot_dominance_landscape_from_matrix(y, xy, xy, moore_neighbourhood)
    
    def plot_dominance_landscape_from_matrix(self, z, x, y, moore_neighbourhood):
        pass

    

if __name__=="__main__":
    import random

    n_objectives = 4
    n_variables = 2 
    n_local_pareto_regions = 3
    n_disconnected_regions = 0 
    n_global_pareto_regions = 1 
    pareto_set_type = 0 
    constraint_type = 0

    problem = DBMOPP(
        n_objectives,
        n_variables,
        n_local_pareto_regions,
        n_disconnected_regions,
        n_global_pareto_regions,
        0,
        pareto_set_type,
        constraint_type, 0, False, False, 0, 10000
    )
    print(problem._print_params())

    print("Initializing works!")
    x = np.random.rand(1, n_variables)
    print(problem.evaluate(x))


    # For desdeos MOProblem only
    moproblem = problem.generate_problem()
    print("\nFormed MOProblem: \n\n", moproblem.evaluate(x)) 

    problem.plot_problem_instance()
    problem.plot_pareto_set_members(150)
    problem.plot_landscape_for_single_objective(0, 100)

    # show all plots
    plt.show()
