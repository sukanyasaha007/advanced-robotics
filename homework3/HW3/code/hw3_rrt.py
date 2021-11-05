'''
CSCI 4302/5302 Advanced Robotics
University of Colorado Boulder
(C) 2021 Prof. Bradley Hayes <bradley.hayes@colorado.edu>
'''

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import pdb


name="Sukanya Saha"
GRAD=True


###############################################################################
## Base Code
###############################################################################
DYNAMICS_MODE = None
class Node:
    """
    Node for RRT/RRT* Algorithm. This is what you'll make your graph with!
    """
    def __init__(self, pt, parent=None):
        self.point = pt # n-Dimensional point
        self.parent = parent # Parent node
        self.path_from_parent = [] # List of points along the way from the parent node (for visualization)
        self.cost = 0

def get_nd_obstacle(state_bounds):
    center_vector = []
    for d in range(state_bounds.shape[0]):
        center_vector.append(state_bounds[d][0] + random.random()*(state_bounds[d][1]-state_bounds[d][0]))
    radius = random.random() * 0.6
    return [np.array(center_vector), radius]

def setup_random_2d_world():
    state_bounds = np.array([[0,10],[0,10]])

    obstacles = [] # [pt, radius] circular obstacles
    for n in range(30):
        obstacles.append(get_nd_obstacle(state_bounds))

    def state_is_valid(state):
        for dim in range(state_bounds.shape[0]):
            if state[dim] < state_bounds[dim][0]: return False
            if state[dim] >= state_bounds[dim][1]: return False
        for obs in obstacles:
            if np.linalg.norm(state - obs[0]) <= obs[1]: return False
        return True

    return state_bounds, obstacles, state_is_valid

def setup_fixed_test_2d_world():
    state_bounds = np.array([[0,1],[0,1]])
    obstacles = [] # [pt, radius] circular obstacles
    obstacles.append([[0.5,0.5],0.2])
    obstacles.append([[0.1,0.7],0.1])
    obstacles.append([[0.7,0.2],0.1])

    def state_is_valid(state):
        for dim in range(state_bounds.shape[0]):
            if state[dim] < state_bounds[dim][0]: return False
            if state[dim] >= state_bounds[dim][1]: return False
        for obs in obstacles:
            if np.linalg.norm(state - obs[0]) <= obs[1]: return False
        return True

    return state_bounds, obstacles, state_is_valid


def plot_circle(x, y, radius, color="-k"):
    deg = np.linspace(0,360,50)

    xl = [x + radius * math.cos(np.deg2rad(d)) for d in deg]
    yl = [y + radius * math.sin(np.deg2rad(d)) for d in deg]
    plt.plot(xl, yl, color)

def visualize_2D_graph(state_bounds, obstacles, nodes, goal_point=None, filename=None):
    '''
    @param state_bounds Array of min/max for each dimension
    @param obstacles Locations and radii of spheroid obstacles
    @param nodes List of vertex locations
    @param edges List of vertex connections
    '''

    fig = plt.figure()
    plt.xlim(state_bounds[0,0], state_bounds[0,1])
    plt.ylim(state_bounds[1,0], state_bounds[1,1])

    for obs in obstacles:
        plot_circle(obs[0][0], obs[0][1], obs[1])

    goal_node = None
    for node in nodes:
        if node.parent is not None:
            node_path = np.array(node.path_from_parent)
            plt.plot(node_path[:,0], node_path[:,1], '-b')
        if goal_point is not None and np.linalg.norm(node.point - np.array(goal_point)) <= 1e-5:
            goal_node = node
            plt.plot(node.point[0], node.point[1], 'k^')
        else:
            plt.plot(node.point[0], node.point[1], 'ro')

    plt.plot(nodes[0].point[0], nodes[0].point[1], 'ko')

    if goal_node is not None:
        cur_node = goal_node
        while cur_node is not None: 
            if cur_node.parent is not None:
                node_path = np.array(cur_node.path_from_parent)
                plt.plot(node_path[:,0], node_path[:,1], '--y')
                cur_node = cur_node.parent
            else:
                break

    if goal_point is not None:
        plt.plot(goal_point[0], goal_point[1], 'gx')


    if filename is not None:
        fig.savefig(filename)
    else:
        plt.show()


def get_random_valid_vertex(state_valid, bounds, obstacles):
    # NOTE: Do not use this function in your implementation directly, but feel free to make your own variant of it. (The grading script will change this)
    vertex = None
    while vertex is None: # Get starting vertex
        pt = np.random.rand(bounds.shape[0]) * (bounds[:,1]-bounds[:,0]) + bounds[:,0]
        if state_valid(pt):
            vertex = pt
    return vertex



def initialize_non_holonomic_actions():
    actions = np.array([[-1.,-1.], [1.,1.], [-1.,1.], [1.,-1.]])
    action_list = list(actions)
    for action in actions: action_list.append(action*0.4*np.random.random() + 0.2) # Macro Actions
    for action in actions: action_list.append(action*0.15*np.random.random()) # Micro Actions
    return action_list

def simulate_non_holonomic_action(state, action):
    '''
    Returns a discretized path along which the agent moves when performing an action in a state.
    '''
    path = np.linspace(state, state+action, 10)
    return path


#####################################
def distance(x1, x2):
    dist = np.sqrt(np.sum(np.square(np.array(x1) - x2)))
    return dist

def get_nearest_vertex(node_list, q_point):
    '''
    @param node_list: List of Node objects
    @param q_point: Query vertex
    @return Node in node_list with closest node.point to query q_point
    '''

    # TODO: Your Code Here
    # Hint: np.linalg.norm returns the length of a vector

    dmin = float("inf")
    nnear = 0
    for node in node_list:
        d = distance(node.point, q_point)
        if d < dmin:
            dmin = d
            nnear = node

    return nnear


def steer(from_point, to_point, delta_q):
    '''
    @param from_point: Point where the path to "to_point" is originating from
    @param to_point: n-Dimensional point (array) indicating destination
    @param delta_q: Max path-length to cover, possibly resulting in changes to "to_point"
    @returns path: list of points leading from "from_node" to "to_point" (inclusive of endpoints)
    '''
    # Don't modify this function, instead change the functions it calls
    if DYNAMICS_MODE == 'holonomic':
        return steer_holonomic(from_point, to_point, delta_q)
    elif DYNAMICS_MODE == 'discrete_non_holonomic':
        return steer_discrete_non_holonomic(from_point, to_point, delta_q)
    elif DYNAMICS_MODE == 'continuous_non_holonomic':
        # Skipping this one this year
        return steer_continuous_non_holonomic(from_point, to_point)

def state_is_valid(state, state_bounds):
    for dim in range(state_bounds.shape[0]):
        if state[dim] < state_bounds[dim][0]: return False
        if state[dim] >= state_bounds[dim][1]: return False
    for obs in obstacles:
        if np.linalg.norm(state - obs[0]) <= obs[1]: return False
    return True
    #holonomic can go any side

# import numpy as np
def steer_holonomic(from_point, to_point, delta_q):
    '''
    @param from_point: Point where the path to "to_point" is originating from
    @param to_point: n-Dimensional point (array) indicating destination
    @param delta_q: Max path-length to cover, possibly resulting in changes to "to_point"
    @returns path: list of points leading from "from_node" to "to_point" (inclusive of endpoints)
    '''

    # TODO: Use a path resolution of 10 steps for computing a path between points
    # Hint: np.linspace may be useful here   
    # it chops the length to n values

    path = []
    # print(from_point, to_point, 10)
    # pdb.set_trace()
    path_points = np.linspace(from_point, to_point, 10)
    # print("__________________________________________________________________")
    # print(path_points)
    # pdb.set_trace()

    # for each point check valid or not
    # add to path if less than q_delta
    # parent, path
    for point in path_points:
        # if state_is_valid(point, state_bounds): # this checks if the path is obstackle free
        if distance(from_point, point) < delta_q:
            path.append(point)
    return path

# steer_holonomic([.02, .03827], [.987876, .546], .0765)

    
def steer_discrete_non_holonomic(from_point, to_point, delta_q):
    '''
    Given a fixed discrete action space and dynamics model, 
    choose the action that gets you closest to "to_point" when executing it from "from_point"

    @param from_point: Point where the path to "to_point" is originating from
    @param to_point: n-Dimensional point (array) indicating destination
    @returns path: list of points leading from "from_node" to "to_point" (inclusive of endpoints)
    '''
    # Our discrete non-holonomic action space will consist of a limited set of movement primitives
    # your code should choose an action from the actions_list and apply it for this implementation
    # of steer. You can simulate an action with simulate_non_holonomic_action(state_vector, action_vector)
    # which will give you a list of points the agent travels, starting at state_vector. 
    # Index -1 (the last element) is where the agent ends up.
    '''s = from_point
    # for i in range(1000):
    actions_list = initialize_non_holonomic_actions()
    # get the action that generates closest path
    min_dist = float("inf")
    for a in actions_list:
        s_new_list = simulate_non_holonomic_action(s, a)
        dist_new = distance(from_point, to_point)
        if dist_new < min_dist and dist_new < delta_q:
            min_dist = dist_new
            path_with_min_dist = s_new_list
    path = path_with_min_dist
    return path'''
    actions_list = initialize_non_holonomic_actions()
    # get the action that generates closest path
    min_dist = float("inf")
    for a in actions_list:
        s_new_list = simulate_non_holonomic_action(from_point, a)
        s_path = []
        for point in s_new_list:
        # if state_is_valid(point, state_bounds): # this checks if the path is obstackle free
            if distance(from_point, point) < delta_q:
                s_path.append(point)
    
        dist_new = distance(s_path[-1], to_point)
        if dist_new < min_dist:
            min_dist = dist_new
            path = s_path
    # path = path_with_min_dist
    return path
    # Hint: Use simulate_non_holonomic_action(state, action) to test actions

def rrt(state_bounds, obstacles, state_is_valid, starting_point, goal_point, k, delta_q):
    '''
    TODO: Implement the RRT algorithm here, making use of the provided state_is_valid function. 
    If goal_point is set, your implementation should return once a path to the goal has been found, 
    using k as an upper-bound. Otherwise, it should build a graph without a goal and terminate after k
    iterations.
    pass

    @param state_bounds: matrix of min/max values for each dimension (e.g., [[0,1],[0,1]] for a 2D 1m by 1m square)
    @param state_is_valid: function that maps states (N-dimensional Real vectors) to a Boolean (indicating free vs. forbidden space)
    @param starting_point: Point within state_bounds to grow the RRT from
    @param goal_point: Point within state_bounds to target with the RRT. (OPTIONAL, can be None)
    @param k: Number of points to sample
    @param delta_q: Maximum distance allowed between vertices
    @returns List of RRT graph nodes
    '''
    def my_get_random_valid_vertex(state_valid, bounds, obstacles):
    # NOTE: Do not use this function in your implementation directly, but feel free to make your own variant of it. (The grading script will change this)
        vertex = None
        while vertex is None: # Get starting vertex
            pt = np.random.rand(bounds.shape[0]) * (bounds[:,1]-bounds[:,0]) + bounds[:,0]
            if state_valid(pt):
                vertex = pt
        return vertex

    node_list = []

    node_list.append(Node(starting_point))
    for i in range(k):
        print("_____________________________rrt--", i, "_____________________________________")
        x_rand = my_get_random_valid_vertex(state_is_valid, state_bounds, obstacles)
        # make some % of time as goal pt
        x_nn = get_nearest_vertex(node_list, x_rand)
        # print(x_nn.point, x_rand, delta_q)
        # pdb.set_trace()
        u_list = steer(x_nn.point, x_rand, delta_q) 
        # make sure u_list is the list of valid pts
        # print("__________________________________________________________________")

        # check if the path is valid
        is_valid_path =False
        for pt in u_list:
            if state_is_valid(pt): 
                is_valid_path = True
                continue
            else:
                is_valid_path = False
                break
        if is_valid_path: # got a valid path
            parent = x_nn
            point = u_list[-1]
            x_new = Node(point, parent)
            x_new.path_from_parent = u_list
            node_list.append(x_new)
            # check if goal is reached
    if goal_point is not None:
        goal_node = Node(goal_point)
        for node in node_list:
            if distance(node.point, goal_point) < delta_q:
                is_valid_path = False
                new_path = steer(node.point, goal_point, delta_q)
                for pt_ in new_path:
                    if state_is_valid(pt_): 
                        is_valid_path = True
                        continue # check if u_list has valid pts
                    else: is_valid_path = False
                if not is_valid_path: continue
                goal_node.parent = node
                goal_node.cost = node.cost + distance(node.cost, goal_point)
                goal_node.path_from_parent = new_path
        node_list.append(goal_node)
    return node_list

def get_nn_list(node_list, q_point, radius):
    dmin = float("inf")
    nnear_list = []
    for node in node_list:
        d = distance(node.point, q_point)
        if d < radius:
            nnear_list.append(node)

    return nnear_list

def rrt_star(state_bounds, obstacles, state_is_valid, starting_point, goal_point, k, delta_q):
    '''
    TODO: Implement the RRT* algorithm here, making use of the provided state_is_valid function

    @param state_bounds: matrix of min/max values for each dimension (e.g., [[0,1],[0,1]] for a 2D 1m by 1m square)
    @param state_is_valid: function that maps states (N-dimensional Real vectors) to a Boolean (indicating free vs. forbidden space)
    @param k: Number of points to sample
    @param delta_q: Maximum distance allowed between vertices
    @returns List of RRT* graph nodes
    '''

    def my_get_random_valid_vertex(state_valid, bounds, obstacles):
        # NOTE: Do not use this function in your implementation directly, but feel free to make your own variant of it. (The grading script will change this)
        vertex = None
        while vertex is None: # Get starting vertex
            pt = np.random.rand(bounds.shape[0]) * (bounds[:,1]-bounds[:,0]) + bounds[:,0]
            if state_valid(pt):
                vertex = pt
        return vertex

    node_list = []
    node_list.append(Node(starting_point))
    goal_node = Node(goal_point)
    # ??? how to represent node/graph here G= (V, E) ?????
    for i in range(k):
        # steer func will return the node list
        # the last point of steer x_new
        print("_____________________________rrt* --", i, "_____________________________________")
        x_rand = my_get_random_valid_vertex(state_is_valid, state_bounds, obstacles)
        x_nearest = get_nearest_vertex(node_list, x_rand)
        u_list = steer(x_nearest.point, x_rand, delta_q)
        #check if u_list has only obstacle free path, else discard the whole path
        is_valid_path = False
        for pt in u_list:
            if state_is_valid(pt): 
                is_valid_path = True
                continue
            else:
                is_valid_path = False
                break
        if is_valid_path:
            parent = x_nearest 
            point = u_list[-1]
            x_new = Node(point, parent)
            x_new.path_from_parent = u_list
            node_list.append(x_new)
            
            # modify cost to new for
            #initialize x_min, c_min
            x_min, c_min = x_nearest, x_nearest.cost + distance(x_nearest.point, x_new.point)
            nn_list= get_nn_list(node_list, x_new.point, radius= delta_q)
            #calculate the cost
            new_path = x_new.path_from_parent
            for x_near in nn_list:
                # check if the path is valid near and new pt
                
                cost = x_near.cost + distance(x_near.point, x_new.point)
                if cost < c_min:
                    x_min, c_min = x_near, cost
                    is_valid_path = False
                    new_path = steer(x_min.point, x_new.point, delta_q)
                    for pt in new_path:
                        if state_is_valid(pt): 
                            is_valid_path = True
                            continue # check if u_list has valid pts
                        else: is_valid_path = False
                    if not is_valid_path: continue
                x_new.parent = x_min
                x_new.cost = c_min
                x_new.path_from_parent = new_path
            #rewire
            for x_near in nn_list:
                cost = x_new.cost + distance(x_near.point, x_new.point)
                if cost < x_near.cost:
                    #new path
                    # flag to check the path
                    is_valid_path = False
                    # while not is_valid_path:
                    new_path = steer(x_near.point, x_new.point, delta_q)
                    # check if the path is valid near and new pt
                    for pt in new_path:
                        if state_is_valid(pt): # check if u_list has valid pts
                            is_valid_path = True
                            continue
                        else: is_valid_path = False
                    if not is_valid_path: continue
                    x_near.parent = x_new
                    x_near.cost = cost
                    x_near.path_from_parent = new_path
    if goal_point is not None:
        for node in node_list:
            if distance(node.point, goal_point) < delta_q:
                is_valid_path = False
                new_path = steer(node.point, goal_point, delta_q)
                for pt_ in new_path:
                    if state_is_valid(pt_): 
                        is_valid_path = True
                        continue # check if u_list has valid pts
                    else: is_valid_path = False
                if not is_valid_path: continue
                goal_node.parent = node
                goal_node.cost = node.cost + distance(node.cost, goal_point)
                goal_node.path_from_parent = new_path
    node_list.append(goal_node)         
    return node_list

if __name__ == "__main__":
    K = 250 # Feel free to adjust as desired
    ###############################
    # Problem 1a
    ###############################
    DYNAMICS_MODE = 'holonomic'
    bounds, obstacles, validity_check = setup_fixed_test_2d_world()
    starting_point = None
    starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    nodes = rrt(bounds, obstacles, validity_check, starting_point, None, K, np.linalg.norm(bounds/10.))
    visualize_2D_graph(bounds, obstacles, nodes, None, './figures/rrt_run1.png') 

    bounds, obstacles, validity_check = setup_random_2d_world()
    starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    nodes = rrt(bounds, obstacles, validity_check, starting_point, None, K, np.linalg.norm(bounds/10.))
    visualize_2D_graph(bounds, obstacles, nodes, None, './figures/rrt_run2.png')

    bounds, obstacles, validity_check = setup_fixed_test_2d_world()
    starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    while np.linalg.norm(starting_point - goal_point) < np.linalg.norm(bounds/2.):
        starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
        goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    nodes = rrt(bounds, obstacles, validity_check, starting_point, goal_point, K, np.linalg.norm(bounds/10.))
    if len(nodes) > 0:
        visualize_2D_graph(bounds, obstacles, [nodes[0]], goal_point, './figures/rrt_run3-empty.png')
    visualize_2D_graph(bounds, obstacles, nodes, goal_point, './figures/rrt_run3.png')

    bounds, obstacles, validity_check = setup_random_2d_world()
    starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    while np.linalg.norm(starting_point - goal_point) < np.linalg.norm(bounds/2.):
        starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
        goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    nodes = rrt(bounds, obstacles, validity_check, starting_point, goal_point, K, np.linalg.norm(bounds/10.))
    if len(nodes) > 0:
        visualize_2D_graph(bounds, obstacles, [nodes[0]], goal_point, './figures/rrt_run4-empty.png')
    visualize_2D_graph(bounds, obstacles, nodes, goal_point, './figures/rrt_run4.png')

    ###############################
    # Problem 1b
    ###############################
    DYNAMICS_MODE = 'holonomic'
    bounds, obstacles, validity_check = setup_fixed_test_2d_world()
    starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    while np.linalg.norm(starting_point - goal_point) < np.linalg.norm(bounds/2.):
        starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
        goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    nodes_rrt = rrt(bounds, obstacles, validity_check, starting_point, goal_point, K, np.linalg.norm(bounds/10.))
    visualize_2D_graph(bounds, obstacles, nodes_rrt, goal_point, './figures/rrt_comparison_run1.png')
    nodes_rrtstar = rrt_star(bounds, obstacles, validity_check, starting_point, goal_point, K, np.linalg.norm(bounds/10.))
    visualize_2D_graph(bounds, obstacles, nodes_rrtstar, goal_point, './figures/rrt_star_run1.png')
    if len(nodes_rrt) > 0:
        visualize_2D_graph(bounds, obstacles, [nodes_rrt[0]], goal_point, './figures/rrt_star_run1-empty.png')

    bounds, obstacles, validity_check = setup_fixed_test_2d_world()
    starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    while np.linalg.norm(starting_point - goal_point) < np.linalg.norm(bounds/2.):
        starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
        goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    nodes_rrt = rrt(bounds, obstacles, validity_check, starting_point, goal_point, K, np.linalg.norm(bounds/10.))
    visualize_2D_graph(bounds, obstacles, nodes_rrt, goal_point, './figures/rrt_comparison_run2.png')
    nodes_rrtstar = rrt_star(bounds, obstacles, validity_check, starting_point, goal_point, K, np.linalg.norm(bounds/10.))
    visualize_2D_graph(bounds, obstacles, nodes_rrtstar, goal_point, './figures/rrt_star_run2.png')
    if len(nodes_rrt) > 0:
        visualize_2D_graph(bounds, obstacles, [nodes_rrt[0]], goal_point, './figures/rrt_star_run2-empty.png')

    ###############################
    # Problem 1c
    ###############################
    if GRAD is True:
        DYNAMICS_MODE = 'discrete_non_holonomic'
        bounds, obstacles, validity_check = setup_fixed_test_2d_world()
        starting_point = None
        starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
        nodes = rrt(bounds, obstacles, validity_check, starting_point, None, K, np.linalg.norm(bounds/10.))
        visualize_2D_graph(bounds, obstacles, nodes, None, './figures/rrt_nh_run1.png')

        bounds, obstacles, validity_check = setup_fixed_test_2d_world()
        starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
        goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
        while np.linalg.norm(starting_point - goal_point) < np.linalg.norm(bounds/4.):
            starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
            goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
        nodes = rrt(bounds, obstacles, validity_check, starting_point, goal_point, K, np.linalg.norm(bounds/10.))
        if len(nodes) > 0:
            visualize_2D_graph(bounds, obstacles, [nodes[0]], goal_point, './figures/rrt_nh_run2-empty.png')
        visualize_2D_graph(bounds, obstacles, nodes, goal_point, './figures/rrt_nh_run2.png')


