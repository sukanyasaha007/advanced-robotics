# CSCI 4302/5302 HW2
# (C) Brad Hayes <bradley.hayes@colorado.edu> 2021
version = "v2021.9.9.0000"

import gym
import gym_mountaincar
import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import itertools
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


student_name = "Sukanya Saha" # Set to your name
GRAD = True  # Set to True if graduate student


class TabularPolicy(object):
    def __init__(self, n_bins_per_dim, num_dims, n_actions):
        self.num_states = n_bins_per_dim ** num_dims        
        self.num_actions = n_actions
        
        self._transition_function = np.zeros(shape=(self.num_states, self.num_actions, self.num_states))
        self._reward_function = np.zeros(shape=(self.num_states, self.num_actions, self.num_states))

        # Create data structure to store mapping from state to value
        # self._value_function[state] = state value
        self._value_function = np.zeros(shape=(self.num_states,))

        # Create data structure to store array with probability of each action for each state
        # self._policy[state] = [array of action probabilities]
        self._policy = np.random.uniform(0, 1, size=(self.num_states, self.num_actions))  

    def get_action(self, state):
        '''
        Returns an action drawn from the policy's action distribution at state
        '''
        prob_dist = np.array(self._policy[state])
        assert prob_dist.ndim == 1

        # Sample from policy distribution for state
        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.multinomial.html
        idx = np.random.multinomial(1, prob_dist / np.sum(prob_dist))
        return np.argmax(idx)

    def set_state_value(self, state, value):
        '''
        Sets the value of a given state
        '''
        self._value_function[state] = value
    
    def get_state_value(self, state):
        '''
        Returns the value of a given state
        '''
        return self._value_function[state]

    def get_value_function(self):
        '''
        Returns the table representing the value function itself.
        Useful to do for storing a value function at the end of an iteration of VI or PI...
        '''
        return copy.deepcopy(self._value_function)

    def set_value_function(self, v):
        '''
        Sets value function with new matrix v
        '''
        self._value_function = copy.copy(v)

    def set_policy(self, state, action_prob_array):
        '''
        Sets the action probabilities for a given state
        '''
        self._policy[state] = copy.copy(action_prob_array)

    def get_policy(self, state):
        '''
        Returns the probability distribution over actions for a given state
        '''
        return self._policy[state]

    def get_policy_function(self):
        '''
        Returns the table representing the policy function itself.
        '''
        return copy.deepcopy(self._policy)

    def set_entire_policy_function(self, p):
        '''
        Sets value function with new matrix v
        '''
        self._policy = copy.deepcopy(p)

    def set_entire_transition_function(self, t):
        '''
        Sets the entire transition function
        '''
        self._transition_function = copy.deepcopy(t)

    def set_transition_function(self, state_idx, action_idx, next_state_idx, val):
        '''
        Sets the transition function entry for a (s,a,s') tuple.
        '''
        self._transition_function[state_idx, action_idx, next_state_idx] = val

    def T(self, state_idx, action_idx, next_state_idx):
        '''
        Gets the transition function entry for a (s,a,s') tuple.
        '''
        return self._transition_function[state_idx, action_idx, next_state_idx]

    def set_reward_function(self, state_idx, action_idx, next_state_idx, val):
        '''
        Sets the reward function entry for a (s,a,s') tuple.
        '''
        self._reward_function[state_idx, action_idx, next_state_idx] = val

    def R(self, state_idx, action_idx, next_state_idx):
        '''
        Gets the reward function entry for a (s,a,s') tuple.
        '''
        return self._reward_function[state_idx, action_idx, next_state_idx]

class DiscretizedSolver(object):
    def __init__(self, mode, num_bins=21, lookahead=-1):
        self._mode = mode
        self._lookahead_steps = lookahead
        assert mode in ['nn', 'linear', 'lookahead']
        self._num_bins = num_bins

        self.env = gym.make("mountaincar5302-v0") # Problem Environment
        self.env_name = 'MountainCar'
        start_state = self.env.reset()

        self.state_lower_bound = self.env.observation_space.low
        self.state_upper_bound = self.env.observation_space.high
        self.bin_sizes = (self.state_upper_bound - self.state_lower_bound) / self._num_bins
        self.num_dims = self.state_lower_bound.shape[0]
        self.gamma = 0.99
        self.solver = TabularPolicy(self._num_bins, self.num_dims, self.env.action_space.n)
        self.performance_history = [] # List of cumulative rewards from successive runs of the policy
        self.populate_transition_and_reward_funcs() # Fill in our own T and R functions by using the simulator for the environment!

    def populate_transition_and_reward_funcs(self):
        '''
        Making use of self.add_transition, this function sets up the last of our discretized MDP variables
        T and R by iterating over all of our discrete states and filling in the appropriate values.
        '''

        # Iterate over all state/action combinations
        # Call self.add_transition for each
        # state env set
        
        # go through list of actions
        # find all possible state action transitions
        for s in range(self.solver.num_states):
            for a in range(self.solver.num_actions):
                self.add_transition(s,a)
        # assert(self.solver._transition_function.shape != (441, 3, 441))

    def add_transition(self, state_idx, action_idx):
        '''
        Sets the discretized MDP transition and reward values for a given state-action pair.
        state_idx: int corresponding to the 'from' state
        action_idx: int corresponding to the action taken
        '''

        # MountainCar is deterministic, so you do not need to sample multiple times for each state/action pair
        # in order to find out T and R.

        # HINT: You can set self.env.state to the state you wish to simulate (e.g., self.env.state = [0.001, 0.5])
        # HINT: You can use the self.env.step function to simulate an action. It returns 4 values: next_state, reward, environment_complete, info.
        # HINT: Keep in mind that we're setting T and R for our approximate, discretized MDP! You'll need to map the continuous state onto your discrete state space somehow.

        # Your code here. Remember to set_transition_function and set_reward_function on self.solver to store the values you compute!
        
        self.env.state= self.get_coordinates_from_state_index(state_idx)
        next_state, reward, environment_complete, info = self.env.step(action_idx)
        dis_prob= self.get_discrete_state_probabilities(next_state)
        for next_st, prob in dis_prob:
            self.solver.set_transition_function(state_idx, action_idx, int(next_st), prob)
            self.solver.set_reward_function(state_idx, action_idx, int(next_st), reward)


    def get_discrete_state_probabilities(self, continuous_state_vector):
        '''
        Given a continuous state (e.g., [1.2,-0.3]), ???? velocity negative?
        return a list of (discrete state index, probability of being in this state) tuples

        For mode=='nn', this list will only have one element, consisting of the **state index**
        closest to continuous_state_vector e.g.,: [(6, 1.0)] to indicate state index 6 with 100% probability.

        For mode=='linear', this list will have multiple elements, indicating the nearest discrete states.
        Specifically, it should have 2^(state_dimension) elements, each containing (state index, probability). Mountain Car is a 2D problem.
        
        To build intuition, for a 1-dimensional problem this would indicate interpolation between two states 
        (e.g., for a 1D continuous state space with 3 discrete states S1 S2 S3, given continuous state "Sx":  S1-----Sx---S2----------S3, 
        it would return a list containing state indices 1 and 2 along with the probability of each, but not state 3.)
        '''
        # HINT: Computing distances to every state in your discrete state space is going to be slow! Try to come up with a faster solution.
        # assert(continuous_state_vector[0] > self.state_upper_bound[0] and continuous_state_vector[0] > self.state_upper_bound[1]) and (continuous_state_vector[0] < self.state_upper_bound[0] and continuous_state_vector[1] < self.state_lower_bound[1])
        
        if self._mode == 'nn':
            print("-----------------------Started Nearest Neighbor------------------------")
            return [(self.get_state_index_from_coordinates(continuous_state_vector), 1)]
        elif self._mode == 'linear':
            print("-----------------------Started Linear Interpolation------------------------")
            # Find the 9 discrete states that surround continuous_state_vector, and assign
            # probability inversely proportionate to their distance from it.
            #matrix of 21 X 21 cells so get diagonal distance and see what cell it ends up. if that cell is inside the matrix
            # then take that cell as one of the 4 nearest cells
            bin_sizes = (self.state_upper_bound - self.state_lower_bound) / self._num_bins
            # func for distance calculation
            def distance(vec1, vec2):
                ar1= np.array(vec1)
                ar2= np.array(vec2)
                return np.sqrt(np.sum(np.square(ar1 - ar2)))
            # check if the bin falls under boundary conditions
            def check_bins(cont_vec):
                if (self.state_upper_bound[0] > cont_vec[0]) or (self.state_upper_bound[1] > cont_vec[1]) or (self.state_lower_bound[0] < cont_vec[0]) or (self.state_lower_bound[1] < cont_vec[1]):
                    return True
                return False
            #get 9 nearest states
            nn1, nn2, nn3, nn4, nn5, nn6, nn7, nn8=((continuous_state_vector[0] - bin_sizes[0], continuous_state_vector[1] + bin_sizes[1]),
                                                    (continuous_state_vector[0], continuous_state_vector[1] + bin_sizes[1]),
                                                    (continuous_state_vector[0] + bin_sizes[0], continuous_state_vector[1] + bin_sizes[1]),
                                                    (continuous_state_vector[0] - bin_sizes[0], continuous_state_vector[1]),
                                                    (continuous_state_vector[0] + bin_sizes[0], continuous_state_vector[1]),
                                                    (continuous_state_vector[0] - bin_sizes[0], continuous_state_vector[1] - bin_sizes[1]),
                                                    (continuous_state_vector[0], continuous_state_vector[1] - bin_sizes[1]),
                                                    (continuous_state_vector[0] + bin_sizes[0], continuous_state_vector[1] - bin_sizes[1]))
            nearest_neigh = np.zeros(shape=(9, 2))
            for i, nn in enumerate([nn1, nn2, nn3, nn4, nn5, nn6, nn7, nn8]):
                nn_id= self.get_state_index_from_coordinates(nn)
                if check_bins(nn):
                    nearest_neigh[i] = np.array((int(nn_id), 1/distance(continuous_state_vector, nn))) # take inverse of the distance
            # append the discreate state for current bin
            nn0 = self.get_state_index_from_coordinates(continuous_state_vector)
            # distance from the current bin
            dist00= distance(continuous_state_vector, self.get_coordinates_from_state_index(nn0))
            temp = [[int(nn0) , 1/dist00 if dist00 != 0 else dist00]]
            nearest_neigh= np.append(nearest_neigh, temp, axis=0)
            nearest_neigh= nearest_neigh[(-nearest_neigh[:, 1]).argsort()] 
            # pick top 4 nn
            print(nearest_neigh)
            nearest_neigh= nearest_neigh[:4] if len(nearest_neigh) > 4 else nearest_neigh
            # create prob distribution norm
            print(nearest_neigh)
            nearest_neigh[:, 1] /= nearest_neigh[:, 1].sum()
            # print(nearest_neigh)
            nearest_neigh[:, 0] = nearest_neigh[:, 0].astype(int)
            nearest_neigh = np.nan_to_num(nearest_neigh, nan= 0)
            return nearest_neigh
            


    def compute_policy(self, max_iterations= 1):
        '''
        Compute a policy and store it in self.solver (type TabularPolicy) using Value Iteration for max_iterations iterations.
        '''
        # HINT: Add the cumulative reward from self.solve() into self.performance_history each iteration
        #       Make sure to pick a reasonable value for the max_steps parameter (i.e., less than infinity)

        # Your Code Here -- You should be able to reuse most of your VI code from tabular_solution.py here
        # GRADs: Make sure to include code supporting the policy evaluation for n-step lookahead at the end of each iteration.
        #        You can use self.solver.set_policy_function and self.solve to get reward values to append to self.performance_history

        # HINT: Because this one can take a while to converge, set up a stopping criteria.
        #       If ||(v_i+1 - v_i)||_2 is small (i.e., distance between v_i+1 and v_i is small), the value function isn't updating much and you can safely stop iterating.


        # Remember to call self.solver.set_policy/value_function at the end!
        n_states= self.solver.num_states
        n_actions= self.solver.num_actions
        v_i= np.zeros(shape=(n_states))
        # v_i= np.zeros(shape=(n_states))
        for i in range(max_iterations):
            print("-----Compute Policy: itr---------", i)
            v_i_plus_1, p_i_plus_1= np.zeros(shape=(n_states)), np.zeros(shape=(n_states, n_actions))
            for s in range(n_states):
                q= np.zeros(n_actions)
                for a in range(n_actions):
                    r= 0
                    trans = self.solver._transition_function[s, a, :]
                    r = self.solver._reward_function[s, a, :]
                    q[a] = np.dot(trans.T, (r + self.gamma * v_i))
                v_i_plus_1[s]= np.max(q)
                # norm vals
                p_i_plus_1[s][np.argmax(q)]= 1
            v_i= v_i_plus_1
            '''lookahead
                change value and policy for n next steps
                '''
            if self._lookahead_steps > 0:
                print("-------------starting n step lookahead-----------------------")
                self.solver.set_value_function(v_i_plus_1)
                self.solver.set_entire_policy_function(p_i_plus_1)
                r, steps= self.solve(max_steps= 200)
                self.performance_history.append(r)           
        self.solver.set_value_function(v_i_plus_1)
        self.solver.set_entire_policy_function(p_i_plus_1)

    def new_method(self, s, a):
        r = self.solver._reward_function[s, a, :]
        return r

    def get_state_index_from_coordinates(self, continuous_state_vector):
        '''
        Returns the discrete state index of a given continuous state vector, 
        using the number of bins provided at instantiation
        '''
        bin_sizes = (self.state_upper_bound - self.state_lower_bound) / self._num_bins
        bin_location = ((continuous_state_vector - self.state_lower_bound) / bin_sizes).astype(int)
        return bin_location[0] * self._num_bins + bin_location[1]
        
    def get_coordinates_from_state_index(self, state_idx):
        '''
        Returns the continuous state vector for a given discrete state index, returning
        the coordinates from the "middle" of discrete state cell
        '''
        bin_sizes = (self.state_upper_bound - self.state_lower_bound) / self._num_bins
        coordinates = np.array([(state_idx//self._num_bins + 0.5) * bin_sizes[0], (state_idx % self._num_bins + 0.5) * bin_sizes[1]]) + self.state_lower_bound
        return coordinates

    def evalute_action_sequence(self, start_state, action_sequence):
        '''
        Simulate forward N steps from start_state using action_sequence, then return the accumulated reward
        ex: evaluate_action_sequence([0.4,0.2], [1,1,1])
        '''

        self.env.state = self.get_coordinates_from_state_index(start_state)
        q= 0
        print(action_sequence)
        for a in action_sequence:
            _, r, _, info= self.env.step(a)
            print(info)
            q += r
        return q
            

    def solve(self, visualize=False, max_steps=float('inf')):
        '''
        Reset the environment, then applies the solver's policy. 
        Returns cumulative reward and number of actions taken.
        '''

        finished = False
        cur_state = self.env.reset()
        if visualize is True: self.env.render()

        cumulative_reward = 0
        num_steps = 0

        while finished is False and num_steps < max_steps:
            # Take an action in the environment
            action = None

            if self._lookahead_steps > 0:
                # Your Code Here -- Implement n-step lookahead here for n = self._lookahead_steps
                # Generate all n-length action sequences and evaluate them using the evaluate_action_sequence function above
                # Remember to record the start state before simulating so you can reset it after you've found an action sequence to execute the first action of.
                # HINT: itertools.product can help you find all the action sequences.
                
                action_sequences = list(itertools.product([0,1,2], repeat= self._lookahead_steps)) # TODO: Replace this line!
                cur_state = self.env.state
                rewards = np.zeros(shape=len(action_sequences))
                for i, action_sequence in enumerate(action_sequences):
                    rewards[i]= self.evalute_action_sequence(cur_state, action_sequence)
                best_ac_seq = np.argmax(rewards)
                action = action_sequence[best_ac_seq][0]
                self.env.state= cur_state
                
                # take n actions instead of 1, using random shooting or max ent
                
            elif self._mode == 'nn':
                # get_dist_prob on curr st----> get action
                action = self.solver.get_action(self.get_state_index_from_coordinates(cur_state)) # TODO: Replace this line!
            elif self._mode == 'linear':
                # get dist st prob
                disc_prob= self.get_discrete_state_probabilities(cur_state)
                st, prob= zip(*disc_prob)
                # list of prob for each 
                # draw state from that
                chosen_st = np.random.choice(st, 1, p = prob)
                # solver.get action
                action = self.solver.get_action(chosen_st)  # TODO: Replace this line!

            else:
                action = self.solver.get_action(self.get_state_index_from_coordinates(cur_state))

            # Execute Action            
            next_state, reward, finished, info = self.env.step(action)
            
            # Update state
            cur_state = next_state
            
            # Update cumulative reward
            cumulative_reward += reward

            # Update action counter
            num_steps += 1

            # Display new state of the world
            if visualize is True: self.env.render()

        return cumulative_reward, num_steps

        
    def plot_value_function(self, value_function, filename=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)
        ax = fig.axes[0]

        V = ((value_function - value_function.min()) / (value_function.max() - value_function.min() + 1e-6))
        V = V.reshape(self._num_bins, self._num_bins).T
        image = (plt.cm.coolwarm(V)[::-1,:,:-1] * 255.).astype(np.uint8)
        ax.set_title("Env: %s" % self.env_name )
        ax.imshow(image)
        width, height = fig.get_size_inches() * fig.get_dpi()
        canvas.draw()
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height),int(width),3)

        if filename is None:
            if os.path.isdir("./figures") is False:
                os.mkdir("./figures")
            filename = "figures/%s_%s_%d.png" % (self.env_name, self._mode, self._num_bins)

        fig.savefig(filename)

        return image, fig


def plot_policy_curves(reward_histories, filename=None):
    plt.close()
    plt.clf()
    symbols = ['bs', 'r--', 'g^']
    for idx, reward_history in enumerate(reward_histories):
        plt.plot(range(len(reward_history)), reward_history, symbols[idx%len(symbols)])

    plt.xlabel("Iteration")
    plt.ylabel("Return")
    plt.title("Policy Iteration Performance" )

    plt.savefig(filename)
    return


if __name__ == '__main__':
    ############ Q3.a ############
    bin_counts = [21, 51, 151]
    for bin_count in bin_counts:
        mc_solver = DiscretizedSolver(mode='nn', num_bins=bin_count, lookahead=-1)
        start_time = time.time()
        mc_solver.compute_policy()
        elapsed_time = time.time() - start_time
        print("Computed Q3.a VI Policy with bin size %d in %g seconds" % (bin_count, elapsed_time))
        mc_solver.plot_value_function(mc_solver.solver.get_value_function())

    ############ Q3.b ############
    linear_solvers = []
    bin_counts = [21, 51, 151]
    for bin_count in bin_counts:
        mc_solver = DiscretizedSolver(mode='linear', num_bins=bin_count, lookahead=-1)
        start_time = time.time()
        mc_solver.compute_policy()
        elapsed_time = time.time() - start_time
        print("Computed Q3.b VI Policy with bin size %d in %g seconds" % (bin_count, elapsed_time))
        mc_solver.plot_value_function(mc_solver.solver.get_value_function())
        linear_solvers.append(mc_solver)

    ############ Q4 ############
    if GRAD is True:
        n_step = [1,2,3]
        performance = []
        for lookahead in n_step:
            mc_solver = DiscretizedSolver(mode='linear', num_bins=51, lookahead=lookahead)
            start_time = time.time()
            mc_solver.compute_policy()
            elapsed_time = time.time() - start_time
            print("Computed Q4 Lookahead Policy in %g seconds" % elapsed_time)
            performance.append(mc_solver.performance_history)
        plot_policy_curves(performance, "figures/lookahead_mc.png")

