# CSCI 4302/5302 HW2
# (C) Brad Hayes <bradley.hayes@colorado.edu> 2021
# 
version = "v2021.9.9.0000"


import gym
import gym_gridworld
import time
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


student_name = "Sukanya Saha" # Set to your name
GRAD = True  # Set to True if graduate student


class TabularPolicy(object):
    def __init__(self, n_states, state_ranges, n_actions):
        self.num_states = n_states
        self.num_actions = n_actions
        self.state_ranges = state_ranges
        # Create data structure to store mapping from state to value
        # self._value_function[state] = state value
        self._value_function = np.zeros(shape=n_states)
        
        # Create data structure to store array with probability of each action for each state
        # self._policy[state] = [array of action probabilities]
        self._policy = np.random.uniform(0, 1, size=(n_states, self.num_actions)) # matrix action vs states

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

        if type(state) == int:
            return self._value_function[state]
        else:
            # Map state vector to state index
            return self._value_function[self.get_state_index_from_coordinates(state)]
            
    def get_state_index_from_coordinates(self, state):
        return state[0] * (self.state_ranges[0][1] - self.state_ranges[0][0]) + state[1]
 
    def get_coordinates_from_state_index(self, state_idx):
        return np.array([state_idx//(self.state_ranges[0][1] - self.state_ranges[0][0]), state_idx % (self.state_ranges[0][1] - self.state_ranges[0][0])])

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

        if type(state) == int:
            return self._policy[state]
        else:
            # Map state vector to state index
            return self._policy[self.get_state_index_from_coordinates(state)]

    def get_policy_function(self):
        '''
        Returns the table representing the policy function itself.
        '''
        return copy.deepcopy(self._policy)

    def set_policy_function(self, p):
        '''
        Sets value function with new matrix v
        '''
        self._policy = copy.copy(p)

class GridworldSolver(object):
    def __init__(self, policy_type='deterministic_vi', gridworld_map_number=0, noisy_transitions=False, max_ent_temperature=1.0):
        self._policy_type = policy_type
        assert policy_type in ['deterministic_vi', 'max_ent_vi', 'deterministic_pi']
        self.env = None # Populated with Gym Environment
        self.env_name = ''
        self.init_environment(gridworld_map_number,noisy_transitions)
        self.temperature = max_ent_temperature # For Max-Ent
        self.eps = 1e-6 # For Max-Ent
        self.gamma = 0.99 # Discount Factor
        self.solver = TabularPolicy(self.env.num_states, self.env.get_state_ranges(), self.env.num_actions) # Stores our policy and value function
        self.performance_history = [] # List of cumulative rewards from successive runs of the policy

    def init_environment(self, gridworld_map_number=0, noisy_transitions=False):
        assert gridworld_map_number in [0,1]
        if noisy_transitions is True:
            self.env_name = 'gridworldnoisy-v%d' % (gridworld_map_number)
        else:
            self.env_name = 'gridworld-v%d' % (gridworld_map_number)
        
        self.env = gym.make(self.env_name)

        self.env.reset()
        # self.env.render() # Can uncomment this line to visualize the Gridworld environment

    def compute_policy(self):
        '''
        Compute a policy and store it in self.solver (type TabularPolicy)

        HINT: You'll want to make use of the following variables/functions:
            self.env.get_state_ranges()  --  Gives valid values for each dimension of state space [min, max)
            self.env.T(S,A)              --  Returns a list of (probability, next_state) tuples
            self.env.T(S,A,S')           --  Returns a list with one (probability, next_state) tuple for s'
            self.env.R(S,A,S')           --  Returns the reward of transitioning from s to s' by action a
            Note: "state" refers to a tuple (x,y) in the Gridworld environment. The indexing for your
                  value and policy function (mapping (x,y) to a single index value) is purely an implementation
                  choice on our part and exists independent of the domain itself!
        '''

        if self._policy_type == 'deterministic_vi':
        #     # Initialize your v_i and p_i here
        #     # 
        #     # Loop for your pre-specified iteration count (horizon)
        #     # experiment- horizon 1000, v0= 0, max over values
        #     #   Create matrices for storing v_iplus1 and p_iplus1
        #     #   Implement value iteration, populating your v_i+1 and p_i+1
        #     #   Set v_i and p_i equal to v_i+1 and p_i+1 at the end of each loop
            n_states= self.env.num_states
            n_actions= self.env.num_actions
            v_i= np.zeros(shape=(n_states))
            # v_i= np.zeros(shape=(n_states))
            for i in range(50):
                print("-----itr---------", i)
                v_i_plus_1, p_i_plus_1= np.zeros(shape=(n_states)), np.zeros(shape=(n_states, n_actions))
                for s in range(n_states):
                    q= np.zeros(n_actions)
                    s_i= self.solver.get_coordinates_from_state_index(s)
                    for a in range(n_actions):
                        for prob, next_state in self.env.T(s_i, a):
                            r= self.env.R(s_i, a, next_state)
                            q[a] += prob * (r + self.gamma * v_i[self.solver.get_state_index_from_coordinates(next_state)])
                    v_i_plus_1[s]= np.max(q) #max(list(q.values()))
                    # s_i= s_future
                    p_i_plus_1[s][np.argmax(q)]= 1 #max(q, key= q.get) # best action for the most value
                    # self.performance_history.append(self.solve(s_id, visualize= False, max_steps= 20)[0]) 
                    # # vi should be a vector
                    self.solver.set_policy(s, q)
                v_i= v_i_plus_1
            self.solver.set_value_function(v_i_plus_1)
            self.solver.set_policy_function(p_i_plus_1)
        elif self._policy_type == 'max_ent_vi':
            # Implement max-ent value iteration
            # Remember to use self.temperature and self.eps here!
            n_states= self.env.num_states
            n_actions= self.env.num_actions
            v_i= np.zeros(shape=(n_states))
            for i in range(100):
                v_i_plus_1, p_i_plus_1= np.zeros(shape=(n_states)), np.zeros(shape=(n_states, n_actions))
                for s in range(n_states):
                    q= np.zeros(n_actions)
                    s_i= self.solver.get_coordinates_from_state_index(s)
                    for a in range(n_actions):
                        for prob, next_state in self.env.T(s_i, a):
                            r= self.env.R(s_i, a, next_state)
                            q[a] += prob * (r + self.gamma * v_i[self.solver.get_state_index_from_coordinates(next_state)])
                    q_log= np.log(np.sum(np.exp((q - q.max())/self.temperature)))
                    # print("******************* q_norm, norm ******************\n", self.temperature * q_norm + q_norm.max(), np.argmax(norm / norm.sum()))
                    v_i_plus_1[s]= np.max(self.temperature * q_log + q.max())
                    p_norm= np.exp((q- q.max())/self.temperature) + self.eps
                    p_norm= p_norm/np.sum(p_norm)
                    # p_i_plus_1[s][np.argmax(p_norm)]= 1
                    p_i_plus_1[s]= p_norm
                    # self.solver.set_policy(s, p_i_plus_1[s])
                    # self.solver.set_state_value(s, v_i_plus_1[s])
                v_i= v_i_plus_1
            self.performance_history.append(self.solve(visualize= False, max_steps= 20)[0])
            self.solver.set_value_function(v_i_plus_1)
            self.solver.set_policy_function(p_i_plus_1)

        if self._policy_type == 'deterministic_pi':
            # HINT: Add the cumulative reward from self.solve() into self.performance_history each iteration ( self.solve()[0] )
            #       Make sure to pick a reasonable value for the max_steps parameter (i.e., less than infinity)

            # HINT: The default policy (self.solver.get_policy_function()) is randomly initialized. You can use that as the initial p_i.
            # Initialize your v_i and p_i here
            # v_i, p_i = 
            # Loop for your pre-specified iteration count (horizon)
            #   Create matrices for storing v_iplus1 and p_iplus1
            #   Implement policy iteration, remembering to separate your Evaluate and Improve steps!
            #   Set v_i and p_i equal to v_i+1 and p_i+1 at the end of each loop
            #
            #   After each iteration, set self.solver's policy function to p_i
            #   Evaluate the policy, calling self.solve and setting the max_steps parameter to a reasonable value (e.g., 20).
            #   Add the cumulative reward from that evaluation to self.performance_history

            # Call self.solver.set_policy/value_function with your final results
            n_states= self.env.num_states
            n_actions= self.env.num_actions
            v_i_plus_1, p_i_plus_1= np.zeros((n_states)), np.zeros(shape=(n_states, n_actions))
            v_i= np.zeros(shape=(n_states))
            p_i= self.solver.get_policy_function()
            for k in range(50): #policy
                # print("Policy iter: ", k)
                #evaluate
                for s in range(n_states):
                    s_i= self.solver.get_coordinates_from_state_index(s)
                    chosen_a= np.argmax(p_i[s])
                    q= np.zeros(n_actions)
                    for prob, next_state in self.env.T(s_i, chosen_a):
                        r= self.env.R(s_i, chosen_a, next_state)
                        q[chosen_a] += prob * (r + self.gamma * v_i[self.solver.get_state_index_from_coordinates(next_state)])
                    v_i_plus_1[s] = np.max(q)#### add q to vi + 1
                v_i = v_i_plus_1
                #improve
                '''
                iter over states
                    iter over actions
                        max over all actions values
                     p_i_plus_one(s, action_id)= 1
                '''
                for s in range(n_states):
                    s_i= self.solver.get_coordinates_from_state_index(s)
                    chosen_a= np.argmax(p_i[s])
                    q= np.zeros(n_actions)
                    for a in range(n_actions): 
                        for prob, next_state in self.env.T(s_i, a):
                            r= self.env.R(s_i, a, next_state)
                            q[a] += prob * (r + self.gamma * v_i[self.solver.get_state_index_from_coordinates(next_state)])
                    best_a= np.argmax(q)
                    if chosen_a != best_a:
                        # print("need to improve")
                        p_i_plus_1[s][chosen_a] = 0
                        p_i_plus_1[s][best_a] = 1
                        self.solver.set_policy(s_i, p_i[s])
                p_i = p_i_plus_1
                self.performance_history.append(self.solve(visualize= False, max_steps= 20)[0])
                self.solver.set_value_function(v_i_plus_1)
                self.solver.set_policy_function(p_i_plus_1)
            

    def solve(self, start_state=None, visualize=False, max_steps=float('inf')):
        '''
        Reset the environment, then applies the solver's policy. 
        Returns cumulative reward and number of actions taken.
        '''

        finished = False
        cur_state = self.env.reset()
        if start_state is not None:
            self.env.agent_state = start_state
        if visualize is True: self.env.render()

        cumulative_reward = 0
        num_steps = 0

        while finished is False and num_steps < max_steps:
            # Take an action in the environment
            next_state, reward, finished, info = self.env.step(self.solver.get_action(self.solver.get_state_index_from_coordinates(cur_state)))
            
            # Update state
            cur_state = next_state
            
            # Update cumulative reward
            cumulative_reward += reward

            # Update action counter
            num_steps += 1

            # Display new state of the world
            if visualize is True: self.env.render()

        return cumulative_reward, num_steps

    def plot_policy_curve(self, reward_history, filename=None):
        plt.close()
        plt.clf()
        plt.plot(range(len(reward_history)), reward_history)
        plt.xlabel("Iteration")
        plt.ylabel("Return")
        plt.title("Env: %s Policy Iteration Performance" % self.env_name )

        if filename is None:
            if os.path.isdir("./figures") is False:
                os.mkdir("./figures")
            filename = "figures/%s_%s.png" % (self.env_name, self._policy_type)

        plt.savefig(filename)
        return

    def plot_rollouts(self, filename=None):
        # Perform rollout from every start state, record cumulative reward for each, plot heatmap
        rewards = np.zeros(shape=(self.solver.num_states))
        for start_state in range(self.solver.num_states):
            reward, step_count = self.solve(start_state=self.solver.get_coordinates_from_state_index(start_state),max_steps=20)
            rewards[start_state] = reward - step_count
        
        self.plot_value_function(rewards, filename)

    def plot_entropy(self, filename=None):
        entropy = np.zeros(shape=(self.solver.num_states))
        for s in range(self.solver.num_states):
            ent = 0
            for a in range(self.solver.num_actions):
                p = self.solver.get_policy(s)[a]
                ent -= p * np.log2(p)
            entropy[s] = ent

        self.plot_value_function(entropy, filename)

    def plot_value_function(self, value_function, filename=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)
        ax = fig.axes[0]
        state_counts = self.env.get_state_ranges()[:,1] - self.env.get_state_ranges()[:,0]
        y_size = state_counts[0]
        x_size = state_counts[1]
        V = ((value_function - value_function.min()) / (value_function.max() - value_function.min() + 1e-6)).reshape(y_size, x_size)
        image = (plt.cm.coolwarm(V)[::1,::1,:-1] * 255.).astype(np.uint8)
        ax.set_title("Env: %s" % self.env_name )
        ax.imshow(image)
        width, height = fig.get_size_inches() * fig.get_dpi()
        canvas.draw()
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height),int(width),3)

        if filename is None:
            if os.path.isdir("./figures") is False:
                os.mkdir("./figures")
            filename = "figures/%s_%s.png" % (self.env_name, self._policy_type)

        fig.savefig(filename)

        return image, fig

if __name__ == '__main__':
    
    ############ Q1.a ############
    gw0_det_solver = GridworldSolver(policy_type='deterministic_vi',gridworld_map_number=0)
    gw1_det_solver = GridworldSolver(policy_type='deterministic_vi',gridworld_map_number=1)
    for solver in [gw0_det_solver, gw1_det_solver]:
        start_time = time.time()
        solver.compute_policy()
        elapsed_time = time.time() - start_time
        print("Computed Q1.a VI Policy in %g seconds" % elapsed_time)
        solver.plot_value_function(solver.solver.get_value_function())

    ############ Q1.b ############
    if GRAD is True:
        for tempidx, temperature in enumerate([10., 1, 1e-5]):
            gw0_solver = GridworldSolver(policy_type='max_ent_vi',gridworld_map_number=0, noisy_transitions=False, max_ent_temperature=temperature)
            gw1_solver = GridworldSolver(policy_type='max_ent_vi',gridworld_map_number=1, noisy_transitions=False, max_ent_temperature=temperature)
            
            for solver in [gw0_solver, gw1_solver]:
                start_time = time.time()
                solver.compute_policy()
                elapsed_time = time.time() - start_time
                print("Computed Q1.b VI Policy in %g seconds" % elapsed_time)
                solver.env.start_grid_map[4,4] = 0 # Set unforeseen wall
                solver.plot_rollouts("figures/rollout_%s_maxent_t%d.png" % (solver.env_name,tempidx))
                # solver.plot_entropy("figures/rollout_ent_%s_maxent_t%d.png" % (solver.env_name,tempidx)) # You can use this to debug your implementation!
        for solver in [gw0_det_solver, gw1_det_solver]:
            solver.env.start_grid_map[4,4] = 0 # Set unforeseen wall
            solver.plot_rollouts("figures/rollout_%s_det.png" % (solver.env_name))
            # solver.plot_entropy("figures/rollout_ent_%s_det.png" % (solver.env_name)) # You can use this to debug your implementation!

    ############ Q2 ############
    gw0_solver = GridworldSolver(policy_type='deterministic_pi',gridworld_map_number=0)
    
    for solver in [gw0_solver]:
        start_time = time.time()
        solver.compute_policy()
        elapsed_time = time.time() - start_time
        print("Computed Q2 PI Policy in %g seconds" % elapsed_time)
        solver.plot_policy_curve(solver.performance_history)
