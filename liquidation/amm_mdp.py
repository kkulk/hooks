import numpy as np
import matplotlib.pyplot as plt
from mdptoolbox.mdp import ValueIteration

class AMM_MDP:
    def __init__(self, T, Delta_bar, R0, R1, gamma, sigma, mu, xi, g, INV_SPACE=80, Z_SPACE=40, SWAP_SPACE=40, dt=12):
        """
        T: Total num blocks
        Delta_bar: Maximum inventory
        R0, R1: Initial reserves of the both assets in AMM
        gamma: AMM fee
        sigma: Volatility of log mispricing
        mu: Drift of log mispricing
        xi: Running inventory cost coefficient
        g: Gas cost per transaction
        """
        self.T = T
        self.Delta_bar = Delta_bar
        self.R0 = R0
        self.R1 = R1
        self.L = self.R0 * self.R1
        self.gamma = gamma
        self.sigma = sigma
        self.mu = mu
        self.xi = xi
        self.g = g
        self.INV_SPACE = INV_SPACE
        self.Z_SPACE = Z_SPACE
        self.SWAP_SPACE = SWAP_SPACE
        self.dt = dt

        # discretize state and action spaces
        self.Delta_space = np.linspace(0, Delta_bar, self.INV_SPACE)  # inventory space
        self.z_space = np.linspace(-1.25 * gamma, 1.25 * gamma, self.Z_SPACE)  # mispricing space
        self.actions = np.linspace(0, Delta_bar / 10, self.SWAP_SPACE)  # swap sizes space

    # keep track of mispricings for both twamm and liquidation separately
    def transition_adjusted(self, Delta, z_liq, z_twamm, a):

        # make sure we can only swap up to our remaining inventory
        Delta_next = max(Delta - a, 0)

        # update optimal liquidation mispricing w initial arbitrageur jump
        z_star = np.clip(z_liq, -self.gamma/10000, self.gamma/10000)

        # update twamm mispricing
        z_star_twamm = np.clip(z_twamm, -self.gamma/10000, self.gamma/10000)

       
        # assume external price will be p_ext in expectation. this is an approximation
        # when the volatility is far lower than the price of the asset
        p_ext = self.R1/self.R0
        p_init = p_ext * np.exp(-z_star)
        p_twamm = p_ext*np.exp(-z_star_twamm)

        # calculate the amount out given a swap input from AMM math
        R0_init =  np.sqrt(self.L/p_init)
        R1_init =  self.L/R0_init
        R0_next = R0_init + a*(1-self.gamma/10000)
        R1_next = self.L / R0_next
        
        amount_out = R1_init - R1_next

        p_next = R1_next/R0_next
        
        # 10000 is for converting to bps
        # z = ln(external_price / pool_price)
        z_next = np.log(p_ext/p_next)
        
        # calculate the amount out given a swap input from AMM math for TWAMM
        R0_init_twamm = np.sqrt(self.L/p_twamm)
        R1_init_twamm = self.L/R0_init_twamm
        R0_next_twamm = R0_init_twamm + self.Delta_bar/self.T*(1-self.gamma/10000)
        R1_next_twamm = self.L / R0_next_twamm
        
        amount_out_twamm = R1_init_twamm - R1_next_twamm
        price_swap = amount_out_twamm/(self.Delta_bar/self.T)

        p_next_twamm = R1_next_twamm/R0_next_twamm
        z_next_twamm = np.log(p_ext/p_next_twamm)
        # 10000 is for converting to bps
        # execution price improvement is how much more we got out
        # than by swapping our a units at a fee's price away from the external venue price
        # this should be positive when mispricing is at the negative part of the arbitrage bound
        execution_amount_improvement = amount_out - (p_ext*(1-self.gamma/10000))*a
        inventory_cost = -self.xi * abs(Delta)
        gas_cost = -self.g * (a > 0)
        
        reward = execution_amount_improvement + inventory_cost + gas_cost
        
        # geometric brownian step
        jump = (
            (self.mu - (self.sigma/10000)**2 / 2) * self.dt
            + (self.sigma/10000)*np.random.normal()*np.sqrt(self.dt)
        )
        z_next_adjusted = z_next + jump

        z_next_twamm =  z_next_twamm + jump

        return Delta_next, z_next_adjusted, z_next_twamm, reward, amount_out, amount_out_twamm

    
    def state_index(self, Delta, z):
        """
        Map a continuous state (Delta, z) to a discrete index.
        """
        Delta_idx = np.searchsorted(self.Delta_space, Delta) - 1
        z_idx = np.searchsorted(self.z_space, z) - 1
        return Delta_idx * len(self.z_space) + z_idx

    def build_mdp_matrices(self):
        external_price = self.R1/self.R0
        n_states = len(self.Delta_space) * len(self.z_space)
        n_actions = len(self.actions)
        
        P = np.zeros((n_actions, n_states, n_states))
        R = np.zeros((n_states, n_actions))

        for i, Delta in enumerate(self.Delta_space):
            for j, z in enumerate(self.z_space):
                state_idx = i * len(self.z_space) + j
                for k, a in enumerate(self.actions):
                    # Simulate transition
                    Delta_next, z_next, _, reward, _, _, = self.transition_adjusted(Delta, z, 0, a)
                    reward = np.clip(reward, -1e6, 1e6)


                    # Find next state index
                    next_state_idx = self.state_index(Delta_next, z_next)

                    # Update matrices
                    P[k, state_idx, next_state_idx] += 1
                    R[state_idx, k] = reward

        # normalize probabilities
        for k in range(n_actions):
            row_sums = P[k].sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            P[k] /= row_sums

        return P, R
    
    def solve_mdp(self, discount_factor=0.99):
        """
        Solve the MDP using value iteration.

        discount_factor: Discount factor for future rewards
        Returns: Optimal policy
        """
        P, R = self.build_mdp_matrices()
        
        vi = ValueIteration(P, R, discount_factor)
        vi.run()
        self.value_function = vi.V
        self.policy = np.array(vi.policy)

        return self.policy
    
    def simulate(self, policy, num_simulations=500):
        """
        Simulate the MDP policy over multiple runs and calculate the mean inventory path.

        Args:
            policy (array): Optimal policy mapping states to actions.
            num_simulations (int): Number of independent simulations to run.

        Returns:
            np.array: Mean inventory path over all simulations.
        """
        inventory_paths = []

        for _ in range(num_simulations):
            Delta = self.Delta_bar
            z = 0 
            external_price = self.R1/self.R0
            z_twamm = 0
            inventory = [Delta]

            for t in range(self.T):
                # current state index
                Delta_idx = np.searchsorted(self.Delta_space, Delta) - 1
                z_idx = np.searchsorted(self.z_space, z) - 1
                state_index = Delta_idx * len(self.z_space) + z_idx

                # get action from policy and constrain it
                action_index = policy[state_index]
                action = self.actions[action_index]
                action = min(action, Delta) 

                # transition
                Delta, z, z_twamm, _, _, _ = self.transition_adjusted(Delta, z, z_twamm, action)
                inventory.append(Delta)

            inventory_paths.append(inventory)
        
        return np.mean(inventory_paths, axis=0)
    
    def benchmark_against_twamm(self, policy, num_simulations = 5000):
        """
        Benchmark the MDP against the TWAMM
        """
        comparison_paths = []


        for _ in range(num_simulations):
            Delta = self.Delta_bar
            external_price = self.R1/self.R0
            z = 0 
            z_twamm = 0
            reward = 0
            twamm_reward = 0
            twamm_usd = 0
            liquidation_usd = 0
            for t in range(self.T):
                # current state index
                Delta_idx = np.searchsorted(self.Delta_space, Delta) - 1
                z_idx = np.searchsorted(self.z_space, z) - 1
                state_index = Delta_idx * len(self.z_space) + z_idx

                # get action from policy and constrain it
                action_index = policy[state_index]
                action = self.actions[action_index]
                action = min(action, Delta) 

                # transition
                Delta, z, z_twamm, reward_next, amount_out, amount_out_twamm = self.transition_adjusted(Delta, z, z_twamm, action)
                
                liquidation_usd += amount_out - self.g*(action>0)
                twamm_usd += amount_out_twamm
                
            twamm_usd = twamm_usd - self.g
            # discount the execution price of the final liquidation by 5%
            liquidation_usd += Delta * self.R1/self.R0 * .95
            comparison_paths.append(liquidation_usd - twamm_usd)
        return comparison_paths
    
    def reward_path(self, policy, num_simulations=500):
        """
        Simulate the MDP policy over multiple runs and calculate the mean inventory path.

        Args:
            policy (array): Optimal policy mapping states to actions.
            num_simulations (int): Number of independent simulations to run.

        Returns:
            np.array: Mean inventory path over all simulations.
        """
        reward_paths = []

        for _ in range(num_simulations):
            Delta = self.Delta_bar
            external_price = self.R1/self.R0
            z = 0 
            z_twamm = 0
            reward = 0
            #inventory = [Delta]

            for t in range(self.T):
                # current state index
                Delta_idx = np.searchsorted(self.Delta_space, Delta) - 1
                z_idx = np.searchsorted(self.z_space, z) - 1
                state_index = Delta_idx * len(self.z_space) + z_idx

                # get action from policy and constrain it
                action_index = policy[state_index]
                action = self.actions[action_index]
                action = min(action, Delta) 

                # transition
                Delta, z, _, reward_next, _, _ = self.transition_adjusted(Delta, z, z_twamm, action)
                reward += reward_next

            reward_paths.append(reward)
        
        return np.mean(reward_paths, axis=0)
