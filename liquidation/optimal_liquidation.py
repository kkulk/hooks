import numpy as np
import matplotlib.pyplot as plt
from mdptoolbox.mdp import ValueIteration

class AMM_MDP:
    def __init__(self, T, Delta_bar, R0, R1, gamma, sigma, mu, phi, g, INV_SPACE=75, Z_SPACE=40, SWAP_SPACE=50):
        """
        T: Total num blocks
        Delta_bar: Maximum inventory
        R0, R1: Initial reserves of the both assets in AMM
        gamma: AMM fee
        sigma: Volatility of log mispricing
        mu: Drift of log mispricing
        phi: Running inventory cost coefficient
        g: Gas cost per transaction
        """
        self.T = T
        self.Delta_bar = Delta_bar
        self.R0 = R0
        self.R1 = R1
        self.L = self.R0 * self.R1 # might be sqrt
        self.gamma = gamma
        self.sigma = sigma
        self.mu = mu
        self.phi = phi
        self.g = g
        self.INV_SPACE = INV_SPACE
        self.Z_SPACE = Z_SPACE
        self.SWAP_SPACE = SWAP_SPACE

        # discretize state and action spaces
        self.Delta_space = np.linspace(0, Delta_bar, self.INV_SPACE)  # inventory space
        self.z_space = np.linspace(-1.25 * gamma, 1.25 * gamma, self.Z_SPACE)  # mispricing space
        self.actions = np.linspace(0, Delta_bar / 10, self.SWAP_SPACE)  # swap sizes space

    def transition(self, Delta, z, a):
        """
        Compute the next state given the current state (Delta, z) and action (a).

        Delta: Current inventory
        z: Current log mispricing
        a: Action (amount to swap)
        Returns: (Delta_next, z_next, R0_next, R1_next, reward)
        """
        # update inventory
        a = min(a, Delta)

        Delta_next = max(Delta - a, 0)

        # update mispricing w initial arbitrageur jump
        z_star = np.clip(z, -self.gamma, self.gamma)


        # update reserves
        p_init = self.R1/self.R0
        if a > 0:
            R0_next = self.R0 + a
            R1_next = self.L / R0_next
        else:
            R0_next, R1_next = self.R0, self.R1 
        
        p_next = R1_next/R0_next
        
        # 10000 is for converting to bps
        z_next = z_star + 10000*(p_next-p_init)/p_init
        
        # 10000 is for converting to bps
        execution_price_improvement = (z_star + z_next)/(2*10000) * (p_init+p_next)/2 * a
        inventory_cost = -self.phi * abs(Delta)
        gas_cost = -self.g * (a > 0)
        
        reward = execution_price_improvement + inventory_cost + gas_cost
        
        z_next = z_next + self.mu + self.sigma * np.random.normal()
        
        return Delta_next, z_next, R0_next, R1_next, reward
    
    def state_index(self, Delta, z):
        """
        Map a continuous state (Delta, z) to a discrete index.
        """
        Delta_idx = np.searchsorted(self.Delta_space, Delta) - 1
        z_idx = np.searchsorted(self.z_space, z) - 1
        return Delta_idx * len(self.z_space) + z_idx

    def build_mdp_matrices(self):
        n_states = len(self.Delta_space) * len(self.z_space)
        n_actions = len(self.actions)
        
        P = np.zeros((n_actions, n_states, n_states))
        R = np.zeros((n_states, n_actions))

        for i, Delta in enumerate(self.Delta_space):
            for j, z in enumerate(self.z_space):
                state_idx = i * len(self.z_space) + j
                for k, a in enumerate(self.actions):
                    # Simulate transition
                    Delta_next, z_next, _, _, reward = self.transition(Delta, z, a)
                    reward = np.clip(reward, -1e5, 1e5)


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
    
    def simulate(self, policy, num_simulations=100):
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
                Delta, z, _, _, _ = self.transition(Delta, z, action)
                inventory.append(Delta)

            inventory_paths.append(inventory)
        
        return np.mean(inventory_paths, axis=0)
