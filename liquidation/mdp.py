from amm import AMM
import numpy as np
from mdptoolbox.mdp import ValueIteration
import matplotlib.pyplot as plt


class MDP:
    def __init__(self, amm, T, D, mu, sigma, phi, g, INV_SPACE=75, SWAP_SPACE=50, dt=12):
        self.amm = amm
        self.T = T
        self.D = D
        self.mu = mu
        self.sigma = sigma
        self.phi = phi
        self.g = g
        self.INV_SPACE = INV_SPACE
        self.SWAP_SPACE = SWAP_SPACE
        self.dt = dt
        self.external_price = self.amm.instantaneous_x_price()

        # discretize state and action spaces
        self.Delta_space = np.linspace(0, self.D, self.INV_SPACE)  # inventory space
        self.actions = np.linspace(0, self.D / 10, self.SWAP_SPACE)  # swap sizes space


    def transition_reward(self, I, Delta):
        # I is the inventory level
        # z is the mispricing level
        # Delta is the swap size
        Delta = min(Delta, I)
        I_next = I - Delta
        print(Delta)
        z = np.log(self.external_price/self.amm.instantaneous_x_price())
        #external_price = self.amm.instantaneous_x_price() *np.exp(z)
        print(self.external_price   )

        z_clipped = np.clip(z, -self.amm.gamma, self.amm.gamma)
          #arbitrage the pool
        self.amm.arb(z_clipped)
        post_arb_price = self.amm.instantaneous_x_price()
        trade_result, average_price = self.amm.trade(Delta, 0)
        post_trade_price = self.amm.instantaneous_x_price()
        z_intermediate = np.log(self.external_price/post_trade_price)
        #print(post_trade_price)
        reward = trade_result.x_out *(average_price - self.external_price* (1-self.amm.gamma)) - self.phi * I - self.g*(Delta>0)
        #z_next = z_intermediate + self.mu*self.dt + self.sigma/10000*np.random.normal()*np.sqrt(self.dt)
        last_price = self.external_price
        self.external_price = last_price + last_price*self.mu*self.dt + last_price*self.sigma/10000*np.random.normal()*np.sqrt(self.dt)
        #z_next_bps = z_next*10000

        return (I_next, reward)
    
    def state_index(self, Delta):
        """
        Map a continuous inventory state to a discrete index.
        """
        return np.searchsorted(self.Delta_space, Delta) - 1

    def build_mdp_matrices(self):
        n_states = len(self.Delta_space)
        n_actions = len(self.actions)
        
        P = np.zeros((n_actions, n_states, n_states))
        R = np.zeros((n_states, n_actions))

        for i, Delta in enumerate(self.Delta_space):
            for k, a in enumerate(self.actions):
                # Simulate transition
                Delta_next, reward = self.transition_reward(Delta, a)
                reward = np.clip(reward, -1e5, 1e5)

                # Find next state index
                next_state_idx = self.state_index(Delta_next)

                # Update matrices
                P[k, i, next_state_idx] += 1
                R[i, k] = reward

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
        """
        inventory_paths = []

        for _ in range(num_simulations):
            Delta = self.D
            inventory = [Delta]

            for t in range(self.T):
                # current state index
                state_index = self.state_index(Delta)

                # get action from policy and constrain it
                action_index = policy[state_index]
                action = self.actions[action_index]
                action = min(action, Delta)

                # transition
                Delta, _ = self.transition_reward(Delta, action)
                inventory.append(Delta)

            inventory_paths.append(inventory)
        
        return np.mean(inventory_paths, axis=0)
    