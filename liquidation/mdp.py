from amm import AMM

class MDP:
    def __init__(self, amm, T, Delta_bar, mu, sigma, phi, g, INV_SPACE=75, Z_SPACE=40, SWAP_SPACE=50):
        self.amm = amm
        self.T =  T
        self.delta_bar = Delta_bar
        self.mu = mu
        self.sigma = sigma
        self.phi = phi
        self.g = g
        self.INV_SPACE = INV_SPACE
        self.Z_SPACE = Z_SPACE
        self.SWAP_SPACE = SWAP_SPAC

        # discretize state and action spaces
        self.Delta_space = np.linspace(0, Delta_bar, self.INV_SPACE)  # inventory space
        self.z_space = np.linspace(-1.25 * gamma, 1.25 * gamma, self.Z_SPACE)  # mispricing space
        self.actions = np.linspace(0, Delta_bar / 10, self.SWAP_SPACE)  # swap sizes spaceE
