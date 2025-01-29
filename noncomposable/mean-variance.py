import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from latexify import latexify   

# Define constants
lambda_param = 0.05  # Lambda, tradeoff between mean and variance
R1 = 600
R0 = 300
constant_sigma = 3  # Constant variance

# Define variance functions
def sigma_squared_linear(delta, k):
    return k * delta

def sigma_squared_quadratic(delta, k):
    return k * delta**2

def sigma_squared_superlinear(delta, k):
    return k * delta**1.5

def sigma_squared_constant(sigma):
    return sigma**2

# Create meshgrid for alpha and k values
alpha_values = np.linspace(0, 1, 20)
k_values = np.linspace(0, 5, 20)
alpha_mesh, k_mesh = np.meshgrid(alpha_values, k_values)

# Initialize arrays to store optimal delta values
optimal_delta_linear = np.zeros_like(alpha_mesh)
optimal_delta_quadratic = np.zeros_like(alpha_mesh)
optimal_delta_superlinear = np.zeros_like(alpha_mesh)
optimal_delta_constant = np.zeros_like(alpha_mesh)

# Perform simulations
for i, k in enumerate(k_values):
    for j, alpha in enumerate(alpha_values):
        delta = cp.Variable()
        
        # Define G1 and G2 terms
        G1_term = 2*(100-delta) - R1 / R0**2 * cp.square(100-delta)
        G2_term = 2*delta - R1 / R0**2 * cp.power(delta, 1+alpha)
        
        # Linear variance
        penalty_term = lambda_param * sigma_squared_linear(delta, k)
        objective = cp.Maximize(G1_term + G2_term - penalty_term)
        constraints = [delta >= 0, delta <= 100]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        optimal_delta_linear[i, j] = delta.value

        # Quadratic variance
        penalty_term = lambda_param * sigma_squared_quadratic(delta, k)
        objective = cp.Maximize(G1_term + G2_term - penalty_term)
        constraints = [delta >= 0, delta <= 100]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        optimal_delta_quadratic[i, j] = delta.value

        # Superlinear variance
        penalty_term = lambda_param * sigma_squared_superlinear(delta, k)
        objective = cp.Maximize(G1_term + G2_term - penalty_term)
        constraints = [delta >= 0, delta <= 100]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        optimal_delta_superlinear[i, j] = delta.value

        # Constant variance
        penalty_term = lambda_param * sigma_squared_constant(constant_sigma)
        objective = cp.Maximize(G1_term + G2_term - penalty_term)
        constraints = [delta >= 0, delta <= 100]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        optimal_delta_constant[i, j] = delta.value

# Plotting the results
def plot_3d_surface(ax, X, Y, Z, title):
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('$\\alpha$', fontsize=14)
    ax.set_ylabel('$\\beta$', fontsize=14)
    ax.set_zlabel('$\\Delta^*$', fontsize=14)
    ax.set_title(title, fontsize=16)
    return surf

# Plot separate surfaces
latexify(fig_width=18, fig_height=8)
fig = plt.figure()

ax1 = fig.add_subplot(141, projection='3d')
surf1 = plot_3d_surface(ax1, alpha_mesh, k_mesh, optimal_delta_linear, 'Linear variance')

ax2 = fig.add_subplot(142, projection='3d')
surf2 = plot_3d_surface(ax2, alpha_mesh, k_mesh, optimal_delta_quadratic, 'Quadratic variance')

ax3 = fig.add_subplot(143, projection='3d')
surf3 = plot_3d_surface(ax3, alpha_mesh, k_mesh, optimal_delta_superlinear, 'Superlinear variance')

ax4 = fig.add_subplot(144, projection='3d')
surf4 = plot_3d_surface(ax4, alpha_mesh, k_mesh, optimal_delta_constant, 'Constant variance')


plt.savefig('../figures/separate_3d_surfaces_delta.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

# Plot linear, quadratic, and superlinear surfaces together
latexify(fig_width=18, fig_height=8)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf1 = ax.plot_surface(alpha_mesh, k_mesh, optimal_delta_linear, cmap='viridis', alpha=0.7)
surf2 = ax.plot_surface(alpha_mesh, k_mesh, optimal_delta_quadratic, cmap='plasma', alpha=0.7)
surf3 = ax.plot_surface(alpha_mesh, k_mesh, optimal_delta_superlinear, cmap='inferno', alpha=0.7)

ax.set_xlabel('$\\alpha$', fontsize=14)
ax.set_ylabel('$\\beta$', fontsize=14)
ax.set_zlabel('Optimal trade, $\\Delta$', fontsize=14)
ax.set_title('Linear vs Quadratic vs Superlinear Variance - Optimal Delta', fontsize=16)

# Add color bars which map values to colors
fig.colorbar(surf1, shrink=0.5, aspect=5, label='Linear variance')
fig.colorbar(surf2, shrink=0.5, aspect=5, label='Quadratic variance')
fig.colorbar(surf3, shrink=0.5, aspect=5, label='Superlinear variance')

plt.tight_layout()
plt.savefig('../figures/combined_3d_surface_delta.png', dpi=300)
