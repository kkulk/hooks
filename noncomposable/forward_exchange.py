import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from latexify import latexify

# Parameters
R1 = 400
R0 = 200

# Create trade size array
delta = np.linspace(0, 100, 1000)

# Define forward exchange functions
def G_constant_product(d):
    """Constant product forward exchange function (α=1)"""
    return 2*d - (R1/R0**2)*d**2

def G_linear(d):
    """Linear forward exchange function (α=0)"""
    return 2*d - (R1/R0**2)*d

def G_noncomposable(d, alpha):
    """Non-composable hook forward exchange function with variable alpha"""
    return 2*d - (R1/R0**2)*d**(1+alpha)

# Create figure
latexify(fig_width=8, fig_height=6)
plt.figure()

# Plot constant product CFMM
plt.plot(delta, G_constant_product(delta), 
         label='Constant Product ($\\alpha$=1)', 
         linestyle='--', 
         color='#1f77b4',
         linewidth=2)

# Plot linear exchange
plt.plot(delta, G_linear(delta), 
         label='Linear ($\\alpha$=0)', 
         linestyle='--', 
         color='#2ca02c',
         linewidth=2)

# Plot non-composable hook with different alpha values
alpha_values = [0.5, 0.75, 0.9]
colors = ['#ff7f0e', '#d62728', '#9467bd']

for alpha, color in zip(alpha_values, colors):
    plt.plot(delta, G_noncomposable(delta, alpha), 
             label=f'Non-composable ($\\alpha$={alpha})', 
             color=color,
             linewidth=1.5)

# Customize plot
plt.xlabel('Trade size, $\Delta$', fontsize=16)
plt.ylabel('Output amount, $G(\Delta)$', fontsize=16)
plt.title('Noncomposable hook forward exchange functions', fontsize=18)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=14)

# Set axis limits
plt.xlim(0, 80)
plt.ylim(0, G_linear(100) * 1.1)

# Format ticks
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x)}'))
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x)}'))

# Adjust layout and save
plt.tight_layout()
plt.savefig('../figures/forward_exchange_functions.png', dpi=300, bbox_inches='tight')
