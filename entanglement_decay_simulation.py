
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define parameters for the Lindblad-based decay model
def entanglement_decay(time, C0, gamma, eta, beta, kappa, alpha):
    """
    Compute the entanglement decay over time.
    """
    return C0 * np.exp(-(gamma + eta - beta - kappa + alpha * np.minimum(1, C0**2)) * time)

# Simulation parameters
time = np.linspace(0, 50, 500)  # Time steps
C0 = 1.0                        # Initial entanglement level
gamma = 0.05                    # Decoherence rate
beta = 0.02                     # Stabilizing feedback
kappa_values = [0.01, 0.05, 0.1, 0.2]  # Hierarchical coupling strengths
alpha = 0.01                    # Nonlinearity parameter

# Generate thermal noise
np.random.seed(42)  # For reproducibility
eta = np.random.normal(loc=0.01, scale=0.005, size=len(time))

# Run simulations for different coupling strengths
results = []
for kappa in kappa_values:
    entanglement = entanglement_decay(time, C0, gamma, eta, beta, kappa, alpha)
    results.append(pd.DataFrame({'Time': time, 'Entanglement': entanglement, 'Coupling': kappa}))

# Combine results into a single DataFrame
results_df = pd.concat(results, ignore_index=True)

# Save results to a CSV file for sharing
results_df.to_csv("simulation_results.csv", index=False)

# Plot results for visualization
plt.figure(figsize=(10, 6))
for kappa in kappa_values:
    subset = results_df[results_df['Coupling'] == kappa]
    plt.plot(subset['Time'], subset['Entanglement'], label=f"Coupling: {kappa}")
plt.xlabel("Time")
plt.ylabel("Entanglement Level")
plt.title("Simulated Entanglement Decay for Various Coupling Strengths")
plt.legend()
plt.grid()
plt.savefig("entanglement_decay_simulation_plot.png")
plt.show()
