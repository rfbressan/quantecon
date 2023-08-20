import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe

import markov_functions as mf

# To simulate a Markov chain, we need a stochastic matrix P and a probability
# mass function psi0 of length n from which to draw an initial realization of X0.

ψ0 = (0.3, 0.7)  # Initial distribution, 2 states 0 and 1
cdf = np.cumsum(ψ0)  # Cumulative distribution
qe.random.draw(cdf, 5)  # Draw 5 states from phi0

# Simulate a Markov chain path of length 1000
P = [[0.4, 0.6], [0.2, 0.8]]  # Stochastic matrix
X = mf.mc_sample_path(P, ψ0=ψ0, ts_length=1000)
# Plot a subset of the simulated Markov chain
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(X[:50], 'b-', lw=2, alpha=0.8)
ax.set_xlabel('time', fontsize=14)
ax.set_ylabel('State ($X_t$)', fontsize=14)
plt.show()

np.mean(X == 0)  # Fraction of time spent in state 0

# Using Quantecon's MarkovChain class
mc = qe.MarkovChain(P)  # Create an instance of MarkovChain
Xqe = mc.simulate(ts_length=1000)
np.mean(Xqe == 0)

# Adding state values and initial conditions
mc = qe.MarkovChain(P, state_values=('unemployed', 'employed'))
mc.simulate(ts_length=4, init='employed')
mc.simulate_indices(ts_length=4, init=1)

# Stationary distributions
ψ = np.array([0.25, 0.75]) 
ψ @ P  # Post-multiply ψ by P and the result is ψ again

# Theorem: If P is everywhere positive, then P has exactly one stationary distribution.

mc.stationary_distributions  # Stationary distributions of P
mf.mc_stationary_distribution(P, initial_guess= [0.30, 0.70])  # Stationary distribution of P

# Visualizing the path of convergence
ψ0 = np.array([0.1, 0.9])
ψs = np.empty((10, len(ψ0)))
ψs[0] = ψ0
for i in range(1, ψs.shape[0]):
    ψs[i] = ψs[i-1] @ P

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ψs[:, 0], ψs[:, 1], 'o', color="blue", alpha=0.5, markersize=10)
ax.set_xlabel('Unemployed')
ax.set_ylabel('Employed')
plt.show()