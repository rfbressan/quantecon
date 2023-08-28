"""McCall job search model
"""

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from quantecon.distributions import BetaBinomial

import quantecon as qe


@dataclass
class McCallModel:
    c: float = 25
    β: float = 0.99
    w: NDArray = np.arange(10, 60, 10)
    q: NDArray = np.ones(len(w)) / len(w)

    def state_action_values(self, i: int, v: NDArray) -> NDArray:
        """Given a state i and a vector of values v, return the value of both actions.

        Parameters
        ----------
        i : int
            Index of the state, 0 <= i < len(w).
        v : ndarray
            Value function represented as a NumPy array.
        """
        accept = self.w[i] / (1 - self.β)
        reject = self.c + self.β * np.sum(v * self.q)

        return np.array([accept, reject])

    def reservation_wage(self, max_iter: int = 500, tol: float = 1e-6) -> float:
        """Compute the reservation wage."""

        # Initial guess for h
        h = self.c / (1 - self.β)
        # Iterate to convergence
        for i in range(max_iter):
            h_new = self.c + self.β * np.sum(
                np.maximum(self.w / (1 - self.β), h) * self.q
            )
            if np.abs(h_new - h) < tol:
                h = h_new
                break

            h = h_new
        else:
            raise ValueError(
                f"Iteration procedure did not converge in {max_iter} iterations."
            )

        return float((1 - self.β) * h)

    def optimal_policy(self) -> NDArray:
        """Compute the optimal policy. Responses are 0 for reject and 1 for accept."""
        # Set up storage
        σ = self.w >= self.reservation_wage()
        return σ


# Auxiliary functions


# Plot the optimal policy
def plot_optimal_policy(mcm: McCallModel):
    """Optimal policy as a function of the wage offer.

    Parameters
    ----------
    mcm : McCallModel
        Instance of McCallModel to extract the optimal policy.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(mcm.w, mcm.optimal_policy(), label="optimal policy")
    ax.set_xlabel("wage")
    ax.set_ylabel("Optimal Response")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Reject", "Accept"])
    plt.show()


# Plot contour for reservation wage
def plot_reservation_wage_contour(
    c_vals: NDArray, β_vals: NDArray, w: NDArray, q: NDArray
):
    """Make a countour plot of the reservation wage for varying values of parameters c and beta."""
    # Validate c_vals and β_vals have the same length
    if len(c_vals) != len(β_vals):
        raise ValueError("c_vals and β_vals must have the same length.")

    grid_size = len(c_vals)
    # Reservation wages array
    R = np.empty((grid_size, grid_size))
    # Iterate over c and β values
    for i, c in enumerate(c_vals):
        for j, β in enumerate(β_vals):
            mcm = McCallModel(c=c, β=β, w=w, q=q)
            R[i, j] = mcm.reservation_wage()

    # Plot the reservation wage
    fig, ax = plt.subplots(figsize=(8, 8))
    cs1 = ax.contourf(c_vals, β_vals, R.T, alpha=0.75)
    ctr1 = ax.contour(c_vals, β_vals, R.T)
    plt.clabel(ctr1, inline=1, fontsize=13)
    plt.colorbar(cs1, ax=ax)

    ax.set_title("Reservation Wage")
    ax.set_xlabel("$c$", fontsize=16)
    ax.set_ylabel("$β$", fontsize=16)

    plt.show()


def compute_stopping_time(mcm: McCallModel, seed: int = 1234) -> int:
    """Compute how many offers until the worker accepts."""
    np.random.seed(seed)
    cdf = np.cumsum(mcm.q)
    w_bar = mcm.reservation_wage()
    t = 0
    while True:
        # Draw a wage offer
        w = mcm.w[qe.random.draw(cdf)]
        if w >= w_bar:
            break
        else:
            t += 1

    return t


def simulate_mean_stopping_time(mcm: McCallModel, num_reps: int = 1_000) -> float:
    """Simulate mean stopping time."""
    stopping_times = np.empty(num_reps)
    for i in range(num_reps):
        stopping_times[i] = compute_stopping_time(mcm, seed=i)

    return stopping_times.mean()


def main():
    """Program to test the McCallModel class."""
    n, a, b = 50, 200, 100  # default parameters
    q_default = BetaBinomial(n, a, b).pdf()  # default choice of q
    w_min, w_max = 10, 60
    w_default = np.linspace(w_min, w_max, n + 1)

    # Instantiate an instance of McCallModel with default parameters
    mcm = McCallModel(c=25, β=0.99, w=w_default, q=q_default)
    w_bar = mcm.reservation_wage()
    print(f"Reservation wage = {w_bar:.5f}")

    # plot_optimal_policy(mcm)
    # c_vals = np.linspace(10, 30, 25)
    # β_vals = np.linspace(0.9, 0.99, 25)
    # plot_reservation_wage_contour(c_vals, β_vals, w_default, q_default)
    avg_stop_time = simulate_mean_stopping_time(mcm)
    print(f"Average stopping time = {avg_stop_time:.3f}")


if __name__ == "__main__":
    main()
