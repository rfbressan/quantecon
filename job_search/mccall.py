"""McCall job search model
"""

from abc import ABC, abstractmethod
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

    def __compute_stopping_time(self, seed: int = 1234) -> int:
        """One time simulation of how many offers until the worker accepts."""
        np.random.seed(seed)
        cdf = np.cumsum(self.q)
        w_bar = self.reservation_wage()
        t = 0
        while True:
            # Draw a wage offer
            w = self.w[qe.random.draw(cdf)]
            if w >= w_bar:
                break
            else:
                t += 1

        return t

    def simulate_mean_stopping_time(self, num_reps: int = 1_000) -> float:
        """Mean stopping time of the worker.

        Parameters
        ----------
        num_reps : int, optional
            Number of simulations to take the average time, by default 1_000

        Returns
        -------
        float
            The average stopping time.
        """
        stopping_times = np.empty(num_reps)
        for i in range(num_reps):
            stopping_times[i] = self.__compute_stopping_time(seed=i)

        return stopping_times.mean()


# Class to hold utility functions
@dataclass
class Utility(ABC):
    """Utility function abstract class."""

    @abstractmethod
    def function(self):
        pass


# CRRA utility class
@dataclass
class CRRAUtility(Utility):
    """CRRA utility function."""

    σ: float = 2.0

    def function(self, c: float) -> float:
        """CRRA utility function."""
        σ = self.σ
        if σ == 1:
            return np.log(c)
        else:
            return (c ** (1 - σ) - 1) / (1 - σ)


# McCall Model with separation class
@dataclass
class McCallModelSeparation:
    """McCall model with separation."""

    utility: Utility
    α: float = 0.2
    β: float = 0.98
    c: float = 6.0
    w: NDArray = np.arange(10, 60, 10)
    q: NDArray = np.ones(len(w)) / len(w)

    def __update(self, v: NDArray, d: float) -> tuple[NDArray, float]:
        """Update rule for value function and continuation value."""
        α, β, c, w, q = self.α, self.β, self.c, self.w, self.q
        v_new = np.empty_like(v)  # Same shape and type as v

        for i in range(len(w)):
            v_new[i] = self.utility.function(w[i]) + β * ((1 - α) * v[i] + α * d)

        d_new = np.sum(np.maximum(v, self.utility.function(c) + β * d) * q)

        return v_new, d_new

    def solve_model(
        self, max_iter: int = 2000, tol: float = 1e-6, verbose: bool = False
    ) -> tuple[NDArray, float]:
        """Solves the iteration procedure. Returns the value function, v, and the continuation value, d."""
        # Initial guesses
        v = np.ones_like(self.w)
        d = 1.0

        for i in range(max_iter):
            v_new, d_new = self.__update(v, d)
            error1 = np.max(np.abs(v - v_new))
            error2 = np.abs(d - d_new)
            error = max(error1, error2)
            if verbose:
                print(f"Error at iteration {i} is {error:.4f}.")
            if error < tol:
                v = v_new
                d = d_new
                break

            v = v_new
            d = d_new
        else:
            raise ValueError(f"Convergence not reached in {max_iter} iterations.")

        return v, d

    def reservation_wage(self, v: NDArray, d: float) -> tuple[int, float]:
        """Compute the reservation wage."""
        α, β, c, w, q = self.α, self.β, self.c, self.w, self.q
        h = self.utility.function(c) + β * d
        idx = np.searchsorted(v, h, side="right")
        return idx, float(w[idx])


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


def main():
    """Program to test the McCallModel class."""
    n, a, b = 60, 600, 400  # default parameters
    q_default = BetaBinomial(n, a, b).pdf()  # default choice of q
    w_min, w_max = 10, 20
    w_default = np.linspace(w_min, w_max, n + 1)

    # Instantiate an instance of McCallModel with default parameters
    # mcm = McCallModel(c=25, β=0.99, w=w_default, q=q_default)
    # w_bar = mcm.reservation_wage()
    # print(f"Reservation wage = {w_bar:.5f}")

    # plot_optimal_policy(mcm)
    # c_vals = np.linspace(10, 30, 25)
    # β_vals = np.linspace(0.9, 0.99, 25)
    # plot_reservation_wage_contour(c_vals, β_vals, w_default, q_default)
    # Simulate average stopping time for different values of c
    # c_vals = np.linspace(10, 40, 25)
    # avg_stop_times = np.empty(len(c_vals))
    # for i, c in enumerate(c_vals):
    #     mcm = McCallModel(c=c, β=0.99, w=w_default, q=q_default)
    #     avg_stop_times[i] = mcm.simulate_mean_stopping_time()

    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(c_vals, avg_stop_times)
    # ax.set_xlabel("c")
    # ax.set_ylabel("Mean Stopping Time")
    # plt.show()

    # Testing the McCallModelSeparation class
    crra_utility = CRRAUtility(σ=2.0)
    # utility = crra_utility.function(w_default)
    # print(f"Utility = {utility}")
    mcm_sep = McCallModelSeparation(
        crra_utility, α=0.2, β=0.98, c=6.0, w=w_default, q=q_default
    )
    v, d = mcm_sep.solve_model(verbose=False)
    # Reservation wage
    idx, w_bar = mcm_sep.reservation_wage(v, d)
    h = mcm_sep.utility.function(mcm_sep.c) + mcm_sep.β * d

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(mcm_sep.w, v, "b-", lw=2, alpha=0.7, label="$v$")
    ax.hlines(
        h, w_default[0], w_default[n], colors="green", lw=2, alpha=0.7, label="$h$"
    )
    ax.plot(w_bar, v[idx], "r*", markersize=15)
    ax.set_xlim(min(mcm_sep.w), max(mcm_sep.w))
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
