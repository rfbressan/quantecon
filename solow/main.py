# Script simulating a Solow-Swan growth model

import matplotlib.pyplot as plt
import numpy as np

from solow import CES, CobbDouglas, Solow

# Auxiliary functions definitions


def plot45(model, kgrid):
    """Plot the 45-degree line and the capital accumulation curve"""
    kstar = model.steady_state()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(kgrid, model.k_update(kgrid), "b-", label="$k_{t+1}$")
    ax.plot(kgrid, kgrid, "k--", label="$45$")
    # plot a star at the steady state
    ax.plot(kstar, kstar, "*", markersize=10.0, color="green")
    # annotate the value of kstar
    ax.annotate(
        f"$k^*=${kstar:.2f}",
        xy=(kstar, kstar),
        xytext=(kstar + 0.1, kstar - 0.1),
        fontsize=12,
        color="green",
    )
    ax.set_title("Capital accumulation")
    ax.set_xlabel("$k_t$")
    ax.set_ylabel("$k_{t+1}$")
    ax.set_title(model.production)
    ax.legend()
    plt.show()


def plot_consumption_savings(model, sgrid):
    """Plot the relation of steady state consumption and savings rate"""
    cons_dict = model.consumption_savings(sgrid)
    consumpiton = cons_dict["consumption"]
    savings_gr = cons_dict["savings_gr"]
    consumpiton_gr = cons_dict["consumption_gr"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sgrid, consumpiton, "b-", label="Consumption")
    ax.plot(savings_gr, consumpiton_gr, "g*", label="Golden Rule")
    ax.set_xlabel("Savings rate")
    ax.set_ylabel("Consumption per capita")
    ax.set_title(model.production)
    ax.legend()
    plt.show()


def plot_time_series(model, ts):
    """Plot the evolution of capital per capita over time towards the steady state"""
    # Plot time_series
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ts, "b-", label="$Capital$")
    # Horizontal dashed line at steady state
    ax.hlines(model.steady_state(), 0, 11, ["red"], "dashed")
    ax.set_xlabel("Period")
    ax.set_ylabel("Capital per capita")
    ax.set_title(model.production)
    plt.show()


cd_parameters = {
    "A": 2.0,
    "alpha": 0.3,
}
# Create a cobbdouglas production function
cobb_douglas = CobbDouglas(**cd_parameters)
# Create the Solow model with CD production function
cd_model = Solow(savings=0.3, delta=0.4, k_0=0.25, production=cobb_douglas)
# print(model)

# Plot the relation of steady state consumption and savings rate
sgrid = np.linspace(0.1, 0.9, 100)
plot_consumption_savings(cd_model, sgrid)
# Plot the 45-degree line and the capital accumulation curve
kgrid = np.linspace(0, 3, 1000)
plot45(cd_model, kgrid)
# Plot the evolution of capital per capita
time_series = cd_model.simulate_ts(k_init=3, ts_length=12)
plot_time_series(cd_model, time_series)

# CES production model
ces_parameters = {
    "A": 2.0,
    "alpha": 0.3,
    "rho": 0.5,
}
ces_function = CES(**ces_parameters)
ces_model = Solow(savings=0.3, delta=0.4, k_0=0.25, production=ces_function)

# Plot the relation of steady state consumption and savings rate
plot_consumption_savings(ces_model, sgrid)
# Plot the 45-degree line and the capital accumulation curve
plot45(ces_model, kgrid)
# Plot the time series
time_series = ces_model.simulate_ts(k_init=3, ts_length=12)
plot_time_series(ces_model, time_series)
