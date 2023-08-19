# Script simulating a Solow-Swan growth model

import numpy as np
import solow

cd_parameters = {"s": 0.3, "delta": 0.4, "k0": 0.25, "production": "Cobb-Douglas", "A": 2.0, "alpha": 0.3}

# Instantiate the model
cd_model = solow.Solow(**cd_parameters)
# print(model)
# Plot the 45-degree line and the capital accumulation curve
# cd_model.plot45(np.linspace(0, 3, 1000), cd_model.steady_state())
# Plot the evolution of capital per capita
# time_series = cd_model.simulate_ts(k_init=3, ts_length=100, plot=True)
# print(time_series)

# Plot the relation of steady state consumption and savings rate
# s_grid = np.linspace(0.1, 0.9, 1000)
# consumption = cd_model.plot_consumption_savings(s_grid)

# CES production model
ces_parameters = {"s": 0.3, "delta": 0.4, "k0": 0.25, "production": "CES", "A": 2.0, "alpha": 0.3, "rho": 0.5}
ces_model = solow.Solow(**ces_parameters)
ces_model.plot45(np.linspace(0, 3, 1000), ces_model.steady_state())
ces_model.simulate_ts(k_init=3, ts_length=100, plot=True)
s_grid = np.linspace(0.1, 0.9, 1000)
ces_model.plot_consumption_savings(s_grid)
