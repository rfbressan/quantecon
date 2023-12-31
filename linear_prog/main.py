import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from ortools.linear_solver import pywraplp
from scipy.optimize import linprog

# Production problem

# \begin{aligned}
# \max_{x_1,x_2} \ & z = 3 x_1 + 4 x_2 \\
# \mbox{subject to } \ & 2 x_1 + 5 x_2 \leq 30 \\
# & 4 x_1 + 2 x_2 \leq 20 \\
# & x_1, x_2 \geq 0 \\
# \end{aligned}

# Draw the lines representing the constraints and the objective function
# Draw the constraint lines
x = np.linspace(-2.2, 17.5, 100)
# Feasible region
feasible_region = Polygon([[0, 0], [0, 6], [2.5, 5], [5, 0]], color="cyan", alpha=0.5)

fig, ax = plt.subplots(figsize=(8, 6))
ax.grid()
ax.hlines(0, -2.2, 17.5, lw=2, color="gray")
ax.vlines(0, -2.2, 12, lw=2, color="gray")
ax.plot(x, (30 - 2 * x) / 5, lw=2, label=r"$2x_1 + 5x_2 \leq 30$", color="red")
ax.plot(x, (20 - 4 * x) / 2, lw=2, label=r"$4x_1 + 2x_2 \leq 20$", color="red")
ax.plot(x, (12 - 3 * x) / 4, lw=2, label=r"$3x_1 + 4x_2$", color="blue")
ax.plot(x, (20 - 3 * x) / 4, lw=2, color="blue")
ax.plot(x, (27.5 - 3 * x) / 4, lw=2, color="blue")
ax.plot(2.5, 5, "*", color="black")
ax.add_patch(feasible_region)
ax.set_ylim(-1.0, 12)
ax.legend()
plt.show()

# Solve the problem using OR-Tools
solver = pywraplp.Solver.CreateSolver("GLOP")
# Create the variables
x1 = solver.NumVar(0, solver.infinity(), "x1")
x2 = solver.NumVar(0, solver.infinity(), "x2")
# Set the objective function
solver.Maximize(3 * x1 + 4 * x2)
# Add the constraints
solver.Add(2 * x1 + 5 * x2 <= 30)
solver.Add(4 * x1 + 2 * x2 <= 20)
# Solve the problem
status = solver.Solve()  # returns an integer code

if status == pywraplp.Solver.OPTIMAL:
    print("Solution:")
    print("Objective value = ", solver.Objective().Value())
    print("x1 = ", x1.solution_value())
    print("x2 = ", x2.solution_value())
else:
    print("The problem does not have an optimal solution.")

# Useful transformations

# The standard form LP problem can be expressed concisely as:
# \begin{aligned}
# \min_{x} \ & c'x \\
# \mbox{subject to } \ & Ax = b\\
#  & x \geq 0\\
# \end{aligned}

# A minimization problem subject to equality constraints and nonnegativity constraints

# Original problem
# \begin{aligned}
# \max_{x_1,x_2} \ & 3 x_1 + 4 x_2 \\
# \mbox{subject to } \ & 2 x_1 + 5 x_2 \le 30 \\
# & 4 x_1 + 2 x_2 \le 20 \\
# & x_1, x_2 \ge 0 \\
# \end{aligned}

# equivalent to the following problem with a standard form:
# \begin{aligned}
# \min_{x_1,x_2} \ & -(3 x_1 + 4 x_2) \\
# \mbox{subject to } \ & 2 x_1 + 5 x_2 + s_1 = 30 \\
# & 4 x_1 + 2 x_2 + s_2 = 20 \\
# & x_1, x_2, s_1, s_2 \ge 0 \\
# \end{aligned}

# Inequality constraints: Given an inequality constraint $\sum_{j=1}^n a_{ij}x_j \le 0$, we can introduce a new variable $s_i\geq 0$, called a slack variable and replace the original constraint by
# $\sum_{j=1}^n a_{ij}x_j + s_i = 0$.

# Using Scipy
# Construct parameters
c_ex1 = np.array([3, 4])

# Inequality constraints
A_ex1 = np.array([[2, 5], [4, 2]])
b_ex1 = np.array([30, 20])

# Solve the maximization problem
res = linprog(-c_ex1, A_ub=A_ex1, b_ub=b_ex1, bounds=(0, None))

if res.success:
    print(f"Max value: {-res.fun}")
    print(f"Solutions: {res.x}")
else:
    print("No solution found.")
