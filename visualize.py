import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# SPRING VISUALIZATION (1D)
# =========================================================

def plot_spring(U_fem, U_ml):

    nodes = np.arange(len(U_fem))

    plt.figure()

    plt.plot(nodes, U_fem, 'bo-', linewidth=2, label="FEM")
    plt.plot(nodes, U_ml, 'ro--', linewidth=2, label="ML")

    plt.title("Spring Displacement Comparison")
    plt.xlabel("Node Index")
    plt.ylabel("Displacement")

    plt.legend()
    plt.grid()

    plt.show()


# =========================================================
# BAR VISUALIZATION (1D DEFORMATION)
# =========================================================

def plot_bar(U_fem, U_ml):

    n = len(U_fem)

    x_original = np.arange(n)

    # scale displacement for visibility
    scale = 1e6

    x_fem = x_original + U_fem * scale
    x_ml  = x_original + U_ml * scale

    plt.figure()

    # original
    plt.plot(x_original, np.zeros(n), 'k--', linewidth=2, label="Original")

    # FEM
    plt.plot(x_fem, np.zeros(n), 'b-', linewidth=3, label="FEM")

    # ML
    plt.plot(x_ml, np.zeros(n), 'r--', linewidth=2, label="ML")

    plt.title("Bar Deformation (Scaled)")
    plt.yticks([])

    plt.legend()
    plt.grid()

    plt.show()


# =========================================================
# TRUSS VISUALIZATION (2D STRUCTURE)
# =========================================================

def plot_truss(nodes, elements, U, title=""):

    scale = 1000  # amplify displacement

    plt.figure()

    for (i, j, *_) in elements:

        # original coords
        x1, y1 = nodes[i]
        x2, y2 = nodes[j]

        # deformed coords
        dx1 = x1 + U[2*i] * scale
        dy1 = y1 + U[2*i+1] * scale

        dx2 = x2 + U[2*j] * scale
        dy2 = y2 + U[2*j+1] * scale

        # original structure
        plt.plot([x1, x2], [y1, y2], 'k--', linewidth=1)

        # deformed structure
        plt.plot([dx1, dx2], [dy1, dy2], 'r', linewidth=2)

    plt.title(title + " (Deformation ×1000)")
    plt.axis("equal")
    plt.grid()

    plt.show()
