import numpy as np


def bar_element_stiffness(E, A, L):

    k = (E*A/L) * np.array([
        [1, -1],
        [-1, 1]
    ])

    return k


def solve_bar_system(num_nodes, elements, forces, fixed_dofs):

    K = np.zeros((num_nodes, num_nodes))

    for element in elements:

        i, j, E, A, L = element

        k_local = bar_element_stiffness(E, A, L)

        dofs = [i, j]

        for a in range(2):
            for b in range(2):
                K[dofs[a], dofs[b]] += k_local[a, b]

    free_dofs = list(set(range(num_nodes)) - set(fixed_dofs))

    K_reduced = K[np.ix_(free_dofs, free_dofs)]
    F_reduced = forces[free_dofs]

    U = np.zeros(num_nodes)
    U[free_dofs] = np.linalg.solve(K_reduced, F_reduced)

    return U, K
