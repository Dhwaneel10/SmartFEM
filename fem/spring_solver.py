import numpy as np


def solve_spring_system(num_nodes, elements, forces, fixed_dofs):
    """
    num_nodes : number of nodes
    elements  : [(node_i, node_j, k)]
    forces    : global force vector
    fixed_dofs: indices of fixed nodes
    """

    K = np.zeros((num_nodes, num_nodes))

    # Assemble global stiffness matrix
    for (i, j, k) in elements:
        k_local = np.array([[k, -k],
                            [-k, k]])

        dofs = [i, j]

        for a in range(2):
            for b in range(2):
                K[dofs[a], dofs[b]] += k_local[a, b]

    # Apply boundary conditions
    free_dofs = list(set(range(num_nodes)) - set(fixed_dofs))

    K_reduced = K[np.ix_(free_dofs, free_dofs)]
    F_reduced = forces[free_dofs]

    # Solve
    U = np.zeros(num_nodes)
    U[free_dofs] = np.linalg.solve(K_reduced, F_reduced)

    return U, K
