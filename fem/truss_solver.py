import numpy as np


def truss_element_stiffness(x1,y1,x2,y2,E,A):

    L = np.sqrt((x2-x1)**2 + (y2-y1)**2)

    c = (x2-x1)/L
    s = (y2-y1)/L

    k = (E*A/L) * np.array([
        [ c*c,  c*s, -c*c, -c*s],
        [ c*s,  s*s, -c*s, -s*s],
        [-c*c, -c*s,  c*c,  c*s],
        [-c*s, -s*s,  c*s,  s*s]
    ])

    return k


def solve_truss(nodes, elements, forces, fixed_dofs):

    num_dofs = 2*len(nodes)

    K = np.zeros((num_dofs,num_dofs))

    for element in elements:

        n1,n2,E,A = element

        x1,y1 = nodes[n1]
        x2,y2 = nodes[n2]

        k_local = truss_element_stiffness(x1,y1,x2,y2,E,A)

        dofs = [
            2*n1,
            2*n1+1,
            2*n2,
            2*n2+1
        ]

        for i in range(4):
            for j in range(4):
                K[dofs[i],dofs[j]] += k_local[i,j]

    free_dofs = list(set(range(num_dofs)) - set(fixed_dofs))

    K_reduced = K[np.ix_(free_dofs, free_dofs)]
    F_reduced = forces[free_dofs]

    U = np.zeros(num_dofs)

    U[free_dofs] = np.linalg.solve(K_reduced,F_reduced)

    return U,K
def compute_element_forces(nodes, elements, U):

    forces = []

    for element in elements:

        n1, n2, E, A = element

        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]

        L = np.sqrt((x2-x1)**2 + (y2-y1)**2)

        c = (x2-x1)/L
        s = (y2-y1)/L

        u_e = np.array([
            U[2*n1],
            U[2*n1+1],
            U[2*n2],
            U[2*n2+1]
        ])

        B = np.array([-c, -s, c, s])

        F = (E*A/L) * np.dot(B, u_e)

        forces.append(F)

    return forces
