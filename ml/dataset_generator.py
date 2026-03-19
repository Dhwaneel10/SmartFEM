import numpy as np
import random
import pickle

from fem.truss_solver import solve_truss
from fem.spring_solver import solve_spring_system
from fem.bar_solver import solve_bar_system


# =========================================================
# ------------------ TRUSS GENERATION ----------------------
# =========================================================

def generate_random_nodes(n_nodes, area_size=1.0):
    return {i: (random.uniform(0, area_size), random.uniform(0, area_size)) for i in range(n_nodes)}


def generate_truss_connectivity(n_nodes):
    edges = [(0,1), (1,2), (0,2)]

    for i in range(3,n_nodes):
        j = random.randint(0,i-1)
        k = random.randint(0,i-1)
        if j != k:
            edges.append((i,j))
            edges.append((i,k))

    return edges


def assign_material(edges):
    elements = []
    for (i,j) in edges:
        E = random.uniform(100e9, 210e9)
        A = random.uniform(0.005, 0.02)
        elements.append((i,j,E,A))
    return elements


def generate_truss_sample():

    n_nodes = random.randint(3,6)

    nodes = generate_random_nodes(n_nodes)
    edges = generate_truss_connectivity(n_nodes)
    elements = assign_material(edges)

    forces = np.zeros(2*n_nodes)

    # avoid applying load on fully fixed nodes (0 and 1 are fixed)
    load_node = random.randint(2, n_nodes-1)

    forces[2*load_node] = random.uniform(-500,500)
    forces[2*load_node+1] = random.uniform(-1000,-100)

    fixed_dofs = [0,1,2,3]

    try:
        U, K = solve_truss(nodes, elements, forces, fixed_dofs)

        # -------------------------
        # Stability checks
        # -------------------------
        if np.any(np.isnan(U)) or np.any(np.isinf(U)):
            return None

        if np.max(np.abs(U)) > 1e-4:
            return None

        node_features = []
        edge_index = []
        edge_features = []

        # -------------------------
        # Node features
        # -------------------------
        for i in nodes:
            fx = forces[2*i]
            fy = forces[2*i+1]

            sx = 1 if 2*i in fixed_dofs else 0
            sy = 1 if 2*i+1 in fixed_dofs else 0

            node_features.append([fx, fy, sx, sy])

        # -------------------------
        # Edge features
        # -------------------------
        for (i,j,E,A) in elements:
            x1, y1 = nodes[i]
            x2, y2 = nodes[j]

            L = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            # prevent division by zero
            if L == 0:
                return None

            c = (x2-x1)/L
            s = (y2-y1)/L

            edge_index.append([i,j])
            edge_features.append([E*A/L, c, s])

        # -------------------------
        # Target (IMPORTANT FIX)
        # -------------------------
        U_nodes = U.reshape(-1,2).tolist()

        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_features": edge_features,
            "target": U_nodes,
            "type": "truss"
        }

    except:
        return None


# =========================================================
# ------------------ SPRING GENERATION ---------------------
# =========================================================

def generate_spring_sample():

    n_nodes = random.randint(3,6)

    # chain of springs
    elements = [(i, i+1, random.uniform(500,2000)) for i in range(n_nodes-1)]

    forces = np.zeros(n_nodes)

    # FIX ONLY ONE NODE
    fixed = [0]

    # apply load on a FREE node (not fixed)
    load_node = random.randint(1, n_nodes-1)
    forces[load_node] = random.uniform(10,100)

    try:
        U, K = solve_spring_system(n_nodes, elements, forces, fixed)

        # -------------------------
        # stability checks
        # -------------------------
        if np.any(np.isnan(U)) or np.any(np.isinf(U)):
            return None

        # reject zero displacement cases ONLY if truly zero
        if np.allclose(U, 0):
            return None

        if np.max(np.abs(U)) > 0.1:
            return None

        # -------------------------
        # build graph
        # -------------------------
        node_features = []
        edge_index = []
        edge_features = []

        for i in range(n_nodes):
            node_features.append([
                forces[i],
                0,
                1 if i in fixed else 0,
                0
            ])

        for (i,j,k) in elements:
            edge_index.append([i,j])
            edge_features.append([k,1,0])

        target = [[u,0] for u in U]

        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_features": edge_features,
            "target": target,
            "type": "spring"
        }

    except:
        return None

# =========================================================
# ------------------ BAR GENERATION ------------------------
# =========================================================

def generate_bar_sample():

    n_nodes = random.randint(2,5)

    elements = []
    for i in range(n_nodes-1):
        E = random.uniform(100e9,200e9)
        A = random.uniform(0.005,0.02)
        L = random.uniform(0.5,2)

        elements.append((i,i+1,E,A,L))

    forces = np.zeros(n_nodes)

    load_node = random.randint(0,n_nodes-1)
    forces[load_node] = random.uniform(100,1000)

    fixed = [0]

    try:
        U, K = solve_bar_system(n_nodes,elements,forces,fixed)

        if np.max(np.abs(U)) > 1e-4:
            return None

        node_features = []
        edge_index = []
        edge_features = []

        for i in range(n_nodes):
            node_features.append([forces[i],0,1 if i in fixed else 0,0])

        for (i,j,E,A,L) in elements:
            stiffness = E*A/L
            edge_index.append([i,j])
            edge_features.append([stiffness,1,0])

        target = [[u,0] for u in U]

        return {
            "node_features":node_features,
            "edge_index":edge_index,
            "edge_features":edge_features,
            "target":target,
            "type":"bar"
        }

    except:
        return None


# =========================================================
# ------------------ COMBINED SAMPLER ----------------------
# =========================================================

def generate_sample():

    choice = random.choices(
        ["spring","bar","truss"],
        weights=[2,1,1]
    )[0]

    if choice == "spring":
        return generate_spring_sample()

    elif choice == "bar":
        return generate_bar_sample()

    else:
        return generate_truss_sample()


# =========================================================
# ------------------ DATASET GENERATOR ---------------------
# =========================================================

def generate_dataset(n_samples=30000):

    dataset = []

    while len(dataset) < n_samples:

        sample = generate_sample()

        if sample is not None:
            dataset.append(sample)

        if len(dataset) % 500 == 0 and len(dataset) != 0:
            print("Generated:", len(dataset))

    return dataset


# =========================================================
# ------------------ MAIN --------------------------------
# =========================================================

if __name__ == "__main__":

    data = generate_dataset(30000)

    with open("data/fem_dataset.pkl","wb") as f:
        pickle.dump(data,f)

    print("Dataset generated:",len(data))
