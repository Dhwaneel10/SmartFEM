import numpy as np

from ml.main_predictor import predict
from fem.spring_solver import solve_spring_system
from fem.bar_solver import solve_bar_system
from fem.truss_solver import solve_truss
from visualize import plot_spring, plot_bar, plot_truss

# =========================================================
# SPRING TEST
# =========================================================

def test_spring():

    print("\n--- SPRING TEST ---")

    n = 3
    elements = [(0,1,1000),(1,2,1000)]
    forces = np.array([0,50,0])
    fixed = [0]

    # FEM
    U_fem, _ = solve_spring_system(n,elements,forces,fixed)

    # convert to predictor format
    node_features = []
    for i in range(n):
        node_features.append([forces[i],0,1 if i in fixed else 0,0])

    edge_features = [[1000,1,0]]

    # ML
    U_ml = predict("spring", node_features, edge_features)

    U_ml = U_ml.flatten()

    # ---------- ROBUST ERROR METRICS ----------

    diff = U_fem - U_ml

    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))

    # safe normalization (uses max displacement instead of tiny values)
    norm = np.max(np.abs(U_fem)) + 1e-12
    normalized_error = mae / norm * 100

    print("FEM:", U_fem)
    print("ML :", U_ml)

    print(f"\nMAE  : {mae:.6e}")
    print(f"RMSE : {rmse:.6e}")
    plot_spring(U_fem, U_ml)
 


# =========================================================
# BAR TEST
# =========================================================

def test_bar():

    print("\n--- BAR TEST ---")

    n = 3
    elements = [(0,1,200e9,0.01,1),(1,2,200e9,0.01,1)]
    forces = np.array([0,500,0])
    fixed = [0]

    # FEM
    U_fem, _ = solve_bar_system(n,elements,forces,fixed)

    node_features = []
    for i in range(n):
        node_features.append([forces[i],0,1 if i in fixed else 0,0])

    EA_L = 200e9*0.01/1
    edge_features = [[EA_L,1,0]]

    # ML
    U_ml = predict("bar", node_features, edge_features)
    U_ml = U_ml.flatten()

    # ---------- ROBUST ERROR METRICS ----------

    diff = U_fem - U_ml

    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))

    # safe normalization (uses max displacement instead of tiny values)
    norm = np.max(np.abs(U_fem)) + 1e-12
    normalized_error = mae / norm * 100

    print("FEM:", U_fem)
    print("ML :", U_ml)

    print(f"\nMAE  : {mae:.6e}")
    print(f"RMSE : {rmse:.6e}")
    plot_bar(U_fem, U_ml)
 



# =========================================================
# TRUSS TEST
# =========================================================

def test_truss():

    print("\n--- TRUSS TEST ---")

    nodes = {
        0:(0,0),
        1:(1,0),
        2:(0.5,0.8)
    }

    elements = [
        (0,1,200e9,0.01),
        (1,2,200e9,0.01),
        (0,2,200e9,0.01)
    ]

    forces = np.array([0,0,0,0,0,-500])
    fixed_dofs = [0,1,2,3]

    # FEM
    U_fem, _ = solve_truss(nodes,elements,forces,fixed_dofs)

    node_features = []
    for i in nodes:
        fx = forces[2*i]
        fy = forces[2*i+1]
        sx = 1 if 2*i in fixed_dofs else 0
        sy = 1 if 2*i+1 in fixed_dofs else 0
        node_features.append([fx,fy,sx,sy])

    edge_index = [[0,1],[1,2],[0,2]]

    # ML
    U_ml = predict("truss", node_features, None, edge_index)

    U_fem = U_fem.reshape(-1,2)

    # ---------- ROBUST ERROR METRICS ----------

    diff = U_fem - U_ml

    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))

    # safe normalization (uses max displacement instead of tiny values)
    norm = np.max(np.abs(U_fem)) + 1e-12
    normalized_error = mae / norm * 100

    print("FEM:", U_fem)
    print("ML :", U_ml)

    print(f"\nMAE  : {mae:.6e}")
    print(f"RMSE : {rmse:.6e}")
    plot_truss(nodes, elements, U_fem.flatten(), title="FEM")
    plot_truss(nodes, elements, U_ml.flatten(), title="ML")




# =========================================================
# RUN ALL
# =========================================================

if __name__ == "__main__":

    test_spring()
    test_bar()
    test_truss()
