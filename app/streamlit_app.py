import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from ml.main_predictor import predict
from fem.spring_solver import solve_spring_system
from fem.bar_solver import solve_bar_system
from fem.truss_solver import solve_truss
from ml.gemini_chat import ask_ai


# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="SmartFEM", layout="wide")

st.markdown("""
<h1 style='text-align: center;'>🚀 SmartFEM: AI-Powered Structural Solver</h1>
<h4 style='text-align: center;'>Bridging Classical FEM with Machine Learning</h4>
<h5 style='text-align: center;'>By Dhwaneel Pandya</h5>
""", unsafe_allow_html=True)

problem = st.selectbox("Select Problem Type", ["Spring", "Bar", "Truss"])
st.divider()


# =========================================================
# SPRING
# =========================================================
if problem == "Spring":

    st.header("Spring System")

    n = st.number_input("Nodes", 2, 10, 3)
    k = st.number_input("Spring Stiffness", value=1000.0)
    force = st.number_input("Load at last node", value=50.0)

    if st.button("Solve Spring"):

        elements = [(i, i+1, k) for i in range(n-1)]
        forces = np.zeros(n)
        forces[-1] = force
        fixed = [0]

        U_fem, K = solve_spring_system(n, elements, forces, fixed)

        node_features = [[forces[i],0,1 if i in fixed else 0,0] for i in range(n)]
        edge_features = [[k,1,0]]
        U_ml = predict("spring", node_features, edge_features).flatten()

        # ---------------------------
        st.subheader("Boundary Conditions")
        st.write("Fixed Nodes:", fixed)
        st.write("Forces:", forces)

        # ---------------------------
        st.subheader("Local Element Matrix")
        k_local = np.array([[k, -k], [-k, k]])
        st.dataframe(k_local)

        # ---------------------------
        st.subheader("Global Stiffness Matrix")
        st.dataframe(K)

        # ---------------------------
        st.subheader("Governing Equation")
        st.latex(r"K \cdot U = F")

        # ---------------------------
        col1, col2 = st.columns(2)
        with col1:
            st.write("FEM Displacement", U_fem)
        with col2:
            st.write("ML Prediction", U_ml)

        # ---------------------------
        st.subheader("Strain (ε = Δu)")
        strain = np.diff(U_fem)
        st.write(strain)

        st.subheader("Stress (σ = kε)")
        stress = k * strain
        st.write(stress)

        # ---------------------------
        fig, ax = plt.subplots()
        ax.plot(U_fem, 'bo-', label="FEM")
        ax.plot(U_ml, 'ro--', label="ML")
        ax.legend()
        st.pyplot(fig)

        # ---------------------------
        st.session_state["problem"] = "Spring"
        st.session_state["forces"] = forces.tolist()
        st.session_state["U_fem"] = U_fem.tolist()

        # 🤖 AI Explanation
        st.subheader("🤖 AI Explanation")

        explanation_prompt = f"""
Explain this spring FEM result:

Forces: {forces}
Displacements: {U_fem}

Explain what is happening physically.
"""
        st.write(ask_ai(explanation_prompt))


# =========================================================
# BAR
# =========================================================
elif problem == "Bar":

    st.header("Bar Element")

    n = st.number_input("Nodes", 2, 10, 3)
    E = st.number_input("Young's Modulus", value=200e9)
    A = st.number_input("Area", value=0.01)
    L = st.number_input("Length", value=1.0)
    force = st.number_input("Load", value=500.0)

    if st.button("Solve Bar"):

        elements = [(i,i+1,E,A,L) for i in range(n-1)]
        forces = np.zeros(n)
        forces[-1] = force
        fixed = [0]

        U_fem, K = solve_bar_system(n, elements, forces, fixed)

        EA_L = E*A/L

        node_features = [[forces[i],0,1 if i in fixed else 0,0] for i in range(n)]
        edge_features = [[EA_L,1,0]]
        U_ml = predict("bar", node_features, edge_features).flatten()

        # ---------------------------
        st.subheader("Local Element Matrix")
        k_local = EA_L * np.array([[1,-1],[-1,1]])
        st.dataframe(k_local)

        # ---------------------------
        st.subheader("Global Matrix")
        st.dataframe(K)

        st.latex("KU = F")

        # ---------------------------
        col1, col2 = st.columns(2)
        with col1:
            st.write("FEM", U_fem)
        with col2:
            st.write("ML", U_ml)

        # ---------------------------
        st.subheader("Strain (ε = Δu/L)")
        strain = np.diff(U_fem)/L
        st.write(strain)

        st.subheader("Stress (σ = Eε)")
        stress = E * strain
        st.write(stress)

        # ---------------------------
        fig, ax = plt.subplots()
        ax.plot(U_fem, 'bo-')
        st.pyplot(fig)

        # ---------------------------
        st.session_state["problem"] = "Bar"
        st.session_state["forces"] = forces.tolist()
        st.session_state["U_fem"] = U_fem.tolist()

        # 🤖 AI Explanation
        st.subheader("🤖 AI Explanation")

        explanation_prompt = f"""
Explain this bar FEM result:

Forces: {forces}
Displacements: {U_fem}
"""
        st.write(ask_ai(explanation_prompt))


# =========================================================
# TRUSS
# =========================================================
elif problem == "Truss":

    st.header("2D Truss")

    n = st.number_input("Nodes", 3, 6, 3)

    nodes = {}
    for i in range(n):
        col1, col2 = st.columns(2)
        x = col1.number_input(f"x{i}", value=float(i))
        y = col2.number_input(f"y{i}", value=0.0)
        nodes[i] = (x,y)

    # ✅ FIX: TRIANGLE CREATION
    elements = []
    for i in range(n-1):
        elements.append((i,i+1,200e9,0.01))
    if n >= 3:
        elements.append((0,n-1,200e9,0.01))   # closes triangle

    forces = np.zeros(2*n)

    load_node = st.number_input("Load node",0,n-1,0)
    fx = st.number_input("Fx",0.0)
    fy = st.number_input("Fy",-500.0)

    forces[2*load_node] = fx
    forces[2*load_node+1] = fy

    fixed = st.number_input("Fixed node",0,n-1,0)
    roller = st.number_input("Roller node",0,n-1,1)

    fixed_dofs = [2*fixed,2*fixed+1,2*roller+1]

    if st.button("Solve Truss"):

        try:
            U, K = solve_truss(nodes,elements,forces,fixed_dofs)
        except:
            st.error("Unstable Structure")
            st.stop()

        # -------------------------
        st.subheader("Local Element Matrix (General Form)")
        st.latex(r"""
k_e = \frac{EA}{L}
\begin{bmatrix}
c^2 & cs & -c^2 & -cs \\
cs & s^2 & -cs & -s^2 \\
-c^2 & -cs & c^2 & cs \\
-cs & -s^2 & cs & s^2
\end{bmatrix}
""")

        # -------------------------
        st.subheader("Global Stiffness Matrix")
        st.dataframe(K)

        st.subheader("Governing Equation")
        st.latex("KU = F")

        # -------------------------
        U_nodes = U.reshape(-1,2)

        st.subheader("Nodal Displacements")
        st.dataframe(U_nodes)

        # -------------------------
        st.subheader("Strain Matrix")

        strain_list = []
        stress_list = []

        for (i,j,E,A) in elements:

            x1,y1 = nodes[i]
            x2,y2 = nodes[j]

            L = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            c = (x2-x1)/L
            s = (y2-y1)/L

            ui = np.array([U[2*i],U[2*i+1]])
            uj = np.array([U[2*j],U[2*j+1]])

            strain = ((uj-ui) @ np.array([c,s]))/L
            stress = E*strain

            strain_list.append(strain)
            stress_list.append(stress)

        st.dataframe(np.array(strain_list).reshape(-1,1))

        st.subheader("Stress Matrix")
        st.dataframe(np.array(stress_list).reshape(-1,1))

        # -------------------------
        # 🔥 FIXED VISUALIZATION
        fig, ax = plt.subplots()

        coords = np.array(list(nodes.values()))
        span = max(coords[:,0].max()-coords[:,0].min(),
                   coords[:,1].max()-coords[:,1].min())

        max_disp = np.max(np.abs(U)) + 1e-12
        scale = 0.2 * span / max_disp   # 🔥 correct scaling

        for (i,j,_,_) in elements:

            x1,y1 = nodes[i]
            x2,y2 = nodes[j]

            # original
            ax.plot([x1,x2],[y1,y2],'gray',linewidth=1)

            # deformed
            dx1 = x1 + U[2*i]*scale
            dy1 = y1 + U[2*i+1]*scale
            dx2 = x2 + U[2*j]*scale
            dy2 = y2 + U[2*j+1]*scale

            ax.plot([dx1,dx2],[dy1,dy2],'red',linewidth=2)

        ax.set_title("Truss Deformation (Scaled)")
        ax.set_aspect('equal')

        st.pyplot(fig)

        # -------------------------
        st.session_state["problem"] = "Truss"
        st.session_state["forces"] = forces.tolist()
        st.session_state["U_fem"] = U_nodes.tolist()

        # -------------------------

        # 🤖 AI Explanation
        st.subheader("🤖 AI Explanation")

        explanation_prompt = f"""
Explain this truss FEM result:

Forces: {forces}
Displacements: {U_nodes}
"""
        st.write(ask_ai(explanation_prompt))


# =========================================================
# CHAT
# =========================================================
st.subheader("💬 Ask Doubts")

query = st.text_input("Ask FEM Doubt")

if query and "U_fem" in st.session_state:

    context = f"""
Problem: {st.session_state.get('problem')}
Forces: {st.session_state.get('forces')}
Displacements: {st.session_state.get('U_fem')}

Question: {query}
"""

    st.write(ask_ai(context))
