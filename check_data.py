import pickle
import numpy as np

# Load dataset
data = pickle.load(open("data/fem_dataset.pkl","rb"))

print("Total samples:", len(data))

# -------------------------------
# Split by type
# -------------------------------

spring_data = [d for d in data if d["type"] == "spring"]
bar_data    = [d for d in data if d["type"] == "bar"]
truss_data  = [d for d in data if d["type"] == "truss"]

print("\nDistribution:")
print("Spring:", len(spring_data))
print("Bar:", len(bar_data))
print("Truss:", len(truss_data))


# -------------------------------
# Extract targets safely
# -------------------------------

def extract_targets(dataset):
    values = []

    for d in dataset:
        for node in d["target"]:
            if isinstance(node, list):
                values.extend(node)
            else:
                values.append(node)

    return np.array(values)


# -------------------------------
# Dataset checker
# -------------------------------

def check_dataset(name, dataset):

    if len(dataset) == 0:
        print(f"\n===== {name} EMPTY =====")
        return

    targets = extract_targets(dataset)

    stiffness = []
    forces = []

    for d in dataset:

        for e in d["edge_features"]:
            stiffness.append(e[0])

        for n in d["node_features"]:
            forces.append(n[0])
            forces.append(n[1])

    stiffness = np.array(stiffness)
    forces = np.array(forces)

    print(f"\n===== {name} =====")

    print("Displacement:")
    print("  Max:", np.max(targets))
    print("  Min:", np.min(targets))
    print("  Mean:", np.mean(targets))

    print("Stiffness:")
    print("  Max:", np.max(stiffness))
    print("  Min:", np.min(stiffness))

    print("Forces:")
    print("  Max:", np.max(forces))
    print("  Min:", np.min(forces))


# -------------------------------
# Run checks
# -------------------------------

check_dataset("SPRING", spring_data)
check_dataset("BAR", bar_data)
check_dataset("TRUSS", truss_data)
