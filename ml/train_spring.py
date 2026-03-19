import torch
import torch.nn as nn
import torch.optim as optim
import pickle


# ---------------- LOAD DATA ----------------

def load_data():

    data = pickle.load(open("data/fem_dataset.pkl","rb"))

    X, Y = [], []

    for d in data:
        if d["type"] != "spring":
            continue

        k = d["edge_features"][0][0]

        for i in range(len(d["node_features"])):

            F = d["node_features"][i][0]
            u = d["target"][i][0]

            inv_k = 1.0 / (k + 1e-12)

            X.append([F, inv_k])
            Y.append(u)

    X = torch.tensor(X, dtype=torch.float)
    Y = torch.tensor(Y, dtype=torch.float).unsqueeze(1)

    return X, Y


# ---------------- MODEL ----------------

class SpringNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self,x):
        return self.net(x)


# ---------------- TRAIN ----------------

def train():

    X, Y = load_data()

    # normalize inputs
    X[:,0] /= 100       # force scaling
    X[:,1] *= 1000      # inverse stiffness scaling

    # normalize target
    mean, std = Y.mean(), Y.std()
    Y = (Y - mean) / (std + 1e-8)

    model = SpringNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(120):

        pred = model(X)
        loss = ((pred - Y)**2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} Loss {loss.item():.6f}")

    torch.save({
        "model": model.state_dict(),
        "mean": mean,
        "std": std
    }, "ml/models/spring_model.pth")

    print("Spring model DONE")


if __name__ == "__main__":
    train()
