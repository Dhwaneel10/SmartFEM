import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv


def load_data():

    data = pickle.load(open("data/fem_dataset.pkl","rb"))

    dataset = []

    for d in data:
        if d["type"] != "truss":
            continue

        x = torch.tensor(d["node_features"], dtype=torch.float)
        edge_index = torch.tensor(d["edge_index"], dtype=torch.long).t().contiguous()
        y = torch.tensor(d["target"], dtype=torch.float)

        dataset.append(Data(x=x, edge_index=edge_index, y=y))

    return dataset


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(4,32)
        self.conv2 = GCNConv(32,32)

        self.fc = nn.Sequential(
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,2)
        )

    def forward(self,data):

        x = torch.relu(self.conv1(data.x, data.edge_index))
        x = torch.relu(self.conv2(x, data.edge_index))

        return self.fc(x)


def train():

    dataset = load_data()

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = Net()
    opt = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(150):

        total = 0

        for batch in loader:

            pred = model(batch)
            loss = ((pred - batch.y)**2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"Epoch {epoch} Loss {total:.6f}")

    torch.save(model.state_dict(),"ml/models/truss_model.pth")

    print("Truss done")


if __name__ == "__main__":
    train()
