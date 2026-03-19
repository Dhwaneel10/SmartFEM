import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np


class TrussNet(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(4,32)
        self.conv2 = GCNConv(32,32)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(32,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,2)
        )

    def forward(self,data):

        x = torch.relu(self.conv1(data.x, data.edge_index))
        x = torch.relu(self.conv2(x, data.edge_index))

        return self.fc(x)


def predict_truss(node_features, edge_index):

    model = TrussNet()
    model.load_state_dict(torch.load("ml/models/truss_model.pth"))
    model.eval()

    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)

    with torch.no_grad():
        pred = model(data)

    return pred.numpy()
