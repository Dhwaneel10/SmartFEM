import torch
import numpy as np


def predict_bar(node_features, edge_features):

    model_data = torch.load("ml/models/bar_model.pth")

    class BarNet(torch.nn.Module):

        def __init__(self):
            super().__init__()

            self.net = torch.nn.Sequential(
                torch.nn.Linear(2,64),
                torch.nn.ReLU(),
                torch.nn.Linear(64,64),
                torch.nn.ReLU(),
                torch.nn.Linear(64,1)
            )

        def forward(self,x):
            return self.net(x)


    model = BarNet()

    model.load_state_dict(model_data["model"])
    model.eval()

    mean = model_data["mean"]
    std = model_data["std"]

    k = edge_features[0][0]

    inputs = []

    for node in node_features:
        F = node[0]
        inv_k = 1.0 / (k + 1e-12)
        inputs.append([F/1000, inv_k*1e9])

    X = torch.tensor(inputs, dtype=torch.float)

    with torch.no_grad():
        pred = model(X)

    pred = pred * std + mean

    return pred.numpy()
