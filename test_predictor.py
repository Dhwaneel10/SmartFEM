from ml.main_predictor import predict

node_features = [
    [0,0,1,0],
    [50,0,0,0]
]

edge_features = [[1000,1,0]]

result = predict("spring", node_features, edge_features)

print("Prediction:", result)
