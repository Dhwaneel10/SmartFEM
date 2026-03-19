from ml.spring_predictor import predict_spring
from ml.bar_predictor import predict_bar
from ml.truss_predictor import predict_truss


def predict(problem_type, node_features, edge_features, edge_index=None):

    if problem_type == "spring":
        return predict_spring(node_features, edge_features)

    elif problem_type == "bar":
        return predict_bar(node_features, edge_features)

    elif problem_type == "truss":
        return predict_truss(node_features, edge_index)

    else:
        raise ValueError("Invalid problem type")
