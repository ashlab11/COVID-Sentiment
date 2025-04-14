import torch

def train_tree(tree, X, y):
    indices = torch.randint(0, len(X), (len(X),))
    X_samples, y_samples = X[indices], y[indices]
    tree.fit(X_samples, y_samples)
    return tree