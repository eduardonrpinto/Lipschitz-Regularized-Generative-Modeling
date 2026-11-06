import torch


def sliced_wasserstein(X, Y, n_proj=200, p=2):
    """Sliced Wasserstein-p distance between two point clouds (n, d)."""
    n = min(len(X), len(Y))
    X, Y = X[:n].flatten(1), Y[:n].flatten(1)
    theta = torch.randn(n_proj, X.shape[1])
    theta = theta / theta.norm(dim=1, keepdim=True)
    px = (X @ theta.T).sort(dim=0).values
    py = (Y @ theta.T).sort(dim=0).values
    return ((px - py).abs() ** p).mean().pow(1.0 / p).item()
