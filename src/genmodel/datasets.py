import math

import torch

DATASETS = {}


def register(name):
    def deco(fn):
        DATASETS[name] = fn
        return fn
    return deco


@register("moons")
def moons(n, noise=0.05):
    from sklearn.datasets import make_moons
    x, _ = make_moons(n, noise=noise)
    return torch.tensor(x, dtype=torch.float32)


@register("circles")
def circles(n, noise=0.05, factor=0.5):
    from sklearn.datasets import make_circles
    x, _ = make_circles(n, noise=noise, factor=factor)
    return torch.tensor(x, dtype=torch.float32)


@register("spiral")
def spiral(n, noise=0.05, turns=2.0):
    t = 0.25 + 0.75 * torch.rand(n)
    theta = t * turns * 2 * math.pi
    r = 2.0 * t
    x = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
    return x + noise * torch.randn(n, 2)


@register("eight_gaussians")
def eight_gaussians(n, radius=2.0, std=0.1):
    angles = torch.arange(8) * (2 * math.pi / 8)
    centers = radius * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    idx = torch.randint(0, 8, (n,))
    return centers[idx] + std * torch.randn(n, 2)


@register("gaussian_shift")
def gaussian_shift(n, mu=(2.0, 0.0)):
    return torch.randn(n, 2) + torch.tensor(mu)


def list_datasets():
    return sorted(DATASETS)


def get_dataset(name, **kwargs):
    """Return a sampler sample_data(n) for the named dataset."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Options: {list_datasets()}")
    return lambda n: DATASETS[name](n, **kwargs)
