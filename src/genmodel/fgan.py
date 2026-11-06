import torch
import torch.nn.functional as F
from torch import nn

from .gpa import LipschitzNet, make_L_schedule

DIVERGENCES = {
    "kl": {
        "g": lambda v: v,
        "f_star": lambda t: torch.exp(t - 1.0),
    },
    "gan": {
        "g": lambda v: F.logsigmoid(v),  
        "f_star": lambda t: -torch.log1p(-torch.exp(t).clamp(max=1 - 1e-6)),
    },
}


class Generator(nn.Module):
    
    def __init__(self, latent_dim, out_dim, width=64, depth=4):
        super().__init__()
        assert latent_dim == out_dim, "residual generator needs equal dims"
        dims = [latent_dim] + [width] * (depth - 1) + [out_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(depth)
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, z):
        h = z
        for layer in self.layers[:-1]:
            h = F.relu(layer(h))
        return z + self.layers[-1](h)

class Discriminator(LipschitzNet):

    def __init__(self, dim, width=64, depth=4, L=1.0, use_projection=True):
        self.use_projection = use_projection
        super().__init__(dim, width, depth, L)

    @torch.no_grad()
    def project(self):
        if not self.use_projection:
            return
        for layer in self.layers:
            s = torch.linalg.matrix_norm(layer.weight, ord=2)
            layer.weight.mul_(self.L ** (1.0 / self.D) / s)


DIVERGENCES = {
    "kl": {
        "g": lambda v: v,
        "f_star": lambda t: torch.exp(t - 1.0),
        "non_saturating": False,
    },
    "gan": {
        "g": lambda v: F.logsigmoid(v),
        # f*(t) = -log(1 - exp(t)); expm1 avoids the clamp and stays accurate near t = 0
        "f_star": lambda t: -torch.log(-torch.expm1(t)),
        "non_saturating": True,
    },
}


def _objective(D, v_real, v_fake, div):
    g, f_star = div["g"], div["f_star"]
    return g(D(v_real)).mean() - f_star(g(D(v_fake))).mean()


def disc_loss(D, v_real, v_fake, div):
    return -_objective(D, v_real, v_fake, div)


def gen_loss(D, v_real, v_fake, div):
    if div["non_saturating"]:
        return -div["g"](D(v_fake)).mean()
    return _objective(D, v_real, v_fake, div)


def train_fgan(sample_prior, sample_data, d, L_values, L_steps,
               divergence="kl", use_projection=True, M=1000, N=1000,
               k_disc=5, lr_g=1e-4, lr_d=1e-4, width=64, depth=4,
               device="cpu", callback=None):
    if divergence not in DIVERGENCES:
        raise ValueError(f"divergence must be one of {list(DIVERGENCES)}")
    div = DIVERGENCES[divergence]
    schedule, n_steps = make_L_schedule(L_values, L_steps, 1.0)  # dt unused

    G = Generator(d, d, width, depth).to(device)
    D = Discriminator(d, width, depth, L_values[0], use_projection).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    z_fixed = sample_prior(M).to(device)

    for n in range(n_steps):
        L_n, _ = schedule(n)
        if L_n != D.L:
            D.L = L_n
            D.project()          # rescale to the new L (no-op without projection)

        for _ in range(k_disc):
            real = sample_data(N).to(device)
            with torch.no_grad():
                fake = G(sample_prior(N).to(device))
            loss_d = disc_loss(D, real, fake, div)
            if not torch.isfinite(loss_d):
                raise RuntimeError(
                    f"Discriminator loss became {loss_d.item()} at step {n}. "
                    "Try a moderate L, --divergence gan, or a lower learning rate.")
            opt_d.zero_grad()
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 10.0)
            opt_d.step()
            D.project()

        fake = G(sample_prior(N).to(device))
        loss_g = gen_loss(D, real, fake, div)
        opt_g.zero_grad()
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), 10.0)
        opt_g.step()

        if callback is not None:
            with torch.no_grad():
                callback(n, G(z_fixed).cpu(), -loss_d.item(), L_n)

    with torch.no_grad():
        return G(z_fixed).cpu()
