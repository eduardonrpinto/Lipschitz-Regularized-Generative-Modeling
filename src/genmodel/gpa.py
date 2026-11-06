import bisect
import itertools

import torch
from torch import nn



def get_f_star(name):
    if name == "kl":     
        return lambda y: torch.exp(y - 1.0)

class LipschitzNet(nn.Module):
    def __init__(self, d, width=64, depth=4, L=3.0):
        super().__init__()
        dims = [d] + [width] * (depth - 1) + [1]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(depth)
        )
        self.D, self.L = depth, L
        self.project()  

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = nn.functional.relu(layer(x))  
        return self.layers[-1](x).squeeze(-1)

    #Spectral Normalization
    @torch.no_grad()
    def project(self):
        for layer in self.layers:
            s = torch.linalg.matrix_norm(layer.weight, ord=2) 
            layer.weight.mul_(self.L ** (1.0 / self.D) / s) 


def make_L_schedule(L_values, L_steps, dt0):
    if len(L_values) != len(L_steps):
        raise ValueError(
            f"L_values and L_steps must have the same length, "
            f"got {len(L_values)} and {len(L_steps)}"
        )
    if len(L_values) == 0:
        raise ValueError("L_values must not be empty")
    if any(s <= 0 or int(s) != s for s in L_steps):
        raise ValueError(f"L_steps must be positive integers, got {L_steps}")
    if any(L <= 0 for L in L_values):
        raise ValueError(f"L_values must be positive, got {L_values}")

    L_steps = [int(s) for s in L_steps]
    boundaries = list(itertools.accumulate(L_steps))
    n_max = boundaries[-1]

    def schedule(n):
        stage = bisect.bisect_right(boundaries, n)
        stage = min(stage, len(L_values) - 1)
        L = L_values[stage]
        return L, dt0 / L

    return schedule, n_max

def _grad_phi(net, Y):
    Yg = Y.detach().clone().requires_grad_(True)
    return torch.autograd.grad(net(Yg).sum(), Yg)[0]

#ODE solvers 
def _step_euler(net, Y, dt):
    return Y - dt * _grad_phi(net, Y)


def _step_heun(net, Y, dt):
    g1 = _grad_phi(net, Y)
    Y_pred = Y - dt * g1
    g2 = _grad_phi(net, Y_pred)
    return Y - 0.5 * dt * (g1 + g2)


_SOLVERS = {"euler": _step_euler, "heun": _step_heun}

#GPA
def gpa(sample_prior, sample_data, make_critic, L_values, L_steps, f="kl",
        dt0=0.5, M=2000, N=2000, lr=1e-4, m_max=20,
        device="cpu", callback=None, solver="euler"):
    if solver not in _SOLVERS:
        raise ValueError(f"solver must be one of {list(_SOLVERS)}, got {solver!r}")
    ode_step = _SOLVERS[solver]

    f_star = get_f_star(f)
    schedule, n_max = make_L_schedule(L_values, L_steps, dt0)

    Y = sample_prior(M).to(device)                            
    nu = nn.Parameter(torch.zeros((), device=device))         
    net = make_critic(L_values[0]).to(device)  # line 4
    opt = torch.optim.Adam(list(net.parameters()) + [nu], lr=lr)

    for n in range(n_max):                                    
        L_n, dt_n = schedule(n)
        if L_n != net.L:
            net.L = L_n
            net.project()

        for _ in range(m_max):                                
            X = sample_data(N).to(device)                     
            obj = net(Y).mean() - nu - f_star(net(X) - nu).mean()   
            opt.zero_grad()
            (-obj).backward()
            opt.step()                                        
            net.project()                                     

        Y = ode_step(net, Y, dt_n).detach()

        if callback is not None:
            callback(n, Y, obj.item(), L_n, dt_n)
    return Y