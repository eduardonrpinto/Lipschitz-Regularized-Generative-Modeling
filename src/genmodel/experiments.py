import torch
from .datasets import get_dataset
from .fgan import train_fgan
from .gpa import LipschitzNet, gpa

def run_experiment(dataset, schedule=None, model="gpa", solver="euler", seed=0,
                   M=2000, N=2000, m_max=20, scale=1.0, dt0=0.25,
                   prior_mean=None, print_every=25,
                   divergence="kl", n_steps=200, L=1.0, use_projection=True):
    torch.manual_seed(seed)
    sample_data = get_dataset(dataset)
    shape = tuple(sample_data(1).shape[1:])

    if prior_mean is None:
        mean = torch.zeros(shape)
    else:
        mean = torch.tensor(prior_mean, dtype=torch.float32)
    sample_prior = lambda n: mean + torch.randn(n, *shape)
    frames, Ls = [], []

    if model == "gpa":
        sched = schedule
        steps = [max(1, round(s * scale)) for s in sched["L_steps"]]
        total = sum(steps)

        def callback(n, Y, obj, L_n, dt):
            frames.append(Y.clone())
            Ls.append(L_n)
            if print_every and (n % print_every == 0 or n == total - 1):
                print(f"  step {n:4d}/{total}  L={L_n:g}  dt={dt:g}  "
                      f"divergence estimate = {obj:.4f}", flush=True)

        Y = gpa(
            sample_prior=sample_prior,
            sample_data=sample_data,
            make_critic=lambda L_: LipschitzNet(shape[0], width=64, depth=4, L=L_),
            L_values=sched["L_values"], L_steps=steps, dt0=dt0,
            M=M, N=N, m_max=m_max, solver=solver, callback=callback,
        )

    elif model == "fgan":
        if schedule is None:
            sched = dict(L_values=[L], L_steps=[n_steps])
        else:
            sched = schedule
        steps = [max(1, round(s * scale)) for s in sched["L_steps"]]
        total = sum(steps)

        def callback(n, Y, obj, L_n):
            frames.append(Y.clone())
            Ls.append(L_n if use_projection else None)
            if print_every and (n % print_every == 0 or n == total - 1):
                L_txt = f"L={L_n:g}" if use_projection else "no L"
                print(f"  step {n:4d}/{total}  {L_txt}  "
                      f"divergence estimate = {obj:.4f}", flush=True)

        Y = train_fgan(
            sample_prior=sample_prior, sample_data=sample_data,
            d=shape[0], L_values=sched["L_values"], L_steps=steps,
            divergence=divergence, use_projection=use_projection,
            M=M, N=N, callback=callback,
        )

    else:
        raise ValueError(f"model must be 'gpa' or 'fgan', got {model!r}")

    return Y, frames, Ls, sample_data
 