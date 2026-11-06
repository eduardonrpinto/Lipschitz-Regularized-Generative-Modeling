import argparse
import csv
import os
import time

from genmodel.datasets import list_datasets
from genmodel.experiments import run_experiment
from genmodel.fgan import DIVERGENCES
from genmodel.metrics import sliced_wasserstein
from genmodel.viz import save_particle_video

p = argparse.ArgumentParser()
p.add_argument("--model", default="gpa", choices=["gpa", "fgan"])
p.add_argument("--datasets", nargs="+", default=list_datasets(), choices=list_datasets())

# GPA options
p.add_argument("--L-values", type=float, nargs="+", default=None,
               help="GPA custom schedule: the L of each stage, e.g. 1 2 5")
p.add_argument("--L-steps", type=int, nargs="+", default=None,
               help="GPA custom schedule: outer steps of each stage, e.g. 100 50 50")
p.add_argument("--solver", default="euler", choices=["euler", "heun"])
p.add_argument("--dt0", type=float, default=0.25, help="GPA: step size at L=1; dt = dt0 / L")

# f-GAN options
p.add_argument("--divergence", default="kl", choices=list(DIVERGENCES), help="f-GAN only")
p.add_argument("--n-steps", type=int, default=200, help="f-GAN generator steps")
p.add_argument("--L", type=float, default=1.0, help="f-GAN discriminator Lipschitz constant")
p.add_argument("--no-projection", action="store_true", help="f-GAN: unconstrained discriminator")

# common options
p.add_argument("--seed", type=int, default=0)
p.add_argument("--prior-mean", type=float, nargs="+", default=None)
p.add_argument("--M", type=int, default=2000)
p.add_argument("--frame-every", type=int, default=25)
p.add_argument("--print-every", type=int, default=25)
p.add_argument("--no-video", action="store_true")
args = p.parse_args()


custom = None
if (args.L_values is None) != (args.L_steps is None):
    p.error("--L-values and --L-steps must be given together")
if args.L_values is not None:
    if len(args.L_values) != len(args.L_steps):
        p.error(f"--L-values has {len(args.L_values)} entries but "
                f"--L-steps has {len(args.L_steps)}; they must match")
    if any(L <= 0 for L in args.L_values) or any(s <= 0 for s in args.L_steps):
        p.error("L values and step counts must be positive")
    custom = dict(L_values=args.L_values, L_steps=args.L_steps)
    sched_name = "L" + "-".join(f"{L:g}" for L in args.L_values) + \
                 "_steps" + "-".join(str(s) for s in args.L_steps)

if args.model == "fgan":
    if custom is not None:
        Lpart = "noproj" if args.no_projection else sched_name
        tag = f"fgan_{args.divergence}_{Lpart}"
        to_run = [(tag, custom)]
    else:
        Lpart = "noproj" if args.no_projection else f"L{args.L:g}"
        tag = f"fgan_{args.divergence}_{Lpart}_steps{args.n_steps}"
        to_run = [(tag, None)]
else:
    to_run = [(sched_name, custom)] if custom is not None else [(s, s) for s in args.schedules]

os.makedirs("assets", exist_ok=True)
rows = []

for dataset in args.datasets:
    os.makedirs(f"assets/{dataset}", exist_ok=True)
    for name, schedule in to_run:
        print(f"running {dataset} / {name} ...", flush=True)
        t0 = time.perf_counter()
        Y, frames, Ls, sample_data = run_experiment(
            dataset, schedule, model=args.model, solver=args.solver,
            seed=args.seed, dt0=args.dt0, prior_mean=args.prior_mean,
            M=args.M, print_every=args.print_every,
            divergence=args.divergence, n_steps=args.n_steps, L=args.L,
            use_projection=not args.no_projection)
        train_secs = time.perf_counter() - t0
        print(f"  training time = {train_secs:.1f} s", flush=True)

        sw = sliced_wasserstein(Y, sample_data(len(Y)))
        floor = sliced_wasserstein(sample_data(len(Y)), sample_data(len(Y)))
        print(f"  SW distance = {sw:.4f}   (data vs fresh data: {floor:.4f})")
        rows.append(dict(dataset=dataset, model=args.model, schedule=name,
                         solver=args.solver, sw=round(sw, 4), floor=round(floor, 4)))

        if not args.no_video:
            out = f"assets/{dataset}/{name}.mp4"
            save_particle_video(frames, Ls, sample_data(2000), out,
                                title=f"{dataset}/{name}",
                                frame_every=args.frame_every)
            print("  saved", out)

with open("assets/results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print("saved assets/results.csv")
