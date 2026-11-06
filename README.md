# Lipschigz Reularized Generative Modeling

This repository contains a Python implementation of **GPA**, a particle-transport generative algorithm, together with an **f-GAN** baseline, and the code to compare them on 2D datasets.

- **GPA**: particles start from a prior (a Gaussian) and are moved toward the data along the gradient of an L-Lipschitz critic. The critic is trained to estimate an f-divergence between the particles and the data, and the Lipschitz constant `L` follows a piecewise-constant schedule that you define. The step size is `dt = dt0 / L`.
- **f-GAN**: a generator network is trained against a discriminator with the same architecture and the same Lipschitz projection, so the two methods can be compared fairly. The `L` schedule works the same way as in GPA.

<!-- Put your best result here, for example a GIF of GPA on eight_gaussians. -->
<!-- ![hero](docs/images/hero.gif) -->

## Contents

1. [Installation](#installation)
2. [Quickstart](#quickstart)
3. [Datasets](#datasets)
4. [Running GPA](#running-gpa)
5. [Running the f-GAN](#running-the-f-gan)
6. [Command-line reference](#command-line-reference)
7. [Output files](#output-files)
8. [Experiments and results](#experiments-and-results)
9. [Measuring computational cost](#measuring-computational-cost)
10. [Repository structure](#repository-structure)
11. [Known limitations](#known-limitations)

## Installation

You need [uv](https://docs.astral.sh/uv/) and git.

```bash
git clone https://github.com/YOUR_USERNAME/my-genmodel.git
cd my-genmodel
uv sync
```

`uv sync` installs the exact versions in `uv.lock`, so everyone gets the same environment.

Check that the install works:

```bash
uv run python -c "from genmodel.gpa import gpa; from genmodel.fgan import train_fgan; print('ok')"
```

**Tested with:** _(fill in: OS, CPU, Python version, torch version)_.

**Intel Macs:** recent PyTorch releases no longer ship Intel macOS builds. If installation fails, pin an older release: `uv add "torch==2.2.2" "numpy<2"`.

**Videos:** MP4 output uses the ffmpeg binary bundled with the `imageio-ffmpeg` package, so you do not need to install ffmpeg yourself.

## Quickstart

A short GPA run on the moons dataset, with a video:

```bash
uv run python scripts/run_all.py --datasets eight_gaussians --L-values 1 2 5 --L-steps 50 50 50
```

This prints the progress, the training time and the final distance to the data, and saves the video to `assets/moons/L1-2-5_steps100-50-50.mp4`. Open it with `open assets/moons/L1-2-5_steps100-50-50.mp4` (macOS) or any video player.

A short f-GAN run on the same data:

```bash
uv run python scripts/run_all.py --model fgan --datasets moons --divergence gan \
    --L-values 1 5 --L-steps 1500 1500
```

## Datasets

All current datasets are 2D and need no download. Each one is a function in `src/genmodel/datasets.py`, registered under a name.

| Name | Description |
|---|---|
| `moons` | Two interlocking half circles (scikit-learn) |
| `circles` | Two concentric rings (scikit-learn) |
| `spiral` | A noisy spiral arm |
| `eight_gaussians` | Eight small Gaussians on a ring of radius 2 |
| `gaussian_shift` | A unit Gaussian centered at (2, 0) |

Choose one or more with `--datasets`:

```bash
uv run python scripts/run_all.py --datasets spiral eight_gaussians --L-values 1 2 5 --L-steps 50 50 50
```

**Adding your own dataset:** add a function to `datasets.py`. It must take `n` and return a float32 tensor of shape `(n, 2)`.

```python
@register("my_dataset")
def my_dataset(n, some_param=1.0):
    return ...   # tensor of shape (n, 2)
```

It then appears in `--datasets` automatically. 

## Running GPA

GPA is the default model. You define the **L schedule** with two lists of the same length: the value of `L` in each stage, and the number of outer steps spent in that stage.

```bash
uv run python scripts/run_all.py --datasets eight_gaussians \
    --L-values 1 2 5 --L-steps 50 50 50
```

This uses L=1 for 50 steps, L=2 for 50 steps and L=5 for 50 steps. The step size in each stage is `dt = dt0 / L`, where `dt0` (default 0.25) is the step size at `L = 1`. The largest possible movement per step, `dt × L`, is therefore `dt0` in every stage.

A constant `L` is a single stage:

```bash
uv run python scripts/run_all.py --datasets moons --L-values 5 --L-steps 200
```

## Running the f-GAN

To run the f-GAN one of the following loss must be chosen:

- Original GAN loss: `--divergence gan --no-projection`
- KL: `-- divergence kl --no-projection`
- (KL,Lipschitz-L): `--divergence KL --L (value of L)`

When using (KL,Lipschitz-L)-divergence, one can also choose adaptive L. Here are some exmaples:

```bash
#Gan loss
uv run python scripts/run_all.py --model fgan --datasets eight_gaussians --divergence gan \
    --no-projection --n-steps 3000
```

```bash
#Constant L
uv run python scripts/run_all.py --model fgan --datasets eight_gaussians --divergence kl \
    --L 5 --n-steps 3000
```

```bash
#With an L schedule: one stage = L and a number of generator steps
uv run python scripts/run_all.py --model fgan --datasets eight_gaussians --divergence kl \
    --L-values 1 20 --L-steps 1500 1500
```

> **Notes:**
>
> - `--divergence` is `kl` or `gan`. Add more in the `DIVERGENCES` dictionary in `src/genmodel/fgan.py`.
> - One f-GAN step is `k_disc` discriminator updates followed by one generator update, so f-GANs need many more steps than GPA (thousands, not hundreds).
> - If you give `--L-values` and `--L-steps`, `--n-steps` and `--L` are ignored.
> - With `--no-projection`, `L` has no effect. Prefer `--divergence gan` in that case, since KL can overflow without a constraint.

## Command-line reference

Run `uv run python scripts/run_all.py --help` for the full list.

| Flag | Applies to | Meaning | Default |
|---|---|---|---|
| `--model` | both | `gpa` or `fgan` | `gpa` |
| `--datasets` | both | One or more dataset names | all |
| `--L-values` | both | `L` of each stage | none |
| `--L-steps` | both | Steps in each stage | none |
| `--solver` | GPA | `euler` or `heun` | `euler` |
| `--dt0` | GPA | Step size at `L = 1`; `dt = dt0 / L` | 0.25 |
| `--divergence` | f-GAN | `kl` or `gan` | `kl` |
| `--n-steps` | f-GAN | Generator steps when no schedule is given | 200 |
| `--L` | f-GAN | Constant `L` when no schedule is given | 1.0 |
| `--no-projection` | f-GAN | Unconstrained discriminator | off |
| `--prior-mean` | both | Mean of the Gaussian prior, one number per dimension | 0 |
| `--M` | both | Number of particles | 1000 |
| `--seed` | both | Random seed | 0 |
| `--frame-every` | both | One video frame every this many steps | 25 |
| `--print-every` | both | Print the divergence estimate every this many steps | 25 |
| `--no-video` | both | Skip the video and only print and save numbers | off |

If your copy of `run_all.py` also has `--lr-g` and `--lr-d`, they set the f-GAN learning rates.

## Output files

- `assets/<dataset>/<run name>.mp4`: video of the particles (blue) moving onto samples of the data (gray). The title shows the step and the current `L`.
- `assets/results.csv`: one row per run with the dataset, model, schedule, solver, sliced Wasserstein distance (`sw`), the `floor`, and the training time.
- The terminal shows the **divergence estimate** every `--print-every` steps. It tracks the stage of training and is noisy, so look at the trend.

### How to read the numbers

```
training time = 42.3 s
SW distance = 0.2310   (data vs fresh data: 0.1149)
```

- **`SW distance`** is the sliced Wasserstein distance between the final particles and a fresh sample from the dataset. Smaller is better.
- **`data vs fresh data`** (the `floor`) is the same distance between two independent samples of the data itself. It is the best score any model can realistically reach, because of sampling noise.
- Judge a run by `SW distance / floor`. A value near 1 means the particles are indistinguishable from the data in this metric.

## Experiments and results

All experiments use the `eight_gaussians` dataset.

<table>
  <tr>
    <td align="center" width="33%">
      <img src="docs/images/fig_1.png" width="100%" alt="Figure 1"><br>
      <sub><b>Figure 1.</b> Caption for the first image.</sub>
    </td>
    <td align="center" width="33%">
      <img src="docs/images/fig_2.png" width="100%" alt="Figure 2"><br>
      <sub><b>Figure 2.</b> Caption for the second image.</sub>
    </td>
    <td align="center" width="33%">
      <img src="docs/images/fig_3.png" width="100%" alt="Figure 3"><br>
      <sub><b>Figure 3.</b> Caption for the third image.</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <img src="docs/images/fig_4.png" width="100%" alt="Figure 4"><br>
      <sub><b>Figure 4.</b> Caption for the fourth image.</sub>
    </td>
    <td align="center" width="33%">
      <img src="docs/images/fig_5.png" width="100%" alt="Figure 5"><br>
      <sub><b>Figure 5.</b> Caption for the fifth image.</sub>
    </td>
  </tr>
</table>

### How to reproduce each figure

**Figure 1** (_method and short description_)

```bash
 uv run python scripts/run_all.py --model fgan --datasets eight_gaussians --divergence gan  --prior-mean 4 4 --no-projection -n-steps 12000
```

**Figure 2** (_method and short description_)

```bash
# command for figure 2
```

**Figure 3** (_method and short description_)

```bash
# command for figure 3
```

**Figure 4** (_method and short description_)

```bash
# command for figure 4
```

**Figure 5** (_method and short description_)

```bash
# command for figure 5
```


### Comments

_Write your comments on the six figures here. What do the particles do? Do they cover all eight modes? How do the runs differ, and why do you think so?_

| Figure | Number of steps | SW distance | SW / floor | Training time |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |
| 6 | | | | |










## Measuring computational cost

A GPA step and an f-GAN step do different amounts of work, so compare **quality against wall-clock time**, not step counts.

With the default settings (`M = N = 1000`, `m_max = 20`, `k_disc = 5`):

| | GPA outer step | f-GAN step |
|---|---|---|
| Critic or discriminator updates | 20 | 5 |
| Generator updates | 0 | 1 |
| Input-gradient evaluations on the particles | 1 (Euler) or 2 (Heun) | 0 |
| `project()` calls | 20 | 5 |
| Networks trained | 1 | 2 |

Every run prints its training time. The time covers training only, not the video or the evaluation.

For a fair comparison, use the same machine, dataset, `M`, `N` and architecture for both methods, run several seeds, tune both methods with the same effort, and report the time to reach a target such as `SW ≤ 2 × floor`.

_Add your measured table here: hardware, number of seeds, and the median time and final `SW / floor` per method and dataset._

## Repository structure

```
my-genmodel/
├── pyproject.toml, uv.lock       # dependencies (exact versions in the lockfile)
├── README.md
├── src/genmodel/
│   ├── gpa.py                    # GPA algorithm, Lipschitz critic, L schedule, ODE solvers
│   ├── fgan.py                   # f-GAN: generator, discriminator, divergences, training loop
│   ├── datasets.py               # dataset registry
│   ├── experiments.py            # run_experiment: runs either model on a dataset
│   ├── metrics.py                # sliced Wasserstein distance
│   └── viz.py                    # particle videos
├── scripts/
│   └── run_all.py                # command-line entry point
├── assets/                       # generated videos and results.csv (ignored by git)
└── docs/images/                  # figures shown in this README
```

## Known limitations

- Only 2D vector datasets are supported so far. Image datasets (such as MNIST) need convolutional critics with a correct Lipschitz projection, which is not implemented yet.
- The Lipschitz projection uses an SVD on every layer after every update, which is the main cost of both methods. Power iteration would be faster.
- f-GAN results are sensitive to `L`, the learning rates and `k_disc`. Compare the two methods with several seeds.
- There is no automated test suite yet.

## License

_Add a license (for example MIT) and a `LICENSE` file._

## Citation

_If this accompanies a paper, add the reference here._
