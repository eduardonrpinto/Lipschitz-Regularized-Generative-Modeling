import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter,FFMpegWriter
import imageio_ffmpeg


matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

def save_particle_video(frames, Ls, target, path, title="", fps=15, frame_every=25):
    idx = list(range(0, len(frames), frame_every))
    if idx[-1] != len(frames) - 1:
        idx.append(len(frames) - 1) 
    target = target.numpy()
    lo = torch.minimum(torch.as_tensor(target).min(0).values, frames[0].min(0).values)
    hi = torch.maximum(torch.as_tensor(target).max(0).values, frames[0].max(0).values)
    pad = 0.5
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(target[:, 0], target[:, 1], s=2, c="lightgray")
    scat = ax.scatter([], [], s=3, c="C0")
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal")

    def update(k):
        i = idx[k]
        scat.set_offsets(frames[i].numpy())
        L_txt = f"  L={Ls[i]:g}" if Ls[i] is not None else "  no L"
        ax.set_title(f"{title}  step {i}{L_txt}", fontsize=8)
        return (scat,)

    anim = FuncAnimation(fig, update, frames=len(idx))
    if path.endswith(".mp4"):
        writer = FFMpegWriter(fps=fps, extra_args=["-pix_fmt", "yuv420p"])
    else:
        writer = PillowWriter(fps=fps)
    anim.save(path, writer=writer)
    plt.close(fig)


def save_image_grid(Y, path, n=16, title=""):
    n = min(n, len(Y))
    cols = int(n ** 0.5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, img in zip(axes.ravel(), Y[:n]):
        img = img.detach().cpu().clamp(0, 1)
        if img.shape[0] == 1:
            ax.imshow(img[0], cmap="gray")
        else:
            ax.imshow(img.permute(1, 2, 0))
    fig.suptitle(title, fontsize=8)
    fig.savefig(path, dpi=120)
    plt.close(fig)
