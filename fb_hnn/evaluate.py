
import os, numpy as np, jax, jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from jax import jit
from .config import GIF_DIR, FIG_DIR, CKPT_BEST
from .utils import progress_bar, load_params
from .dataio import sample_at_seconds
from .dynamics import rollout_model

def render_gif_comparison(true_hr: np.ndarray, pred_hr: np.ndarray, out_path: str, fps: int = 25, dpi: int = 100, xy_lim: float = 4.0):
    assert true_hr.shape[0] == pred_hr.shape[0]
    total = true_hr.shape[0]
    os.makedirs(GIF_DIR, exist_ok=True)
    writer = imageio.get_writer(out_path, mode="I", fps=fps)
    fig = plt.figure(figsize=(8, 3.5), dpi=dpi)
    axL = fig.add_subplot(1, 2, 1); axR = fig.add_subplot(1, 2, 2)
    try:
        for i in range(total):
            if i == 0: print(f"[gif] render & write -> {out_path}")
            progress_bar(i + 1, total, prefix="[gif]")
            for ax, title in [(axL, "True (high-res)"), (axR, "Predicted (high-res)")]:
                ax.cla(); ax.set_title(title, fontsize=10)
                ax.set_xlim(-xy_lim, xy_lim); ax.set_ylim(-xy_lim, xy_lim)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
            axL.plot(true_hr[: i + 1, 0], true_hr[: i + 1, 1], "-", lw=2); axL.plot(true_hr[i, 0], true_hr[i, 1], "o")
            axR.plot(pred_hr[: i + 1, 0], pred_hr[: i + 1, 1], "-", lw=2); axR.plot(pred_hr[i, 0], pred_hr[i, 1], "o")
            fig.canvas.draw(); buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(); writer.append_data(buf)
    finally:
        writer.close(); plt.close(fig)
    print(f"✅ Saved GIF -> {out_path}")

def compare_on_dataset(params_path: str, ds: dict, idx: int, out_gif: str | None = None, fps: int | None = None):
    t_hr = ds["t_hr"].astype(np.float32)
    x0 = ds["x0_list"][idx].astype(np.float32)
    true_hr = ds["true_hr_list"][idx].astype(np.float32)
    seconds = float(ds["seconds"]); dt_sub = float(ds["dt_sub"])
    params = load_params(params_path)

    @jit
    def _roll_pred(s0_, t_):
        return rollout_model(params, s0_, t_)

    pred_hr = np.array(jax.device_get(_roll_pred(jnp.array(x0), jnp.array(t_hr))))
    sec_grid = np.arange(0, int(seconds) + 1, dtype=np.float32)
    true_sec = sample_at_seconds(true_hr, dt_sub, sec_grid)
    pred_sec = sample_at_seconds(pred_hr, dt_sub, sec_grid)

    if out_gif is not None:
        fps_val = fps if fps is not None else int(round(1.0 / dt_sub))
        render_gif_comparison(true_hr, pred_hr, out_gif, fps=fps_val)

    mse_all = float(np.mean((pred_sec - true_sec) ** 2))
    mse_pos = float(np.mean((pred_sec[:, :2] - true_sec[:, :2]) ** 2))
    mse_vel = float(np.mean((pred_sec[:, 2:] - true_sec[:, 2:]) ** 2))
    return {"x0": x0, "mse_all": mse_all, "mse_pos": mse_pos, "mse_vel": mse_vel, "true_sec": true_sec, "pred_sec": pred_sec}
