
import os, numpy as np, jax, jax.numpy as jnp
from jax import jit
from jax.example_libraries import stax, optimizers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import CKPT_LAST, CKPT_BEST, FIG_DIR, LOG_DIR
from .utils import save_params
from .dataio import sample_at_seconds
from .networks import init_psi, init_coef
from .dynamics import f_dynamics, rollout_model
from .losses import loss_derivative, loss_timestep

def grad_global_norm(grad_tree) -> float:
    leaves = jax.tree_util.tree_leaves(grad_tree)
    sq = 0.0
    for leaf in leaves:
        arr = np.array(leaf); sq += float((arr * arr).sum())
    return float(np.sqrt(max(sq, 0.0)))

def evaluate_sec_mse(params, ds: dict, idx_list):
    t_hr = ds["t_hr"].astype(np.float32)
    dt_sub = float(ds["dt_sub"])
    seconds = float(ds["seconds"])
    sec_grid = np.arange(0, int(seconds) + 1, dtype=np.float32)

    @jit
    def _roll_pred(s0_, t_):
        return rollout_model(params, s0_, t_)
    _roll_pred = jit(_roll_pred)

    mse_all_list, mse_pos_list, mse_vel_list = [], [], []
    for idx in idx_list:
        x0 = ds["x0_list"][idx].astype(np.float32)
        true_hr = ds["true_hr_list"][idx].astype(np.float32)
        pred_hr = np.array(jax.device_get(_roll_pred(jnp.array(x0), jnp.array(t_hr))))
        true_sec = sample_at_seconds(true_hr, dt_sub, sec_grid)
        pred_sec = sample_at_seconds(pred_hr, dt_sub, sec_grid)
        mse_all = float(np.mean((pred_sec - true_sec) ** 2))
        mse_pos = float(np.mean((pred_sec[:, :2] - true_sec[:, :2]) ** 2))
        mse_vel = float(np.mean((pred_sec[:, 2:] - true_sec[:, 2:]) ** 2))
        mse_all_list.append(mse_all); mse_pos_list.append(mse_pos); mse_vel_list.append(mse_vel)
    return {"mse_all": float(np.mean(mse_all_list)), "mse_pos": float(np.mean(mse_pos_list)), "mse_vel": float(np.mean(mse_vel_list))}

def train_model(
    train, test, dt, ds,
    train_traj_idx,
    num_steps=18000,
    mode="derivative",
    eval_every: int = 200,
    eval_sec_subset: int = 5,
    lambda_flow_smooth: float = 0.0,
):
    # Fixed test eval subset for sec-level MSE
    if train_traj_idx is not None:
        # infer test idx from dataset split
        all_idx = set(range(ds["x0_list"].shape[0]))
        test_traj_idx = list(all_idx - set(train_traj_idx))
        test_eval_indices = list(test_traj_idx[: min(eval_sec_subset, len(test_traj_idx))])
    else:
        test_eval_indices = list(range(min(eval_sec_subset, ds["x0_list"].shape[0])))

    # Init networks
    rng = jax.random.PRNGKey(0)
    _, psi_params = init_psi(rng, (-1, 2))
    _, coef_params = init_coef(rng, (-1, 3))
    params = {"psi": psi_params, "coef": coef_params}

    # Optimizer schedule
    lr_schedule = lambda t: jnp.select([t < 8000, t < 15000], [1e-3, 3e-4], 1e-4)
    opt_init, opt_update, get_params = optimizers.adam(lr_schedule)
    opt_state = opt_init(params)

    # Loss selector
    if mode == "derivative":
        def compute_loss(p, batch):
            return loss_derivative(p, batch, lambda_flow_smooth=lambda_flow_smooth)
        train_batch = train; test_batch = test
    else:
        S_tr, dS_tr = train; S_te, dS_te = test
        train_batch = (S_tr, S_tr + dS_tr * dt)
        test_batch  = (S_te, S_te + dS_te * dt)
        def compute_loss(p, batch):
            return loss_timestep(p, batch, dt, lambda_flow_smooth=lambda_flow_smooth)

    @jit
    def step(i, opt_state, batch):
        p = get_params(opt_state)
        g = jax.grad(compute_loss)(p, batch)
        return opt_update(i, g, opt_state)

    os.makedirs(LOG_DIR, exist_ok=True)
    open(os.path.join(LOG_DIR, "train_log.txt"), "w").write("iter,lr,train_loss,test_loss,grad_norm,mse_all,mse_pos,mse_vel\n")
    train_losses, test_losses = [], []
    mse_all_hist, mse_pos_hist, mse_vel_hist = [], [], []
    best_params = None; best_test = 1e30

    for i in range(num_steps + 1):
        opt_state = step(i, opt_state, train_batch)
        if i % eval_every == 0:
            p = get_params(opt_state)
            lr = float(lr_schedule(i))
            tr = float(compute_loss(p, train_batch))
            te = float(compute_loss(p, test_batch))
            g = jax.grad(compute_loss)(p, train_batch)
            gnorm = float(jnp.sqrt(sum([jnp.sum(leaf*leaf) for leaf in jax.tree_util.tree_leaves(g)])))
            eval_sec = evaluate_sec_mse(p, ds, test_eval_indices)
            mse_all, mse_pos, mse_vel = eval_sec["mse_all"], eval_sec["mse_pos"], eval_sec["mse_vel"]
            train_losses.append(tr); test_losses.append(te)
            mse_all_hist.append(mse_all); mse_pos_hist.append(mse_pos); mse_vel_hist.append(mse_vel)
            if te < best_test: best_test, best_params = te, p
            print(f"iter={i:5d}  lr={lr:.1e}  train={tr:.6e}  test={te:.6e}  | grad={gnorm:.3e}  secMSE_all={mse_all:.3e} pos={mse_pos:.3e} vel={mse_vel:.3e}")
            with open(os.path.join(LOG_DIR, "train_log.txt"), "a") as f:
                f.write(f"{i},{lr:.3e},{tr:.6e},{te:.6e},{gnorm:.6e},{mse_all:.6e},{mse_pos:.6e},{mse_vel:.6e}\n")

    p_last = get_params(opt_state)
    if best_params is None: best_params = p_last
    save_params(p_last, CKPT_LAST)
    save_params(best_params, CKPT_BEST)

    # Plots
    os.makedirs(FIG_DIR, exist_ok=True)
    steps_axis = np.arange(0, num_steps + 1, eval_every)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7.5, 3))
    plt.plot(steps_axis, train_losses, label="train_loss")
    plt.plot(steps_axis, test_losses, label="test_loss")
    plt.yscale("log"); plt.xlabel(f"iterations (every {eval_every})"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "loss_curve.png"), dpi=150)

    plt.figure(figsize=(7.5, 3))
    plt.plot(steps_axis, mse_all_hist, label="sec MSE (all)")
    plt.plot(steps_axis, mse_pos_hist, label="sec MSE (pos)")
    plt.plot(steps_axis, mse_vel_hist, label="sec MSE (vel)")
    plt.yscale("log"); plt.xlabel(f"iterations (every {eval_every})"); plt.ylabel("sec-level MSE"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "sec_mse_curve.png"), dpi=150)

    return best_params
