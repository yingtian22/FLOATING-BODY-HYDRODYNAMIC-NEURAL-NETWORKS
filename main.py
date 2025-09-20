
"""Entry point that wires the modularized HNN together."""
import os, time, random
from fb_hnn import config, dataio, training, viz, debugtools
from fb_hnn.evaluate import compare_on_dataset
from fb_hnn.config import CKPT_BEST, GIF_DIR

def main():
    # 1) Data
    ds = dataio.load_or_create_dataset(config.DATA_PATH, seconds=8.0, dt_sub=0.02, size=64, seed=0)
    train_idx, test_idx = dataio.split_by_trajectory(ds, train_frac=0.8, seed=0)
    (S_tr, dS_tr), (S_te, dS_te), dt = dataio.build_dataset_from_npz(ds, train_traj_idx=train_idx, test_traj_idx=test_idx)

    # 2) Train
    start = time.time()
    best_params = training.train_model(
        train=(S_tr, dS_tr),
        test=(S_te, dS_te),
        dt=dt,
        ds=ds,
        train_traj_idx=train_idx,
        num_steps=18000,
        mode="derivative",
        eval_every=200,
        eval_sec_subset=5,
        lambda_flow_smooth=1e-4,
    )
    print(f"Training took {time.time() - start:.1f}s")

    # 3) Visualize flow field
    viz.visualize_flow_field(best_params, xlim=(-4, 4), ylim=(-4, 4), grid_n=25, prefix="best", compare_true=True)

    # 4) GIF comparisons on a few test trajectories
    random.seed(0)
    picks = random.sample(list(test_idx), k=min(3, len(test_idx)))
    os.makedirs(GIF_DIR, exist_ok=True)
    for idx in picks:
        out_gif = os.path.join(GIF_DIR, f"fb_true_vs_pred_idx{idx}.gif")
        metrics = compare_on_dataset(CKPT_BEST, ds, idx, out_gif=out_gif)
        print(f"[idx={idx}] x0={metrics['x0']} | MSE(all)={metrics['mse_all']:.6e} | pos={metrics['mse_pos']:.6e} | vel={metrics['mse_vel']:.6e}")

    # 5) Force decomposition on one test state
    if len(test_idx) > 0:
        idx_dbg = list(test_idx)[0]
        s0_dbg = ds["true_hr_list"][idx_dbg][0]
        print("\n[Decomposition on TEST sample] traj_idx=", idx_dbg, ", t=0")
        dbg = debugtools.debug_decompose_forces(best_params, s0_dbg, t=0.0)
        debugtools.print_debug_decompose(dbg)

    print("[saved] figures/loss_curve.png, figures/sec_mse_curve.png, "
          "figures/flow_quiver_best.png, figures/flow_stream_best.png, "
          "figures/flow_quiver_true.png, figures/flow_stream_true.png, "
          "logs/train_log.txt, data/split_idx_train.npy, data/split_idx_test.npy and gifs/*.gif")

if __name__ == "__main__":
    main()
