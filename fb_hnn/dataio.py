
import os, numpy as np, jax, jax.numpy as jnp
from typing import Sequence, Dict, Any, Tuple
from .config import DATA_PATH, TRAIN_IDX_PATH, TEST_IDX_PATH, TOL
from .utils import progress_bar
from .physics import rollout_true, make_synth_params

def make_highres_timegrid(seconds: float, dt_sub: float) -> np.ndarray:
    n = int(np.floor(seconds / dt_sub))
    t = np.arange(n + 1, dtype=np.float32) * dt_sub
    if t[-1] < seconds - 1e-12:
        t = np.append(t, np.float32(seconds))
    return t.astype(np.float32)

def sample_at_seconds(states_hr: np.ndarray, dt_sub: float, seconds_list) -> np.ndarray:
    idx = np.rint(np.array(seconds_list, dtype=np.float32) / dt_sub).astype(int)
    idx = np.clip(idx, 0, len(states_hr) - 1)
    return states_hr[idx]

def load_or_create_dataset(path: str = DATA_PATH, seconds: float = 8.0, dt_sub: float = 0.02, size: int = 64, seed: int = 0):
    if os.path.exists(path):
        ds = np.load(path)
        for k in ["seconds", "dt_sub", "t_hr", "x0_list", "true_hr_list"]:
            if k not in ds: raise ValueError(f"Dataset missing key `{k}` in {path}")
        print(f"✅ Loaded dataset from {path}")
        return {
            "seconds": float(ds["seconds"]),
            "dt_sub": float(ds["dt_sub"]),
            "t_hr": np.array(ds["t_hr"]),
            "x0_list": np.array(ds["x0_list"]),
            "true_hr_list": np.array(ds["true_hr_list"]),
        }
    print(f"⚙️  Dataset not found. Generating -> {path}")
    rng = np.random.RandomState(seed)
    t_hr = make_highres_timegrid(seconds, dt_sub).astype(np.float32)
    x0_list, true_list = [], []
    for i in range(size):
        x0 = np.array([
            rng.uniform(-2.5, 2.5),
            rng.uniform(-2.5, 2.5),
            rng.uniform(-0.9, 0.9),
            rng.uniform(-0.9, 0.9),
        ], dtype=np.float32)
        x0_list.append(x0)
        true_hr = np.array(jax.device_get(rollout_true(jnp.array(x0), jnp.array(t_hr))))
        true_list.append(true_hr)
        progress_bar(i + 1, size, prefix="[dataset]")
    x0_arr = np.stack(x0_list, axis=0).astype(np.float32)
    true_arr = np.stack(true_list, axis=0).astype(np.float32)
    np.savez_compressed(path, seconds=np.float32(seconds), dt_sub=np.float32(dt_sub), t_hr=t_hr, x0_list=x0_arr, true_hr_list=true_arr)
    print(f"✅ Saved dataset -> {path} | size={size}, T={t_hr.shape[0]}")
    return {"seconds": seconds, "dt_sub": dt_sub, "t_hr": t_hr, "x0_list": x0_arr, "true_hr_list": true_arr}

def _is_valid_split(train_idx: np.ndarray, test_idx: np.ndarray, M: int) -> bool:
    if train_idx.size == 0 or test_idx.size == 0: return False
    if train_idx.min() < 0 or test_idx.min() < 0: return False
    if train_idx.max() >= M or test_idx.max() >= M: return False
    if np.intersect1d(train_idx, test_idx).size > 0: return False
    return True

def split_by_trajectory(ds: dict, train_frac: float = 0.8, seed: int = 0):
    M = ds["x0_list"].shape[0]
    if os.path.exists(TRAIN_IDX_PATH) and os.path.exists(TEST_IDX_PATH):
        try:
            tr = np.load(TRAIN_IDX_PATH); te = np.load(TEST_IDX_PATH)
            if _is_valid_split(tr, te, M):
                print(f"✅ Loaded split indices from {TRAIN_IDX_PATH} & {TEST_IDX_PATH}")
                return tr.astype(int), te.astype(int)
            else:
                print("⚠️  Saved split indices invalid for current dataset. Regenerating...")
        except Exception as e:
            print(f"⚠️  Failed to load split indices ({e}). Regenerating...")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(M)
    n_train = int(round(train_frac * M))
    tr = perm[:n_train]; te = perm[n_train:]
    np.save(TRAIN_IDX_PATH, tr); np.save(TEST_IDX_PATH, te)
    print(f"💾 Saved split indices -> {TRAIN_IDX_PATH}, {TEST_IDX_PATH}")
    return tr, te

def build_dataset_from_npz(ds: dict, train_traj_idx: np.ndarray | None = None, test_traj_idx: np.ndarray | None = None):
    t_hr = ds["t_hr"].astype(np.float32)
    X = ds["true_hr_list"].astype(np.float32)
    M, T, _ = X.shape
    true_f = make_synth_params()
    dt = float(t_hr[1] - t_hr[0])

    def states_and_derivs_from_trajs(traj_idx):
        X_sel = X[traj_idx]
        S_all = X_sel.reshape(-1, 4)
        t_all = np.tile(t_hr, (len(traj_idx),))
        dS_all = np.array(jax.device_get(jax.vmap(true_f)(jnp.array(S_all), jnp.array(t_all))))
        perm = np.random.permutation(S_all.shape[0])
        return jnp.array(S_all[perm]), jnp.array(dS_all[perm])

    if train_traj_idx is not None and test_traj_idx is not None:
        S_tr, dS_tr = states_and_derivs_from_trajs(train_traj_idx)
        S_te, dS_te = states_and_derivs_from_trajs(test_traj_idx)
        return (S_tr, dS_tr), (S_te, dS_te), dt

    S_all = X.reshape(M * T, 4)
    t_all = np.tile(t_hr, (M,))
    dS_all = np.array(jax.device_get(jax.vmap(true_f)(jnp.array(S_all), jnp.array(t_all))))
    idx = np.random.permutation(S_all.shape[0])
    S_all = S_all[idx]; dS_all = dS_all[idx]
    n_train = int(0.8 * S_all.shape[0])
    return (jnp.array(S_all[:n_train]), jnp.array(dS_all[:n_train])), (jnp.array(S_all[n_train:]), jnp.array(dS_all[n_train:])), dt
