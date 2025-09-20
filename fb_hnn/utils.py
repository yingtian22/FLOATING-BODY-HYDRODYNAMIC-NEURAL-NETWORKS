
import sys, os, pickle, numpy as np, jax

def tree_to_numpy(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)

def save_params(params, path):
    with open(path, "wb") as f:
        pickle.dump(tree_to_numpy(params), f)
    print(f"[checkpoint] Saved -> {path}")

def load_params(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Params not found: {path}")
    with open(path, "rb") as f:
        params = pickle.load(f)
    print(f"✅ Loaded params from {path}")
    return params

def progress_bar(step, total, prefix="", length=28):
    step = min(step, total)
    filled = int(length * step / max(total, 1))
    bar = "#" * filled + "-" * (length - filled)
    pct = 100.0 * step / max(total, 1)
    sys.stdout.write(f"\r{prefix} |{bar}| {pct:6.2f}% ({step}/{total})")
    sys.stdout.flush()
    if step >= total:
        sys.stdout.write("\n")
