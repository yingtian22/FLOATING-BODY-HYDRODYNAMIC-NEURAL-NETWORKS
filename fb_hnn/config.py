
import os

# Paths & Dirs
DATA_DIR = "data"; os.makedirs(DATA_DIR, exist_ok=True)
FIG_DIR = "figures"; os.makedirs(FIG_DIR, exist_ok=True)
CKPT_DIR = "checkpoints"; os.makedirs(CKPT_DIR, exist_ok=True)
GIF_DIR = "gifs"; os.makedirs(GIF_DIR, exist_ok=True)
LOG_DIR = "logs"; os.makedirs(LOG_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "fb_dataset.npz")
CKPT_LAST = os.path.join(CKPT_DIR, "fb_hnn_last.pkl")
CKPT_BEST = os.path.join(CKPT_DIR, "fb_hnn_best.pkl")
TXT_LOG = os.path.join(LOG_DIR, "train_log.txt")
TRAIN_IDX_PATH = os.path.join(DATA_DIR, "split_idx_train.npy")
TEST_IDX_PATH = os.path.join(DATA_DIR, "split_idx_test.npy")

# Physics constants & ODE tol
RHO_WATER = 1000.0  # kg/m^3
A_REF = 1.0         # m^2
M_BODY = 100.0      # kg
TOL = 1e-6
