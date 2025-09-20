
import os, numpy as np, jax, jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jax import vmap
from .config import FIG_DIR
from .networks import incompressible_u

def compute_flow_field_grid(params_psi, xlim=(-4.0, 4.0), ylim=(-4.0, 4.0), grid_n=25):
    x = np.linspace(xlim[0], xlim[1], grid_n, dtype=np.float32)
    y = np.linspace(ylim[0], ylim[1], grid_n, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    u_vmapped = vmap(lambda p: incompressible_u(params_psi, p))
    UV = np.array(jax.device_get(u_vmapped(jnp.array(pts))))
    U = UV[:, 0].reshape(X.shape)
    V = UV[:, 1].reshape(Y.shape)
    S = np.sqrt(U**2 + V**2)
    return X, Y, U, V, S

def visualize_flow_field(params, xlim=(-4.0, 4.0), ylim=(-4.0, 4.0), grid_n=25, prefix="model", compare_true=True):
    os.makedirs(FIG_DIR, exist_ok=True)
    X, Y, U, V, S = compute_flow_field_grid(params["psi"], xlim=xlim, ylim=ylim, grid_n=grid_n)

    plt.figure(figsize=(5.6, 5))
    plt.quiver(X, Y, U, V, angles="xy", scale_units="xy")
    plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.axis("equal"); plt.xlim(*xlim); plt.ylim(*ylim); plt.tight_layout()
    out_quiver = os.path.join(FIG_DIR, f"flow_quiver_{prefix}.png")
    plt.savefig(out_quiver, dpi=150); plt.close()

    plt.figure(figsize=(5.6, 5))
    plt.contourf(X, Y, S, levels=20, alpha=0.75)
    plt.streamplot(X, Y, U, V, density=1.0, linewidth=1.0, arrowsize=1.5)
    plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.axis("equal"); plt.xlim(*xlim); plt.ylim(*ylim); plt.tight_layout()
    out_stream = os.path.join(FIG_DIR, f"flow_stream_{prefix}.png")
    plt.savefig(out_stream, dpi=150); plt.close()

    if compare_true:
        U_t = -0.2 * Y; V_t = 0.2 * X; S_t = np.sqrt(U_t**2 + V_t**2)
        plt.figure(figsize=(5.6, 5)); plt.quiver(X, Y, U_t, V_t, angles="xy", scale_units="xy")
        plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.axis("equal"); plt.xlim(*xlim); plt.ylim(*ylim); plt.tight_layout()
        out_quiver_true = os.path.join(FIG_DIR, f"flow_quiver_true.png")
        plt.savefig(out_quiver_true, dpi=150); plt.close()

        plt.figure(figsize=(5.6, 5)); plt.contourf(X, Y, S_t, levels=20, alpha=0.75)
        plt.streamplot(X, Y, U_t, V_t, density=1.0, linewidth=1.0, arrowsize=1.5)
        plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.axis("equal"); plt.xlim(*xlim); plt.ylim(*ylim); plt.tight_layout()
        out_stream_true = os.path.join(FIG_DIR, f"flow_stream_true.png")
        plt.savefig(out_stream_true, dpi=150); plt.close()

    return {"quiver": out_quiver, "stream": out_stream}
