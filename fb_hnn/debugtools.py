
import numpy as np, jax, jax.numpy as jnp
from typing import Dict, Any
from .config import RHO_WATER, A_REF
from .dynamics import hydrodynamic_forces
from .networks import coef_from_invariants

def debug_decompose_forces(params: Dict[str, Any], s_np: np.ndarray, t: float = 0.0) -> Dict[str, np.ndarray]:
    s = jnp.array(s_np, dtype=jnp.float32)
    F_h, m_eff, u, v_rel = hydrodynamic_forces(params, s)
    speed = jnp.linalg.norm(v_rel)
    r = jnp.linalg.norm(s[:2])
    m_ax, m_ay, cd_q, cd_l = coef_from_invariants(params["coef"], r, speed)
    F_q = -0.5 * RHO_WATER * A_REF * cd_q * speed * v_rel
    F_l = -cd_l * v_rel
    a = F_h / m_eff
    out = {
        "state": s,
        "position_xy": s[:2],
        "velocity_v": s[2:4],
        "flow_u": u,
        "v_rel": v_rel,
        "r_norm": r,
        "s_speed": speed,
        "m_ax": m_ax, "m_ay": m_ay,
        "cd_q": cd_q, "cd_l": cd_l,
        "F_q": F_q, "F_l": F_l, "F_h": F_h,
        "m_eff": m_eff, "accel_a": a,
    }
    return {k: np.array(jax.device_get(v)) if isinstance(v, (jnp.ndarray,)) else (float(v) if hasattr(v, "__float__") else v)
            for k, v in out.items()}

def print_debug_decompose(d, prefix: str = "[debug] "):
    def fmt(arr):
        arr = np.array(arr)
        if arr.ndim == 0: return f"{arr.item(): .6f}"
        return np.array2string(arr, precision=6, floatmode="fixed", suppress_small=False)
    print(prefix + "state [x,y,vx,vy]     :", fmt(d["state"]))
    print(prefix + "pos xy                 :", fmt(d["position_xy"]))
    print(prefix + "vel v                  :", fmt(d["velocity_v"]))
    print(prefix + "flow u(x,y)            :", fmt(d["flow_u"]))
    print(prefix + "v_rel                  :", fmt(d["v_rel"]))
    print(prefix + "r = ||x||              :", fmt(d["r_norm"]))
    print(prefix + "s = ||v_rel||          :", fmt(d["s_speed"]))
    print(prefix + "m_ax, m_ay             :", fmt(np.array([d["m_ax"], d["m_ay"]])))
    print(prefix + "cd_q, cd_l             :", fmt(np.array([d["cd_q"], d["cd_l"]])))
    print(prefix + "F_q (quad drag)        :", fmt(d["F_q"]))
    print(prefix + "F_l (lin  drag)        :", fmt(d["F_l"]))
    print(prefix + "F_h = F_q + F_l        :", fmt(d["F_h"]))
    print(prefix + "m_eff                  :", fmt(d["m_eff"]))
    print(prefix + "accel a                :", fmt(d["accel_a"]))
