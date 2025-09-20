
from typing import Dict, Any
import numpy as np, jax, jax.numpy as jnp
from jax import jit
from jax.experimental.ode import odeint
from .config import RHO_WATER, A_REF, M_BODY, TOL

@jit
def external_force(s, t):
    return jnp.array([0.0, 0.0])

def make_synth_params():
    @jit
    def true_world(s, t):
        x, y, vx, vy = s
        v = jnp.array([vx, vy])
        u = 0.2 * jnp.array([-y, x])  # weak CCW vortex
        v_rel = v - u
        speed = jnp.linalg.norm(v_rel) + 1e-8
        m_ax = 30.0 + 10.0 * jnp.tanh(0.2 * speed)
        m_ay = 30.0 + 10.0 * jnp.tanh(0.2 * speed)
        cd_q = 0.8
        cd_l = 5.0
        F_q = -0.5 * RHO_WATER * A_REF * cd_q * speed * v_rel
        F_l = -cd_l * v_rel
        F_h = F_q + F_l
        m_eff = jnp.array([M_BODY + m_ax, M_BODY + m_ay])
        a = F_h / m_eff
        return jnp.concatenate([v, a])
    return true_world

@jit
def rollout_true(s0, t_grid):
    true_f = make_synth_params()
    return odeint(lambda s, tt: true_f(s, tt), s0, t_grid, rtol=TOL, atol=TOL)
