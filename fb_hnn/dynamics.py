
import jax, jax.numpy as jnp
from jax import jit, vmap, lax
from jax.experimental.ode import odeint
from .config import RHO_WATER, A_REF, M_BODY, TOL
from .networks import incompressible_u, coef_from_invariants
from .physics import external_force

@jit
def hydrodynamic_forces(params, s):
    x, y, vx, vy = s
    xy = jnp.array([x, y])
    v = jnp.array([vx, vy])

    u = incompressible_u(params["psi"], xy)
    v_rel = v - u
    speed = jnp.linalg.norm(v_rel) + 1e-8
    r = jnp.linalg.norm(xy)

    m_ax, m_ay, cd_q, cd_l = coef_from_invariants(params["coef"], r, speed)

    F_q = -0.5 * RHO_WATER * A_REF * cd_q * speed * v_rel
    F_l = -cd_l * v_rel
    F_h = F_q + F_l

    m_eff = jnp.array([M_BODY + m_ax, M_BODY + m_ay])
    return F_h, m_eff, u, v_rel

@jit
def acceleration(params, s, t):
    F_h, m_eff, _, _ = hydrodynamic_forces(params, s)
    a = (F_h + external_force(s, t)) / m_eff
    return jnp.concatenate([s[2:4], a])

@jit
def f_dynamics(params, s, t):
    return acceleration(params, s, t)

@jit
def rollout_model(params, s0, t_grid):
    return odeint(lambda s, tt: f_dynamics(params, s, tt), s0, t_grid, rtol=TOL, atol=TOL)

@jit
def rk4_step(f, x, t, h):
    k1 = h * f(x, t)
    k2 = h * f(x + 0.5 * k1, t + 0.5 * h)
    k3 = h * f(x + 0.5 * k2, t + 0.5 * h)
    k4 = h * f(x + k3, t + h)
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
