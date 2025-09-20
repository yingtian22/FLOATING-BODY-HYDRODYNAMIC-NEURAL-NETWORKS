
import jax, jax.numpy as jnp
from jax import jit, vmap, lax
from .networks import psi_scalar
from .dynamics import f_dynamics, rk4_step

@jit
def loss_derivative(params, batch, lambda_flow_smooth=0.0):
    S, dS_true = batch
    preds = vmap(lambda s: f_dynamics(params, s, 0.0))(S)
    base = jnp.mean((preds - dS_true) ** 2)

    def with_reg(_):
        def grad_norm_xy(s):
            xy = s[:2]
            H = jax.hessian(lambda q: psi_scalar(params["psi"], q))(xy)
            return jnp.sum(H**2)
        reg = jnp.mean(vmap(grad_norm_xy)(S))
        return base + lambda_flow_smooth * reg

    return lax.cond(lambda_flow_smooth > 0.0, with_reg, lambda _: base, operand=None)

def loss_timestep(params, batch, dt, lambda_flow_smooth=0.0):
    S, S_next = batch
    step = lambda s, t: rk4_step(lambda ss, tt: f_dynamics(params, ss, tt), s, t, dt)
    preds = jax.vmap(lambda s: step(s, 0.0))(S)
    base = jnp.mean((preds - S_next) ** 2)

    def with_reg(_):
        def grad_norm_xy(s):
            xy = s[:2]
            H = jax.hessian(lambda q: psi_scalar(params["psi"], q))(xy)
            return jnp.sum(H**2)
        reg = jnp.mean(jax.vmap(grad_norm_xy)(S))
        return base + lambda_flow_smooth * reg

    return jax.lax.cond(lambda_flow_smooth > 0.0, with_reg, lambda _: base, operand=None)
