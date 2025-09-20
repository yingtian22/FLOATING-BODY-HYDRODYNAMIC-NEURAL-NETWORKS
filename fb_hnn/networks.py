
import jax, jax.numpy as jnp
from jax import jit
from jax.example_libraries import stax

# Streamfunction network ψ(x,y) → scalar
init_psi, psi_forward = stax.serial(
    stax.Dense(64), stax.Softplus,
    stax.Dense(64), stax.Softplus,
    stax.Dense(1),
)

# Coefficient network φ=[r, s, 1] → 4 scalars
init_coef, coef_forward = stax.serial(
    stax.Dense(64), stax.Softplus,
    stax.Dense(64), stax.Softplus,
    stax.Dense(4),
)

CAP_MA = 60.0  # added mass upper bound
CAP_CQ = 2.0   # quadratic drag coef upper bound
CAP_CL = 10.0  # linear drag coef upper bound

@jit
def psi_scalar(params_psi, xy2):
    return psi_forward(params_psi, xy2[None, :])[0, 0]

@jit
def incompressible_u(params_psi, xy2):
    # u = ∇^⟂ ψ = (-∂ψ/∂y, ∂ψ/∂x), guarantees div u = 0
    g = jax.grad(lambda q: psi_scalar(params_psi, q))(xy2)
    return jnp.array([-g[1], g[0]])

@jit
def coef_from_invariants(params_coef, r, s):
    feats = jnp.array([r, s, 1.0])[None, :]
    raw = coef_forward(params_coef, feats)[0]
    # positivity + caps
    m_ax = jax.nn.softplus(raw[0])
    m_ay = jax.nn.softplus(raw[1])
    cd_q = jax.nn.softplus(raw[2])
    cd_l = jax.nn.softplus(raw[3])
    m_ax = jnp.minimum(m_ax, CAP_MA)
    m_ay = jnp.minimum(m_ay, CAP_MA)
    cd_q = jnp.minimum(cd_q, CAP_CQ)
    cd_l = jnp.minimum(cd_l, CAP_CL)
    return m_ax, m_ay, cd_q, cd_l
