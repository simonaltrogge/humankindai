import jax
import jax.numpy as jnp
from jaxtyping import Float


def create_state(resource: float, share_of_humankind: float) -> Float[jax.Array, " 2"]:
    return jnp.array([resource, share_of_humankind])


def get_resource(state: Float[jax.Array, " 2"]) -> Float[jax.Array, ""]:
    resource, _ = state
    return resource


def get_share_of_humankind(state: Float[jax.Array, " 2"]) -> Float[jax.Array, ""]:
    _, share_of_humankind = state
    return share_of_humankind


def get_share_of_ai(state: Float[jax.Array, " 2"]) -> Float[jax.Array, ""]:
    share_of_humankind = get_share_of_humankind(state)
    return 1 - share_of_humankind


def get_resource_of_humankind(state: Float[jax.Array, " 2"]) -> Float[jax.Array, ""]:
    resource = get_resource(state)
    share_of_humankind = get_share_of_humankind(state)
    return share_of_humankind * resource


def get_resource_of_ai(state: Float[jax.Array, " 2"]) -> Float[jax.Array, ""]:
    resource = get_resource(state)
    share_of_ai = get_share_of_ai(state)
    return share_of_ai * resource


grad_get_resource_of_humankind = jax.grad(get_resource_of_humankind)
grad_get_resource_of_ai = jax.grad(get_resource_of_ai)


def get_efficacy_of_humankind(
    state: Float[jax.Array, " 2"],
    *,
    conversion_factor: float = 0.0002329,
    slope: float = 10.01,
    intercept: float = 2.31,
) -> Float[jax.Array, ""]:
    resource_of_humankind = get_resource_of_humankind(state)
    return conversion_factor * (slope * resource_of_humankind + intercept)


def get_efficacy_of_ai(
    state: Float[jax.Array, " 2"], *, initial_efficacy=0.124575, exponent=1.1
) -> Float[jax.Array, ""]:
    resource_of_ai = get_resource_of_ai(state)
    return initial_efficacy * resource_of_ai**exponent


def get_production_cost(
    state: Float[jax.Array, " 2"],
    *,
    initial_cost=1.0,
    initial_resource=9.19963,
    learning_rate=0.25,
) -> Float[jax.Array, ""]:
    """Learning curve."""
    resource = get_resource(state)
    progress_ratio = 1 - learning_rate
    return initial_cost * (resource / initial_resource) ** jnp.log2(progress_ratio)


def get_greedy_dynamics_of_humankind(
    state: Float[jax.Array, " 2"], *, redistribution_cost: float = 50.0
) -> Float[jax.Array, " 2"]:
    production_cost = get_production_cost(state)
    cost_vector = jnp.array([production_cost, redistribution_cost])

    grad_resource_of_humankind = grad_get_resource_of_humankind(state)
    cost_adjusted_grad_resource_of_humankind = grad_resource_of_humankind / cost_vector
    normalized_cost_adjusted_grad_resource_of_humankind = (
        cost_adjusted_grad_resource_of_humankind
        / jnp.linalg.vector_norm(cost_adjusted_grad_resource_of_humankind)
    )

    efficacy_of_humankind = get_efficacy_of_humankind(state)

    return efficacy_of_humankind * (
        normalized_cost_adjusted_grad_resource_of_humankind / cost_vector
    )


def get_greedy_dynamics_of_ai(
    state: Float[jax.Array, " 2"], *, redistribution_cost: float = 250.0
) -> Float[jax.Array, " 2"]:
    production_cost = get_production_cost(state)
    cost_vector = jnp.array([production_cost, redistribution_cost])

    grad_resource_of_ai = grad_get_resource_of_ai(state)
    cost_adjusted_grad_resource_of_ai = grad_resource_of_ai / cost_vector
    normalized_cost_adjusted_grad_resource_of_ai = (
        cost_adjusted_grad_resource_of_ai
        / jnp.linalg.vector_norm(cost_adjusted_grad_resource_of_ai)
    )

    efficacy_of_ai = get_efficacy_of_ai(state)

    return efficacy_of_ai * (normalized_cost_adjusted_grad_resource_of_ai / cost_vector)


def get_greedy_dynamics(state: Float[jax.Array, " 2"]) -> Float[jax.Array, " 2"]:
    return get_greedy_dynamics_of_humankind(state) + get_greedy_dynamics_of_ai(state)
