import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Float


def get_resource(state: Float[ArrayLike, " 2"]) -> Float[jax.Array, ""]:
    resource, _ = jnp.asarray(state)
    return resource


def get_share_of_humankind(state: Float[ArrayLike, " 2"]) -> Float[jax.Array, ""]:
    _, share_of_humankind = jnp.asarray(state)
    return share_of_humankind


def get_share_of_ai(state: Float[ArrayLike, " 2"]) -> Float[jax.Array, ""]:
    share_of_humankind = get_share_of_humankind(state)
    return 1 - share_of_humankind


def get_resource_of_humankind(state: Float[ArrayLike, " 2"]) -> Float[jax.Array, ""]:
    resource = get_resource(state)
    share_of_humankind = get_share_of_humankind(state)
    return share_of_humankind * resource


def get_resource_of_ai(state: Float[ArrayLike, " 2"]) -> Float[jax.Array, ""]:
    resource = get_resource(state)
    share_of_ai = get_share_of_ai(state)
    return share_of_ai * resource


def get_efficacy_of_humankind(
    state: Float[ArrayLike, " 2"], *, slope: float = 10.0, intercept: float = 2.3
) -> Float[jax.Array, ""]:
    resource_of_humankind = get_resource_of_humankind(state)
    return slope * resource_of_humankind + intercept


def get_efficacy_of_ai(
    state: Float[ArrayLike, " 2"], *, initial_efficacy=1.0, exponent=1.1
) -> Float[jax.Array, ""]:
    resource_of_ai = get_resource_of_ai(state)
    return initial_efficacy * resource_of_ai**exponent


def get_production_cost(
    state: Float[ArrayLike, " 2"],
    *,
    initial_cost=1.0,
    initial_resource=1.0,
    learning_rate=0.25,
) -> Float[jax.Array, ""]:
    """Learning curve."""
    resource = get_resource(state)
    progress_ratio = 1 - learning_rate
    return initial_cost * (resource / initial_resource) ** jnp.log2(progress_ratio)
