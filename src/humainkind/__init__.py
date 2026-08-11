import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Float


def get_resource(state: Float[ArrayLike, "*dim 2"]) -> Float[jax.Array, "*dim"]:
    resource = jnp.asarray(state)[..., 0]
    return resource


def get_share_of_humankind(
    state: Float[ArrayLike, "*dim 2"],
) -> Float[jax.Array, "*dim"]:
    share_of_humankind = jnp.asarray(state)[..., 1]
    return share_of_humankind


def get_share_of_ai(state: Float[ArrayLike, "*dim 2"]) -> Float[jax.Array, "*dim"]:
    share_of_humankind = get_share_of_humankind(state)
    return 1 - share_of_humankind


def get_resource_of_humankind(
    state: Float[ArrayLike, "*dim 2"],
) -> Float[jax.Array, "*dim"]:
    resource = get_resource(state)
    share_of_humankind = get_share_of_humankind(state)
    return share_of_humankind * resource


def get_resource_of_ai(state: Float[ArrayLike, "*dim 2"]) -> Float[jax.Array, "*dim"]:
    resource = get_resource(state)
    share_of_ai = get_share_of_ai(state)
    return share_of_ai * resource
