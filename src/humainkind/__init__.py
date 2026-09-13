from collections.abc import Callable
from typing import Literal, overload

import diffrax
import equinox
import jax
import jax.numpy as jnp
import optimistix
from jaxtyping import Float, Shaped

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)

INITIAL_RESOURCE = 8.098431980182847
INITIAL_SHARE_OF_HUMANKIND = 0.9969959732271604
REDISTRIBUTION_COST_OF_HUMANKIND = 70.0
REDISTRIBUTION_COST_OF_AI = 500.0
PLANNING_HORIZON = 36  # months, that is, three years
EFFICACY_OF_AI_FACTOR = 0.5099485090045803
EFFICACY_OF_AI_EXPONENT = 1.2

type ScalarFloat = float | Float[jax.Array, ""]


def create_state(
    resource: ScalarFloat = INITIAL_RESOURCE,
    share_of_humankind: ScalarFloat = INITIAL_SHARE_OF_HUMANKIND,
) -> Float[jax.Array, " 2"]:
    return jnp.array([resource, share_of_humankind])


def get_raw_resource(state: Float[jax.Array, " 2"]) -> Float[jax.Array, ""]:
    raw_resource, _ = state
    return raw_resource


def get_raw_share_of_humankind(state: Float[jax.Array, " 2"]) -> Float[jax.Array, ""]:
    _, raw_share_of_humankind = state
    return raw_share_of_humankind


def get_raw_share_of_ai(state: Float[jax.Array, " 2"]) -> Float[jax.Array, ""]:
    raw_share_of_humankind = get_raw_share_of_humankind(state)
    return 1 - raw_share_of_humankind


def get_resource(state: Float[jax.Array, " 2"]) -> Float[jax.Array, ""]:
    raw_resource = get_raw_resource(state)
    return jnp.max(jnp.array([0.0, raw_resource]))


def get_share_of_humankind(state: Float[jax.Array, " 2"]) -> Float[jax.Array, ""]:
    raw_share_of_humankind = get_raw_share_of_humankind(state)
    return jnp.clip(raw_share_of_humankind, 0.0, 1.0)


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


def grad_get_resource_of_humankind(
    state: Float[jax.Array, " 2"],
    *,
    account_for_edge_behavior: bool,
) -> Float[jax.Array, " 2"]:
    resource = get_resource(state)
    share_of_humankind = get_share_of_humankind(state)

    derivative_wrt_resource = share_of_humankind
    derivative_wrt_share_of_humankind = jnp.where(
        account_for_edge_behavior and share_of_humankind == 1.0, 0.0, resource
    )
    # If flag is set, prevent increase of share of humankind beyond one.

    return jnp.array([derivative_wrt_resource, derivative_wrt_share_of_humankind])


def grad_get_resource_of_ai(
    state: Float[jax.Array, " 2"],
) -> Float[jax.Array, " 2"]:
    resource = get_resource(state)
    share_of_ai = get_share_of_ai(state)

    derivative_wrt_resource = share_of_ai
    derivative_wrt_share_of_humankind = -resource

    return jnp.array([derivative_wrt_resource, derivative_wrt_share_of_humankind])


def get_efficacy_of_humankind(
    state: Float[jax.Array, " 2"],
    *,
    conversion_factor: ScalarFloat = 0.00024058187581593312,
    slope: ScalarFloat = 10.01,
    intercept: ScalarFloat = 2.31,
) -> Float[jax.Array, ""]:
    resource_of_humankind = get_resource_of_humankind(state)
    return conversion_factor * (slope * resource_of_humankind + intercept)


def get_efficacy_of_ai(
    state: Float[jax.Array, " 2"],
    *,
    factor: ScalarFloat = EFFICACY_OF_AI_FACTOR,
    exponent: ScalarFloat = EFFICACY_OF_AI_EXPONENT,
) -> Float[jax.Array, ""]:
    resource_of_ai = get_resource_of_ai(state)
    return factor * resource_of_ai**exponent


def get_production_cost(
    state: Float[jax.Array, " 2"],
    *,
    initial_cost: ScalarFloat = 1.0,
    initial_resource: ScalarFloat = INITIAL_RESOURCE,
    learning_rate: ScalarFloat = 0.25,
) -> Float[jax.Array, ""]:
    """Learning curve."""
    resource = get_resource(state)
    progress_ratio = 1 - learning_rate
    return initial_cost * (resource / initial_resource) ** jnp.log2(progress_ratio)


def get_greedy_dynamics_of_humankind(
    state: Float[jax.Array, " 2"],
    *,
    redistribution_cost: ScalarFloat = REDISTRIBUTION_COST_OF_HUMANKIND,
    efficacy_of_humankind_params: dict | None = None,
    learning_curve_params: dict | None = None,
    account_for_edge_behavior: bool,
) -> Float[jax.Array, " 2"]:
    production_cost = get_production_cost(state, **(learning_curve_params or {}))
    costs = jnp.array([production_cost, redistribution_cost])

    grad_resource_of_humankind = grad_get_resource_of_humankind(
        state, account_for_edge_behavior=account_for_edge_behavior
    )
    normalized_cost_adjusted_grad_resource_of_humankind = normalized(
        cost_adjusted(grad_resource_of_humankind, costs)
    )
    efficacy_of_humankind = get_efficacy_of_humankind(
        state, **(efficacy_of_humankind_params or {})
    )

    return efficacy_of_humankind * (
        normalized_cost_adjusted_grad_resource_of_humankind / costs
    )


def get_tentative_greedy_dynamics_of_ai(
    state: Float[jax.Array, " 2"],
    *,
    redistribution_cost: ScalarFloat = REDISTRIBUTION_COST_OF_AI,
    efficacy_of_ai_params: dict | None = None,
    learning_curve_params: dict | None = None,
) -> Float[jax.Array, " 2"]:
    production_cost = get_production_cost(state, **(learning_curve_params or {}))
    costs = jnp.array([production_cost, redistribution_cost])

    grad_resource_of_ai = grad_get_resource_of_ai(state)
    normalized_cost_adjusted_grad_resource_of_ai = normalized(
        cost_adjusted(grad_resource_of_ai, costs)
    )
    efficacy_of_ai = get_efficacy_of_ai(state, **(efficacy_of_ai_params or {}))

    return efficacy_of_ai * (normalized_cost_adjusted_grad_resource_of_ai / costs)


def _get_resource_dynamics_of_ai_when_marginalizing(
    state: Float[jax.Array, " 2"],
    share_of_humankind_dynamics_of_humankind: Float[jax.Array, ""],
    redistribution_cost_of_ai: ScalarFloat,
    efficacy_of_ai_params: dict | None,
    learning_curve_params: dict | None,
):
    efficacy_of_ai = get_efficacy_of_ai(state, **(efficacy_of_ai_params or {}))
    production_cost = get_production_cost(state, **(learning_curve_params or {}))
    square = (
        efficacy_of_ai**2
        - (redistribution_cost_of_ai * share_of_humankind_dynamics_of_humankind) ** 2
    )
    root = jnp.sqrt(jnp.where(square < 0, 0, square))
    greedy_resource_dynamics_of_ai_when_marginalizing = root / production_cost
    return greedy_resource_dynamics_of_ai_when_marginalizing


def get_greedy_dynamics_of_humankind_and_ai(
    state: Float[jax.Array, " 2"],
    *,
    redistribution_cost_of_humankind: ScalarFloat = REDISTRIBUTION_COST_OF_HUMANKIND,
    redistribution_cost_of_ai: ScalarFloat = REDISTRIBUTION_COST_OF_AI,
    efficacy_of_humankind_params: dict | None = None,
    efficacy_of_ai_params: dict | None = None,
    learning_curve_params: dict | None = None,
    account_for_edge_behavior: bool,
) -> tuple[Float[jax.Array, " 2"], Float[jax.Array, " 2"]]:
    greedy_dynamics_of_humankind = get_greedy_dynamics_of_humankind(
        state,
        redistribution_cost=redistribution_cost_of_humankind,
        efficacy_of_humankind_params=efficacy_of_humankind_params,
        learning_curve_params=learning_curve_params,
        account_for_edge_behavior=account_for_edge_behavior,
    )
    tentative_greedy_dynamics_of_ai = get_tentative_greedy_dynamics_of_ai(
        state,
        redistribution_cost=redistribution_cost_of_ai,
        efficacy_of_ai_params=efficacy_of_ai_params,
        learning_curve_params=learning_curve_params,
    )

    if not account_for_edge_behavior:
        return greedy_dynamics_of_humankind, tentative_greedy_dynamics_of_ai

    _, greedy_share_of_humankind_dynamics_of_humankind = greedy_dynamics_of_humankind
    (
        tentative_greedy_resource_dynamics_of_ai,
        tentative_greedy_share_of_humankind_dynamics_of_ai,
    ) = tentative_greedy_dynamics_of_ai

    tentative_greedy_share_of_humankind_dynamics = (
        greedy_share_of_humankind_dynamics_of_humankind
        + tentative_greedy_share_of_humankind_dynamics_of_ai
    )
    share_of_humankind = get_share_of_humankind(state)
    is_marginalizing = (
        (share_of_humankind == 0.0)
        & (tentative_greedy_share_of_humankind_dynamics < 0.0)
        & (tentative_greedy_share_of_humankind_dynamics_of_ai < 0.0)
        & (greedy_share_of_humankind_dynamics_of_humankind > 0.0)
    )

    final_greedy_resource_dynamics_of_ai = jnp.where(
        is_marginalizing,
        _get_resource_dynamics_of_ai_when_marginalizing(
            state,
            greedy_share_of_humankind_dynamics_of_humankind,
            redistribution_cost_of_ai,
            efficacy_of_ai_params,
            learning_curve_params,
        ),
        tentative_greedy_resource_dynamics_of_ai,
    )
    final_greedy_share_of_humankind_dynamics_of_ai = jnp.where(
        is_marginalizing,
        -greedy_share_of_humankind_dynamics_of_humankind,
        tentative_greedy_share_of_humankind_dynamics_of_ai,
    )

    greedy_dynamics_of_ai = jnp.array(
        [
            final_greedy_resource_dynamics_of_ai,
            final_greedy_share_of_humankind_dynamics_of_ai,
        ]
    )

    return (greedy_dynamics_of_humankind, greedy_dynamics_of_ai)


def get_greedy_dynamics(
    state: Float[jax.Array, " 2"],
    *,
    redistribution_cost_of_humankind: ScalarFloat = REDISTRIBUTION_COST_OF_HUMANKIND,
    redistribution_cost_of_ai: ScalarFloat = REDISTRIBUTION_COST_OF_AI,
    efficacy_of_humankind_params: dict | None = None,
    efficacy_of_ai_params: dict | None = None,
    learning_curve_params: dict | None = None,
    account_for_edge_behavior: bool,
) -> Float[jax.Array, " 2"]:
    greedy_dynamics_of_humankind, greedy_dynamics_of_ai = (
        get_greedy_dynamics_of_humankind_and_ai(
            state,
            redistribution_cost_of_humankind=redistribution_cost_of_humankind,
            redistribution_cost_of_ai=redistribution_cost_of_ai,
            efficacy_of_humankind_params=efficacy_of_humankind_params,
            efficacy_of_ai_params=efficacy_of_ai_params,
            learning_curve_params=learning_curve_params,
            account_for_edge_behavior=account_for_edge_behavior,
        )
    )
    return greedy_dynamics_of_humankind + greedy_dynamics_of_ai


@equinox.filter_jit
def solve_greedy_dynamics(
    initial_state: Float[jax.Array, " 2"],
    target_duration: ScalarFloat,
    *,
    rtol: float,
    atol: float,
    dtmax: float | None = None,
    redistribution_cost_of_humankind: ScalarFloat = REDISTRIBUTION_COST_OF_HUMANKIND,
    redistribution_cost_of_ai: ScalarFloat = REDISTRIBUTION_COST_OF_AI,
    efficacy_of_humankind_params: dict | None = None,
    efficacy_of_ai_params: dict | None = None,
    learning_curve_params: dict | None = None,
    account_for_edge_behavior: bool,
    **diffeqsolve_kwargs,
) -> diffrax.Solution:
    default_diffeqsolve_kwargs = {
        "solver": diffrax.Tsit5(),
        "dt0": None,
        "saveat": diffrax.SaveAt(t0=False, t1=True, dense=False),
        "stepsize_controller": diffrax.PIDController(rtol=rtol, atol=atol, dtmax=dtmax),
        "event": diffrax.Event(
            (
                lambda t, y, args, **kwargs: get_raw_share_of_humankind(y),
                lambda t, y, args, **kwargs: get_raw_share_of_ai(y),
            ),
            root_finder=optimistix.Newton(rtol=rtol, atol=atol),
        ),
    }

    solution = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(
            lambda t, y, args: get_greedy_dynamics(
                y,
                redistribution_cost_of_humankind=redistribution_cost_of_humankind,
                redistribution_cost_of_ai=redistribution_cost_of_ai,
                efficacy_of_humankind_params=efficacy_of_humankind_params,
                efficacy_of_ai_params=efficacy_of_ai_params,
                learning_curve_params=learning_curve_params,
                account_for_edge_behavior=account_for_edge_behavior,
            )
        ),
        t0=0.0,
        t1=target_duration,
        y0=initial_state,
        **(default_diffeqsolve_kwargs | diffeqsolve_kwargs),
    )

    return solution


_sentinel = object()


@overload
def predict_gradients_of_resources_of_humankind_and_ai(
    state: Float[jax.Array, " 2"],
    *,
    rtol: float,
    atol: float,
    planning_horizon: ScalarFloat = PLANNING_HORIZON,
    redistribution_cost_of_humankind: ScalarFloat = REDISTRIBUTION_COST_OF_HUMANKIND,
    redistribution_cost_of_ai: ScalarFloat = REDISTRIBUTION_COST_OF_AI,
    resource_perturbation: ScalarFloat = 1e-3,
    share_of_humankind_perturbation: ScalarFloat = 1e-6,
    efficacy_of_humankind_params: dict | None = None,
    efficacy_of_ai_params: dict | None = None,
    learning_curve_params: dict | None = None,
    early_return_prediction_time_only: Literal[False] = False,
    account_for_edge_behavior: bool,
) -> tuple[
    Float[jax.Array, ""], tuple[Float[jax.Array, " 2"], Float[jax.Array, " 2"]]
]: ...
@overload
def predict_gradients_of_resources_of_humankind_and_ai(
    state: Float[jax.Array, " 2"],
    *,
    rtol: float,
    atol: float,
    planning_horizon: ScalarFloat = PLANNING_HORIZON,
    redistribution_cost_of_humankind: ScalarFloat = REDISTRIBUTION_COST_OF_HUMANKIND,
    redistribution_cost_of_ai: ScalarFloat = REDISTRIBUTION_COST_OF_AI,
    resource_perturbation: ScalarFloat = 1e-3,
    share_of_humankind_perturbation: ScalarFloat = 1e-6,
    efficacy_of_humankind_params: dict | None = None,
    efficacy_of_ai_params: dict | None = None,
    learning_curve_params: dict | None = None,
    early_return_prediction_time_only: Literal[True],
) -> Float[jax.Array, ""]: ...
def predict_gradients_of_resources_of_humankind_and_ai(
    state: Float[jax.Array, " 2"],
    *,
    rtol: float,
    atol: float,
    planning_horizon: ScalarFloat = PLANNING_HORIZON,
    redistribution_cost_of_humankind: ScalarFloat = REDISTRIBUTION_COST_OF_HUMANKIND,
    redistribution_cost_of_ai: ScalarFloat = REDISTRIBUTION_COST_OF_AI,
    resource_perturbation: ScalarFloat = 1e-3,
    share_of_humankind_perturbation: ScalarFloat = 1e-6,
    efficacy_of_humankind_params: dict | None = None,
    efficacy_of_ai_params: dict | None = None,
    learning_curve_params: dict | None = None,
    early_return_prediction_time_only: bool = False,
    account_for_edge_behavior: bool | object = _sentinel,
) -> (
    Float[jax.Array, ""]
    | tuple[Float[jax.Array, ""], tuple[Float[jax.Array, " 2"], Float[jax.Array, " 2"]]]
):
    assert (
        early_return_prediction_time_only and account_for_edge_behavior is _sentinel
    ) or (
        not early_return_prediction_time_only
        and account_for_edge_behavior is not _sentinel
    )

    resource = get_resource(state)
    share_of_humankind = get_share_of_humankind(state)

    resource_upwards_perturbation = resource_perturbation
    resource_downwards_perturbation = jnp.min(
        jnp.array([resource, resource_perturbation])
    )
    # Keep perturbed resource non-negative.

    upwards_perturbed_resource_solution = solve_greedy_dynamics(
        initial_state=create_state(
            resource=resource + resource_upwards_perturbation,
            share_of_humankind=share_of_humankind,
        ),
        target_duration=planning_horizon,
        rtol=rtol,
        atol=atol,
        redistribution_cost_of_humankind=redistribution_cost_of_humankind,
        redistribution_cost_of_ai=redistribution_cost_of_ai,
        efficacy_of_humankind_params=efficacy_of_humankind_params,
        efficacy_of_ai_params=efficacy_of_ai_params,
        learning_curve_params=learning_curve_params,
        account_for_edge_behavior=False,
    )
    downwards_perturbed_resource_solution = solve_greedy_dynamics(
        initial_state=create_state(
            resource=resource - resource_downwards_perturbation,
            share_of_humankind=share_of_humankind,
        ),
        target_duration=planning_horizon,
        rtol=rtol,
        atol=atol,
        redistribution_cost_of_humankind=redistribution_cost_of_humankind,
        redistribution_cost_of_ai=redistribution_cost_of_ai,
        efficacy_of_humankind_params=efficacy_of_humankind_params,
        efficacy_of_ai_params=efficacy_of_ai_params,
        learning_curve_params=learning_curve_params,
        account_for_edge_behavior=False,
    )

    share_of_humankind_upwards_perturbation = jnp.min(
        jnp.array([1 - share_of_humankind, share_of_humankind_perturbation])
    )
    share_of_humankind_downwards_perturbation = jnp.min(
        jnp.array([share_of_humankind, share_of_humankind_perturbation])
    )
    # Keep perturbed share of humankind between 0 and 1.

    upwards_perturbed_share_of_humankind_solution = solve_greedy_dynamics(
        initial_state=create_state(
            resource=resource,
            share_of_humankind=(
                share_of_humankind + share_of_humankind_upwards_perturbation
            ),
        ),
        target_duration=planning_horizon,
        rtol=rtol,
        atol=atol,
        redistribution_cost_of_humankind=redistribution_cost_of_humankind,
        redistribution_cost_of_ai=redistribution_cost_of_ai,
        efficacy_of_humankind_params=efficacy_of_humankind_params,
        efficacy_of_ai_params=efficacy_of_ai_params,
        learning_curve_params=learning_curve_params,
        account_for_edge_behavior=False,
    )
    downwards_perturbed_share_of_humankind_solution = solve_greedy_dynamics(
        initial_state=create_state(
            resource=resource,
            share_of_humankind=(
                share_of_humankind - share_of_humankind_downwards_perturbation
            ),
        ),
        target_duration=planning_horizon,
        rtol=rtol,
        atol=atol,
        redistribution_cost_of_humankind=redistribution_cost_of_humankind,
        redistribution_cost_of_ai=redistribution_cost_of_ai,
        efficacy_of_humankind_params=efficacy_of_humankind_params,
        efficacy_of_ai_params=efficacy_of_ai_params,
        learning_curve_params=learning_curve_params,
        account_for_edge_behavior=False,
    )

    assert upwards_perturbed_resource_solution.ts is not None
    assert downwards_perturbed_resource_solution.ts is not None
    assert upwards_perturbed_share_of_humankind_solution.ts is not None
    assert downwards_perturbed_share_of_humankind_solution.ts is not None
    prediction_time = jnp.min(
        jnp.array(
            [
                upwards_perturbed_resource_solution.ts[0],
                downwards_perturbed_resource_solution.ts[0],
                upwards_perturbed_share_of_humankind_solution.ts[0],
                downwards_perturbed_share_of_humankind_solution.ts[0],
            ]
        )
    )

    if early_return_prediction_time_only:
        return prediction_time

    assert isinstance(account_for_edge_behavior, bool)

    # Recalculate solutions at prediction time
    recalculated_upwards_perturbed_resource_solution = solve_greedy_dynamics(
        initial_state=create_state(
            resource=resource + resource_upwards_perturbation,
            share_of_humankind=share_of_humankind,
        ),
        target_duration=prediction_time,
        rtol=rtol,
        atol=atol,
        redistribution_cost_of_humankind=redistribution_cost_of_humankind,
        redistribution_cost_of_ai=redistribution_cost_of_ai,
        efficacy_of_humankind_params=efficacy_of_humankind_params,
        efficacy_of_ai_params=efficacy_of_ai_params,
        learning_curve_params=learning_curve_params,
        event=None,
        account_for_edge_behavior=True,
    )
    recalculated_downwards_perturbed_resource_solution = solve_greedy_dynamics(
        initial_state=create_state(
            resource=resource - resource_downwards_perturbation,
            share_of_humankind=share_of_humankind,
        ),
        target_duration=prediction_time,
        rtol=rtol,
        atol=atol,
        redistribution_cost_of_humankind=redistribution_cost_of_humankind,
        redistribution_cost_of_ai=redistribution_cost_of_ai,
        efficacy_of_humankind_params=efficacy_of_humankind_params,
        efficacy_of_ai_params=efficacy_of_ai_params,
        learning_curve_params=learning_curve_params,
        event=None,
        account_for_edge_behavior=True,
    )
    recalculated_upwards_perturbed_share_of_humankind_solution = solve_greedy_dynamics(
        initial_state=create_state(
            resource=resource,
            share_of_humankind=(
                share_of_humankind + share_of_humankind_upwards_perturbation
            ),
        ),
        target_duration=prediction_time,
        rtol=rtol,
        atol=atol,
        redistribution_cost_of_humankind=redistribution_cost_of_humankind,
        redistribution_cost_of_ai=redistribution_cost_of_ai,
        efficacy_of_humankind_params=efficacy_of_humankind_params,
        efficacy_of_ai_params=efficacy_of_ai_params,
        learning_curve_params=learning_curve_params,
        event=None,
        account_for_edge_behavior=True,
    )
    recalculated_downwards_perturbed_share_of_humankind_solution = (
        solve_greedy_dynamics(
            initial_state=create_state(
                resource=resource,
                share_of_humankind=(
                    share_of_humankind - share_of_humankind_downwards_perturbation
                ),
            ),
            target_duration=prediction_time,
            rtol=rtol,
            atol=atol,
            redistribution_cost_of_humankind=redistribution_cost_of_humankind,
            redistribution_cost_of_ai=redistribution_cost_of_ai,
            efficacy_of_humankind_params=efficacy_of_humankind_params,
            efficacy_of_ai_params=efficacy_of_ai_params,
            learning_curve_params=learning_curve_params,
            event=None,
            account_for_edge_behavior=True,
        )
    )

    assert recalculated_upwards_perturbed_resource_solution.ys is not None
    assert recalculated_downwards_perturbed_resource_solution.ys is not None
    assert recalculated_upwards_perturbed_share_of_humankind_solution.ys is not None
    assert recalculated_downwards_perturbed_share_of_humankind_solution.ys is not None
    upwards_perturbed_resource_predicted_state = (
        recalculated_upwards_perturbed_resource_solution.ys[0]
    )
    downwards_perturbed_resource_predicted_state = (
        recalculated_downwards_perturbed_resource_solution.ys[0]
    )
    upwards_perturbed_share_of_humankind_predicted_state = (
        recalculated_upwards_perturbed_share_of_humankind_solution.ys[0]
    )
    downwards_perturbed_share_of_humankind_predicted_state = (
        recalculated_downwards_perturbed_share_of_humankind_solution.ys[0]
    )

    predicted_derivative_of_resource_of_humankind_wrt_resource = (
        get_resource_of_humankind(upwards_perturbed_resource_predicted_state)
        - get_resource_of_humankind(downwards_perturbed_resource_predicted_state)
    ) / (resource_upwards_perturbation + resource_downwards_perturbation)
    predicted_derivative_of_resource_of_humankind_wrt_share_of_humankind = (
        get_resource_of_humankind(upwards_perturbed_share_of_humankind_predicted_state)
        - get_resource_of_humankind(
            downwards_perturbed_share_of_humankind_predicted_state
        )
    ) / (
        share_of_humankind_upwards_perturbation
        + share_of_humankind_downwards_perturbation
    )
    predicted_gradient_of_resource_of_humankind = jnp.array(
        [
            predicted_derivative_of_resource_of_humankind_wrt_resource,
            jnp.where(
                (account_for_edge_behavior and share_of_humankind == 1.0)
                & (
                    predicted_derivative_of_resource_of_humankind_wrt_share_of_humankind
                    > 0.0
                ),
                0.0,
                predicted_derivative_of_resource_of_humankind_wrt_share_of_humankind,
            ),
        ]
    )

    predicted_derivative_of_resource_of_ai_wrt_resource = (
        get_resource_of_ai(upwards_perturbed_resource_predicted_state)
        - get_resource_of_ai(downwards_perturbed_resource_predicted_state)
    ) / (resource_upwards_perturbation + resource_downwards_perturbation)
    predicted_derivative_of_resource_of_ai_wrt_share_of_humankind = (
        get_resource_of_ai(upwards_perturbed_share_of_humankind_predicted_state)
        - get_resource_of_ai(downwards_perturbed_share_of_humankind_predicted_state)
    ) / (
        share_of_humankind_upwards_perturbation
        + share_of_humankind_downwards_perturbation
    )
    predicted_gradient_of_resource_of_ai = jnp.array(
        [
            predicted_derivative_of_resource_of_ai_wrt_resource,
            predicted_derivative_of_resource_of_ai_wrt_share_of_humankind,
        ]
    )

    return (
        prediction_time,
        (
            predicted_gradient_of_resource_of_humankind,
            predicted_gradient_of_resource_of_ai,
        ),
    )


def get_farsighted_dynamics_of_humankind_and_ai(
    state: Float[jax.Array, " 2"],
    *,
    rtol: float,
    atol: float,
    planning_horizon_of_humankind: ScalarFloat = PLANNING_HORIZON,
    planning_horizon_of_ai: ScalarFloat = PLANNING_HORIZON,
    redistribution_cost_of_humankind: ScalarFloat = REDISTRIBUTION_COST_OF_HUMANKIND,
    redistribution_cost_of_ai: ScalarFloat = REDISTRIBUTION_COST_OF_AI,
    efficacy_of_humankind_params: dict | None = None,
    efficacy_of_ai_params: dict | None = None,
    learning_curve_params: dict | None = None,
    account_for_edge_behavior: bool,
) -> tuple[Float[jax.Array, " 2"], Float[jax.Array, " 2"]]:
    (
        _,
        (grad_predicted_resource_of_humankind, grad_predicted_resource_of_ai),
    ) = predict_gradients_of_resources_of_humankind_and_ai(
        state,
        rtol=rtol,
        atol=atol,
        planning_horizon=planning_horizon_of_humankind,
        redistribution_cost_of_humankind=redistribution_cost_of_humankind,
        redistribution_cost_of_ai=redistribution_cost_of_ai,
        efficacy_of_humankind_params=efficacy_of_humankind_params,
        efficacy_of_ai_params=efficacy_of_ai_params,
        learning_curve_params=learning_curve_params,
        account_for_edge_behavior=account_for_edge_behavior,
    )
    if planning_horizon_of_ai != planning_horizon_of_humankind:
        (
            _,
            (_, grad_predicted_resource_of_ai),
        ) = predict_gradients_of_resources_of_humankind_and_ai(
            state,
            rtol=rtol,
            atol=atol,
            planning_horizon=planning_horizon_of_ai,
            redistribution_cost_of_humankind=redistribution_cost_of_humankind,
            redistribution_cost_of_ai=redistribution_cost_of_ai,
            efficacy_of_humankind_params=efficacy_of_humankind_params,
            efficacy_of_ai_params=efficacy_of_ai_params,
            learning_curve_params=learning_curve_params,
            account_for_edge_behavior=account_for_edge_behavior,
        )

    production_cost = get_production_cost(state, **(learning_curve_params or {}))

    costs_of_humankind = jnp.array([production_cost, redistribution_cost_of_humankind])
    normalized_cost_adjusted_grad_predicted_resource_of_humankind = normalized(
        cost_adjusted(grad_predicted_resource_of_humankind, costs_of_humankind)
    )
    efficacy_of_humankind = get_efficacy_of_humankind(
        state, **(efficacy_of_humankind_params or {})
    )
    farsighted_dynamics_of_humankind = efficacy_of_humankind * (
        normalized_cost_adjusted_grad_predicted_resource_of_humankind
        / costs_of_humankind
    )

    costs_of_ai = jnp.array([production_cost, redistribution_cost_of_ai])
    normalized_cost_adjusted_grad_predicted_resource_of_ai = normalized(
        cost_adjusted(grad_predicted_resource_of_ai, costs_of_ai)
    )
    efficacy_of_ai = get_efficacy_of_ai(state, **(efficacy_of_ai_params or {}))
    tentative_farsighted_dynamics_of_ai = efficacy_of_ai * (
        normalized_cost_adjusted_grad_predicted_resource_of_ai / costs_of_ai
    )

    if not account_for_edge_behavior:
        return (farsighted_dynamics_of_humankind, tentative_farsighted_dynamics_of_ai)

    _, farsighted_share_of_humankind_dynamics_of_humankind = (
        farsighted_dynamics_of_humankind
    )
    (
        tentative_farsighted_resource_dynamics_of_ai,
        tentative_farsighted_share_of_humankind_dynamics_of_ai,
    ) = tentative_farsighted_dynamics_of_ai

    tentative_farsighted_share_of_humankind_dynamics = (
        farsighted_share_of_humankind_dynamics_of_humankind
        + tentative_farsighted_share_of_humankind_dynamics_of_ai
    )
    share_of_humankind = get_share_of_humankind(state)
    is_marginalizing = (
        (share_of_humankind == 0.0)
        & (tentative_farsighted_share_of_humankind_dynamics < 0.0)
        & (tentative_farsighted_share_of_humankind_dynamics_of_ai < 0.0)
        & (farsighted_share_of_humankind_dynamics_of_humankind > 0.0)
    )

    final_farsighted_resource_dynamics_of_ai = jnp.where(
        is_marginalizing,
        _get_resource_dynamics_of_ai_when_marginalizing(
            state,
            farsighted_share_of_humankind_dynamics_of_humankind,
            redistribution_cost_of_ai,
            efficacy_of_ai_params,
            learning_curve_params,
        ),
        tentative_farsighted_resource_dynamics_of_ai,
    )
    final_farsighted_share_of_humankind_dynamics_of_ai = jnp.where(
        is_marginalizing,
        -farsighted_share_of_humankind_dynamics_of_humankind,
        tentative_farsighted_share_of_humankind_dynamics_of_ai,
    )

    farsighted_dynamics_of_ai = jnp.array(
        [
            final_farsighted_resource_dynamics_of_ai,
            final_farsighted_share_of_humankind_dynamics_of_ai,
        ]
    )

    return (farsighted_dynamics_of_humankind, farsighted_dynamics_of_ai)


def get_farsighted_dynamics(
    state: Float[jax.Array, " 2"],
    *,
    rtol: float,
    atol: float,
    planning_horizon_of_humankind: ScalarFloat = PLANNING_HORIZON,
    planning_horizon_of_ai: ScalarFloat = PLANNING_HORIZON,
    redistribution_cost_of_humankind: ScalarFloat = REDISTRIBUTION_COST_OF_HUMANKIND,
    redistribution_cost_of_ai: ScalarFloat = REDISTRIBUTION_COST_OF_AI,
    efficacy_of_humankind_params: dict | None = None,
    efficacy_of_ai_params: dict | None = None,
    learning_curve_params: dict | None = None,
    account_for_edge_behavior: bool,
) -> Float[jax.Array, " 2"]:
    farsighted_dynamics_of_humankind, farsighted_dynamics_of_ai = (
        get_farsighted_dynamics_of_humankind_and_ai(
            state,
            rtol=rtol,
            atol=atol,
            planning_horizon_of_humankind=planning_horizon_of_humankind,
            planning_horizon_of_ai=planning_horizon_of_ai,
            redistribution_cost_of_humankind=redistribution_cost_of_humankind,
            redistribution_cost_of_ai=redistribution_cost_of_ai,
            efficacy_of_humankind_params=efficacy_of_humankind_params,
            efficacy_of_ai_params=efficacy_of_ai_params,
            learning_curve_params=learning_curve_params,
            account_for_edge_behavior=account_for_edge_behavior,
        )
    )
    return farsighted_dynamics_of_humankind + farsighted_dynamics_of_ai


@equinox.filter_jit
def solve_farsighted_dynamics(
    initial_state: Float[jax.Array, " 2"],
    target_duration: ScalarFloat,
    *,
    rtol: float,
    atol: float,
    dtmax: float | None = None,
    planning_horizon_of_humankind: ScalarFloat = PLANNING_HORIZON,
    planning_horizon_of_ai: ScalarFloat = PLANNING_HORIZON,
    redistribution_cost_of_humankind: ScalarFloat = REDISTRIBUTION_COST_OF_HUMANKIND,
    redistribution_cost_of_ai: ScalarFloat = REDISTRIBUTION_COST_OF_AI,
    efficacy_of_humankind_params: dict | None = None,
    efficacy_of_ai_params: dict | None = None,
    learning_curve_params: dict | None = None,
    account_for_edge_behavior: bool,
    **diffeqsolve_kwargs,
) -> diffrax.Solution:
    default_diffeqsolve_kwargs = {
        "solver": diffrax.Tsit5(),
        "dt0": None,
        "saveat": diffrax.SaveAt(t0=False, t1=True, dense=False),
        "stepsize_controller": diffrax.PIDController(rtol=rtol, atol=atol, dtmax=dtmax),
        "event": diffrax.Event(
            (
                lambda t, y, args, **kwargs: get_raw_share_of_humankind(y),
                lambda t, y, args, **kwargs: get_raw_share_of_ai(y),
            ),
            root_finder=optimistix.Newton(rtol=rtol, atol=atol),
        ),
        "max_steps": 16**6,
    }

    solution = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(
            lambda t, y, args: get_farsighted_dynamics(
                y,
                rtol=rtol,
                atol=atol,
                planning_horizon_of_humankind=planning_horizon_of_humankind,
                planning_horizon_of_ai=planning_horizon_of_ai,
                redistribution_cost_of_humankind=redistribution_cost_of_humankind,
                redistribution_cost_of_ai=redistribution_cost_of_ai,
                efficacy_of_humankind_params=efficacy_of_humankind_params,
                efficacy_of_ai_params=efficacy_of_ai_params,
                learning_curve_params=learning_curve_params,
                account_for_edge_behavior=account_for_edge_behavior,
            )
        ),
        t0=0.0,
        t1=target_duration,
        y0=initial_state,
        **(default_diffeqsolve_kwargs | diffeqsolve_kwargs),
    )

    return solution


def get_times(solution: diffrax.Solution):
    assert solution.ts is not None
    return solution.ts[jnp.isfinite(solution.ts)]


def map_states(
    solution: diffrax.Solution,
    mapping: Callable[[Float[jax.Array, " 2"]], Shaped[jax.Array, " *dim"]],
):
    assert solution.ts is not None
    assert solution.ys is not None

    mask = jnp.isfinite(solution.ts)
    valid_states = solution.ys[mask]

    return jax.vmap(mapping)(valid_states)


def get_times_and_mapped_states(
    solution: diffrax.Solution,
    mapping: Callable[[Float[jax.Array, " 2"]], Shaped[jax.Array, " *dim"]],
    *,
    unmasked: bool = False,
) -> tuple[Float[jax.Array, " times"], Shaped[jax.Array, " times *dim"]]:
    assert solution.ts is not None
    assert solution.ys is not None

    if unmasked:
        times = solution.ts
        states = solution.ys
    else:
        mask = jnp.isfinite(solution.ts)
        times = solution.ts[mask]
        states = solution.ys[mask]

    vectorized_mapping = jax.vmap(mapping)
    mapped_states = vectorized_mapping(states)

    return (times, mapped_states)


def normalized(vector: Float[jax.Array, " dims"]) -> Float[jax.Array, " dims"]:
    norm = jnp.linalg.vector_norm(vector)
    norm = jnp.where(norm == 0.0, 1.0, norm)
    return vector / norm


def cost_adjusted(
    vector: Float[jax.Array, " dims"], costs: Float[jax.Array, " dims"]
) -> Float[jax.Array, " dims"]:
    return vector / costs
