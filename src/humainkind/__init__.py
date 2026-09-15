from collections.abc import Callable
from typing import Literal, overload

import diffrax
import equinox
import jax
import jax.numpy as jnp
import optimistix
from jaxtyping import Float, Int, Shaped

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)

INITIAL_RESOURCE = 8.098431980182847
INITIAL_SHARE_OF_HUMANKIND = 0.9969959732271604
REDISTRIBUTION_COST_OF_HUMANKIND = 70.0
REDISTRIBUTION_COST_OF_AI = 500.0

EFFICACY_OF_HUMANKIND_CONVERSION_FACTOR = 0.00021379531796626502
EFFICACY_OF_HUMANKIND_SLOPE = 11.300198969875929
EFFICACY_OF_HUMANKIND_INTERCEPT = 2.3084355749119707

EFFICACY_OF_AI_FACTOR = 0.5099485090045803
EFFICACY_OF_AI_EXPONENT = 1.2

LEARNING_RATE = 0.25

RELATIVE_TOLERANCE = 1e-10
ABSOLUTE_TOLERANCE = 1e-10

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
    conversion_factor: ScalarFloat = EFFICACY_OF_HUMANKIND_CONVERSION_FACTOR,
    slope: ScalarFloat = EFFICACY_OF_HUMANKIND_SLOPE,
    intercept: ScalarFloat = EFFICACY_OF_HUMANKIND_INTERCEPT,
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
    learning_rate: ScalarFloat = LEARNING_RATE,
) -> Float[jax.Array, ""]:
    """Learning curve."""
    resource = get_resource(state)
    progress_ratio = 1 - learning_rate
    return initial_cost * (resource / initial_resource) ** jnp.log2(progress_ratio)


def get_greedy_dynamics_of_humankind(
    state: Float[jax.Array, " 2"],
    *,
    account_for_edge_behavior: bool,
    redistribution_cost: ScalarFloat = REDISTRIBUTION_COST_OF_HUMANKIND,
    efficacy_of_humankind_params: dict | None = None,
    learning_curve_params: dict | None = None,
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
    account_for_edge_behavior: bool,
    redistribution_cost_of_humankind: ScalarFloat = REDISTRIBUTION_COST_OF_HUMANKIND,
    redistribution_cost_of_ai: ScalarFloat = REDISTRIBUTION_COST_OF_AI,
    efficacy_of_humankind_params: dict | None = None,
    efficacy_of_ai_params: dict | None = None,
    learning_curve_params: dict | None = None,
) -> Float[jax.Array, "2 2"]:
    greedy_dynamics_of_humankind = get_greedy_dynamics_of_humankind(
        state,
        account_for_edge_behavior=account_for_edge_behavior,
        redistribution_cost=redistribution_cost_of_humankind,
        efficacy_of_humankind_params=efficacy_of_humankind_params,
        learning_curve_params=learning_curve_params,
    )
    tentative_greedy_dynamics_of_ai = get_tentative_greedy_dynamics_of_ai(
        state,
        redistribution_cost=redistribution_cost_of_ai,
        efficacy_of_ai_params=efficacy_of_ai_params,
        learning_curve_params=learning_curve_params,
    )

    if not account_for_edge_behavior:
        return jnp.stack(
            [greedy_dynamics_of_humankind, tentative_greedy_dynamics_of_ai]
        )

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

    return jnp.stack([greedy_dynamics_of_humankind, greedy_dynamics_of_ai])


def get_greedy_dynamics(
    state: Float[jax.Array, " 2"],
    *,
    account_for_edge_behavior: bool,
    redistribution_cost_of_humankind: ScalarFloat = REDISTRIBUTION_COST_OF_HUMANKIND,
    redistribution_cost_of_ai: ScalarFloat = REDISTRIBUTION_COST_OF_AI,
    efficacy_of_humankind_params: dict | None = None,
    efficacy_of_ai_params: dict | None = None,
    learning_curve_params: dict | None = None,
) -> Float[jax.Array, " 2"]:
    greedy_dynamics_of_humankind, greedy_dynamics_of_ai = (
        get_greedy_dynamics_of_humankind_and_ai(
            state,
            account_for_edge_behavior=account_for_edge_behavior,
            redistribution_cost_of_humankind=redistribution_cost_of_humankind,
            redistribution_cost_of_ai=redistribution_cost_of_ai,
            efficacy_of_humankind_params=efficacy_of_humankind_params,
            efficacy_of_ai_params=efficacy_of_ai_params,
            learning_curve_params=learning_curve_params,
        )
    )
    return greedy_dynamics_of_humankind + greedy_dynamics_of_ai


@equinox.filter_jit
def solve_greedy_dynamics(
    initial_state: Float[jax.Array, " 2"],
    target_duration: ScalarFloat,
    *,
    account_for_edge_behavior: bool,
    redistribution_cost_of_humankind: ScalarFloat = REDISTRIBUTION_COST_OF_HUMANKIND,
    redistribution_cost_of_ai: ScalarFloat = REDISTRIBUTION_COST_OF_AI,
    efficacy_of_humankind_params: dict | None = None,
    efficacy_of_ai_params: dict | None = None,
    learning_curve_params: dict | None = None,
    rtol: float = RELATIVE_TOLERANCE,
    atol: float = ABSOLUTE_TOLERANCE,
    dtmax: float | None = None,
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
        "max_steps": 16**4,
    }

    solution = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(
            lambda t, y, args: get_greedy_dynamics(
                y,
                account_for_edge_behavior=account_for_edge_behavior,
                redistribution_cost_of_humankind=redistribution_cost_of_humankind,
                redistribution_cost_of_ai=redistribution_cost_of_ai,
                efficacy_of_humankind_params=efficacy_of_humankind_params,
                efficacy_of_ai_params=efficacy_of_ai_params,
                learning_curve_params=learning_curve_params,
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
    planning_horizon: ScalarFloat,
    account_for_edge_behavior: bool,
    redistribution_cost_of_humankind: ScalarFloat = REDISTRIBUTION_COST_OF_HUMANKIND,
    redistribution_cost_of_ai: ScalarFloat = REDISTRIBUTION_COST_OF_AI,
    efficacy_of_humankind_params: dict | None = None,
    efficacy_of_ai_params: dict | None = None,
    learning_curve_params: dict | None = None,
    early_return_prediction_time_only: Literal[False] = False,
    rtol: float = RELATIVE_TOLERANCE,
    atol: float = ABSOLUTE_TOLERANCE,
    performance_over_memory: bool = False,
    **diffeqsolve_kwargs,
) -> tuple[Float[jax.Array, ""], Float[jax.Array, "2 2"]]: ...
@overload
def predict_gradients_of_resources_of_humankind_and_ai(
    state: Float[jax.Array, " 2"],
    *,
    planning_horizon: ScalarFloat,
    redistribution_cost_of_humankind: ScalarFloat = REDISTRIBUTION_COST_OF_HUMANKIND,
    redistribution_cost_of_ai: ScalarFloat = REDISTRIBUTION_COST_OF_AI,
    efficacy_of_humankind_params: dict | None = None,
    efficacy_of_ai_params: dict | None = None,
    learning_curve_params: dict | None = None,
    early_return_prediction_time_only: Literal[True],
    rtol: float = RELATIVE_TOLERANCE,
    atol: float = ABSOLUTE_TOLERANCE,
    performance_over_memory: bool = False,
    **diffeqsolve_kwargs,
) -> Float[jax.Array, ""]: ...
def predict_gradients_of_resources_of_humankind_and_ai(
    state: Float[jax.Array, " 2"],
    *,
    planning_horizon: ScalarFloat,
    account_for_edge_behavior: bool | object = _sentinel,
    redistribution_cost_of_humankind: ScalarFloat = REDISTRIBUTION_COST_OF_HUMANKIND,
    redistribution_cost_of_ai: ScalarFloat = REDISTRIBUTION_COST_OF_AI,
    efficacy_of_humankind_params: dict | None = None,
    efficacy_of_ai_params: dict | None = None,
    learning_curve_params: dict | None = None,
    early_return_prediction_time_only: bool = False,
    rtol: float = RELATIVE_TOLERANCE,
    atol: float = ABSOLUTE_TOLERANCE,
    performance_over_memory: bool = False,
    **diffeqsolve_kwargs,
) -> tuple[Float[jax.Array, ""], Float[jax.Array, "2 2"]] | Float[jax.Array, ""]:
    assert (
        early_return_prediction_time_only and account_for_edge_behavior is _sentinel
    ) or (
        not early_return_prediction_time_only
        and account_for_edge_behavior is not _sentinel
    )

    resource = get_resource(state)
    share_of_humankind = get_share_of_humankind(state)
    share_of_ai = get_share_of_ai(state)

    planning_horizon = jnp.where(
        (share_of_humankind == 0.0) | (share_of_ai == 0.0), 0.0, planning_horizon
    )

    resource_perturbation = resource * 1e-8
    share_of_humankind_perturbation = jnp.max(
        jnp.array([share_of_humankind * 1e-8, 1e-9])
    )  # Prevent perturbation becoming zero when `share_of_humankind` is zero.

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
        redistribution_cost_of_humankind=redistribution_cost_of_humankind,
        redistribution_cost_of_ai=redistribution_cost_of_ai,
        efficacy_of_humankind_params=efficacy_of_humankind_params,
        efficacy_of_ai_params=efficacy_of_ai_params,
        learning_curve_params=learning_curve_params,
        rtol=rtol,
        atol=atol,
        account_for_edge_behavior=False,
        saveat=diffrax.SaveAt(
            t0=(not performance_over_memory), t1=True, steps=performance_over_memory
        ),
        **diffeqsolve_kwargs,
    )
    downwards_perturbed_resource_solution = solve_greedy_dynamics(
        initial_state=create_state(
            resource=resource - resource_downwards_perturbation,
            share_of_humankind=share_of_humankind,
        ),
        target_duration=planning_horizon,
        redistribution_cost_of_humankind=redistribution_cost_of_humankind,
        redistribution_cost_of_ai=redistribution_cost_of_ai,
        efficacy_of_humankind_params=efficacy_of_humankind_params,
        efficacy_of_ai_params=efficacy_of_ai_params,
        learning_curve_params=learning_curve_params,
        rtol=rtol,
        atol=atol,
        account_for_edge_behavior=False,
        saveat=diffrax.SaveAt(
            t0=(not performance_over_memory), t1=True, steps=performance_over_memory
        ),
        **diffeqsolve_kwargs,
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
        redistribution_cost_of_humankind=redistribution_cost_of_humankind,
        redistribution_cost_of_ai=redistribution_cost_of_ai,
        efficacy_of_humankind_params=efficacy_of_humankind_params,
        efficacy_of_ai_params=efficacy_of_ai_params,
        learning_curve_params=learning_curve_params,
        rtol=rtol,
        atol=atol,
        account_for_edge_behavior=False,
        saveat=diffrax.SaveAt(
            t0=(not performance_over_memory), t1=True, steps=performance_over_memory
        ),
        **diffeqsolve_kwargs,
    )
    downwards_perturbed_share_of_humankind_solution = solve_greedy_dynamics(
        initial_state=create_state(
            resource=resource,
            share_of_humankind=(
                share_of_humankind - share_of_humankind_downwards_perturbation
            ),
        ),
        target_duration=planning_horizon,
        redistribution_cost_of_humankind=redistribution_cost_of_humankind,
        redistribution_cost_of_ai=redistribution_cost_of_ai,
        efficacy_of_humankind_params=efficacy_of_humankind_params,
        efficacy_of_ai_params=efficacy_of_ai_params,
        learning_curve_params=learning_curve_params,
        rtol=rtol,
        atol=atol,
        account_for_edge_behavior=False,
        saveat=diffrax.SaveAt(
            t0=(not performance_over_memory), t1=True, steps=performance_over_memory
        ),
        **diffeqsolve_kwargs,
    )

    prediction_time = jnp.min(
        jnp.array(
            [
                _get_last_valid_time(upwards_perturbed_resource_solution),
                _get_last_valid_time(downwards_perturbed_resource_solution),
                _get_last_valid_time(upwards_perturbed_share_of_humankind_solution),
                _get_last_valid_time(downwards_perturbed_share_of_humankind_solution),
            ]
        )
    )

    if early_return_prediction_time_only:
        return prediction_time

    assert isinstance(account_for_edge_behavior, bool)

    # Recalculate solutions at prediction time
    upwards_perturbed_resource_predicted_state = _recalculate_state_at_prediction_time(
        upwards_perturbed_resource_solution,
        prediction_time,
        redistribution_cost_of_humankind=redistribution_cost_of_humankind,
        redistribution_cost_of_ai=redistribution_cost_of_ai,
        efficacy_of_humankind_params=efficacy_of_humankind_params,
        efficacy_of_ai_params=efficacy_of_ai_params,
        learning_curve_params=learning_curve_params,
        rtol=rtol,
        atol=atol,
        **diffeqsolve_kwargs,
    )
    downwards_perturbed_resource_predicted_state = (
        _recalculate_state_at_prediction_time(
            downwards_perturbed_resource_solution,
            prediction_time,
            redistribution_cost_of_humankind=redistribution_cost_of_humankind,
            redistribution_cost_of_ai=redistribution_cost_of_ai,
            efficacy_of_humankind_params=efficacy_of_humankind_params,
            efficacy_of_ai_params=efficacy_of_ai_params,
            learning_curve_params=learning_curve_params,
            rtol=rtol,
            atol=atol,
            **diffeqsolve_kwargs,
        )
    )
    upwards_perturbed_share_of_humankind_predicted_state = (
        _recalculate_state_at_prediction_time(
            upwards_perturbed_share_of_humankind_solution,
            prediction_time,
            redistribution_cost_of_humankind=redistribution_cost_of_humankind,
            redistribution_cost_of_ai=redistribution_cost_of_ai,
            efficacy_of_humankind_params=efficacy_of_humankind_params,
            efficacy_of_ai_params=efficacy_of_ai_params,
            learning_curve_params=learning_curve_params,
            rtol=rtol,
            atol=atol,
            **diffeqsolve_kwargs,
        )
    )
    downwards_perturbed_share_of_humankind_predicted_state = (
        _recalculate_state_at_prediction_time(
            downwards_perturbed_share_of_humankind_solution,
            prediction_time,
            redistribution_cost_of_humankind=redistribution_cost_of_humankind,
            redistribution_cost_of_ai=redistribution_cost_of_ai,
            efficacy_of_humankind_params=efficacy_of_humankind_params,
            efficacy_of_ai_params=efficacy_of_ai_params,
            learning_curve_params=learning_curve_params,
            rtol=rtol,
            atol=atol,
            **diffeqsolve_kwargs,
        )
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
    predicted_gradient_of_resource_of_humankind = jnp.where(
        prediction_time == 0.0,
        grad_get_resource_of_humankind(
            state, account_for_edge_behavior=account_for_edge_behavior
        ),
        jnp.array(
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
        ),
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
    predicted_gradient_of_resource_of_ai = jnp.where(
        prediction_time == 0.0,
        grad_get_resource_of_ai(state),
        jnp.array(
            [
                predicted_derivative_of_resource_of_ai_wrt_resource,
                predicted_derivative_of_resource_of_ai_wrt_share_of_humankind,
            ]
        ),
    )

    return (
        prediction_time,
        jnp.stack(
            [
                predicted_gradient_of_resource_of_humankind,
                predicted_gradient_of_resource_of_ai,
            ]
        ),
    )


def _get_last_valid_time(solution: diffrax.Solution) -> Float[jax.Array, ""]:
    assert solution.ts is not None
    return jnp.max(jnp.where(jnp.isfinite(solution.ts), solution.ts, -jnp.inf))


def _recalculate_state_at_prediction_time(
    solution: diffrax.Solution,
    prediction_time: Float[jax.Array, ""],
    *,
    redistribution_cost_of_humankind: ScalarFloat,
    redistribution_cost_of_ai: ScalarFloat,
    efficacy_of_humankind_params: dict | None,
    efficacy_of_ai_params: dict | None,
    learning_curve_params: dict | None,
    rtol: float,
    atol: float,
    **diffeqsolve_kwargs,
) -> Float[jax.Array, " 2"]:
    assert solution.ts is not None
    assert solution.ys is not None
    index_of_last_time_before_or_at = _last_nonzero(solution.ts <= prediction_time)
    last_time_before_or_at = solution.ts[index_of_last_time_before_or_at]
    last_state_before_or_at = solution.ys[index_of_last_time_before_or_at]

    recalculated_solution = solve_greedy_dynamics(
        initial_state=last_state_before_or_at,
        target_duration=(prediction_time - last_time_before_or_at),
        redistribution_cost_of_humankind=redistribution_cost_of_humankind,
        redistribution_cost_of_ai=redistribution_cost_of_ai,
        efficacy_of_humankind_params=efficacy_of_humankind_params,
        efficacy_of_ai_params=efficacy_of_ai_params,
        learning_curve_params=learning_curve_params,
        rtol=rtol,
        atol=atol,
        event=None,
        account_for_edge_behavior=True,
        **diffeqsolve_kwargs,
    )
    assert recalculated_solution.ys is not None
    return recalculated_solution.ys[-1]


def _last_nonzero(array: Shaped[jax.Array, "*dims"]) -> tuple[Int[jax.Array, ""], ...]:
    mask = array != 0
    flat_mask_view = jnp.ravel(mask)
    flat_index = jnp.max(
        jnp.where(flat_mask_view, jnp.arange(jnp.size(flat_mask_view)), -1)
    )
    index = jnp.unravel_index(flat_index, jnp.shape(array))
    return index


def get_farsighted_dynamics_of_humankind_and_ai(
    state: Float[jax.Array, " 2"],
    *,
    planning_horizon_of_humankind: ScalarFloat,
    planning_horizon_of_ai: ScalarFloat,
    account_for_edge_behavior: bool,
    redistribution_cost_of_humankind: ScalarFloat = REDISTRIBUTION_COST_OF_HUMANKIND,
    redistribution_cost_of_ai: ScalarFloat = REDISTRIBUTION_COST_OF_AI,
    efficacy_of_humankind_params: dict | None = None,
    efficacy_of_ai_params: dict | None = None,
    learning_curve_params: dict | None = None,
    rtol: float = RELATIVE_TOLERANCE,
    atol: float = ABSOLUTE_TOLERANCE,
    performance_over_memory: bool = False,
) -> Float[jax.Array, "2 2"]:
    (
        _,
        (grad_predicted_resource_of_humankind, grad_predicted_resource_of_ai),
    ) = predict_gradients_of_resources_of_humankind_and_ai(
        state,
        planning_horizon=planning_horizon_of_humankind,
        account_for_edge_behavior=account_for_edge_behavior,
        redistribution_cost_of_humankind=redistribution_cost_of_humankind,
        redistribution_cost_of_ai=redistribution_cost_of_ai,
        efficacy_of_humankind_params=efficacy_of_humankind_params,
        efficacy_of_ai_params=efficacy_of_ai_params,
        learning_curve_params=learning_curve_params,
        rtol=rtol,
        atol=atol,
        performance_over_memory=performance_over_memory,
    )
    if planning_horizon_of_ai != planning_horizon_of_humankind:
        (
            _,
            (_, grad_predicted_resource_of_ai),
        ) = predict_gradients_of_resources_of_humankind_and_ai(
            state,
            planning_horizon=planning_horizon_of_ai,
            account_for_edge_behavior=account_for_edge_behavior,
            redistribution_cost_of_humankind=redistribution_cost_of_humankind,
            redistribution_cost_of_ai=redistribution_cost_of_ai,
            efficacy_of_humankind_params=efficacy_of_humankind_params,
            efficacy_of_ai_params=efficacy_of_ai_params,
            learning_curve_params=learning_curve_params,
            rtol=rtol,
            atol=atol,
            performance_over_memory=performance_over_memory,
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
        return jnp.stack(
            [farsighted_dynamics_of_humankind, tentative_farsighted_dynamics_of_ai]
        )

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

    return jnp.stack([farsighted_dynamics_of_humankind, farsighted_dynamics_of_ai])


def get_farsighted_dynamics(
    state: Float[jax.Array, " 2"],
    *,
    planning_horizon_of_humankind: ScalarFloat,
    planning_horizon_of_ai: ScalarFloat,
    account_for_edge_behavior: bool,
    redistribution_cost_of_humankind: ScalarFloat = REDISTRIBUTION_COST_OF_HUMANKIND,
    redistribution_cost_of_ai: ScalarFloat = REDISTRIBUTION_COST_OF_AI,
    efficacy_of_humankind_params: dict | None = None,
    efficacy_of_ai_params: dict | None = None,
    learning_curve_params: dict | None = None,
    rtol: float = RELATIVE_TOLERANCE,
    atol: float = ABSOLUTE_TOLERANCE,
    performance_over_memory: bool = False,
) -> Float[jax.Array, " 2"]:
    farsighted_dynamics_of_humankind, farsighted_dynamics_of_ai = (
        get_farsighted_dynamics_of_humankind_and_ai(
            state,
            planning_horizon_of_humankind=planning_horizon_of_humankind,
            planning_horizon_of_ai=planning_horizon_of_ai,
            account_for_edge_behavior=account_for_edge_behavior,
            redistribution_cost_of_humankind=redistribution_cost_of_humankind,
            redistribution_cost_of_ai=redistribution_cost_of_ai,
            efficacy_of_humankind_params=efficacy_of_humankind_params,
            efficacy_of_ai_params=efficacy_of_ai_params,
            learning_curve_params=learning_curve_params,
            rtol=rtol,
            atol=atol,
            performance_over_memory=performance_over_memory,
        )
    )
    return farsighted_dynamics_of_humankind + farsighted_dynamics_of_ai


@equinox.filter_jit
def solve_farsighted_dynamics(
    initial_state: Float[jax.Array, " 2"],
    target_duration: ScalarFloat,
    *,
    planning_horizon_of_humankind: ScalarFloat,
    planning_horizon_of_ai: ScalarFloat,
    account_for_edge_behavior: bool,
    redistribution_cost_of_humankind: ScalarFloat = REDISTRIBUTION_COST_OF_HUMANKIND,
    redistribution_cost_of_ai: ScalarFloat = REDISTRIBUTION_COST_OF_AI,
    efficacy_of_humankind_params: dict | None = None,
    efficacy_of_ai_params: dict | None = None,
    learning_curve_params: dict | None = None,
    rtol: float = RELATIVE_TOLERANCE,
    atol: float = ABSOLUTE_TOLERANCE,
    dtmax: float | None = None,
    performance_over_memory: bool = False,
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
                planning_horizon_of_humankind=planning_horizon_of_humankind,
                planning_horizon_of_ai=planning_horizon_of_ai,
                account_for_edge_behavior=account_for_edge_behavior,
                redistribution_cost_of_humankind=redistribution_cost_of_humankind,
                redistribution_cost_of_ai=redistribution_cost_of_ai,
                efficacy_of_humankind_params=efficacy_of_humankind_params,
                efficacy_of_ai_params=efficacy_of_ai_params,
                learning_curve_params=learning_curve_params,
                rtol=rtol,
                atol=atol,
                performance_over_memory=performance_over_memory,
            )
        ),
        t0=0.0,
        t1=target_duration,
        y0=initial_state,
        **(default_diffeqsolve_kwargs | diffeqsolve_kwargs),
    )

    return solution


def get_times(solution: diffrax.Solution) -> Float[jax.Array, " times"]:
    assert solution.ts is not None

    mask = jnp.isfinite(solution.ts)
    valid_times = solution.ts[mask]
    return valid_times


def get_states(solution: diffrax.Solution) -> Float[jax.Array, " times 2"]:
    assert solution.ts is not None
    assert solution.ys is not None

    mask = jnp.isfinite(solution.ts)
    valid_states = solution.ys[mask]
    return valid_states


def map_states(
    solution: diffrax.Solution,
    mapping: Callable[[Float[jax.Array, " 2"]], Shaped[jax.Array, " *dim"]],
) -> Shaped[jax.Array, " times *dim"]:
    valid_states = get_states(solution)
    mapped_states = jax.vmap(mapping)(valid_states)
    return mapped_states


def get_times_and_states(
    solution: diffrax.Solution,
) -> tuple[Float[jax.Array, " times"], Float[jax.Array, " times 2"]]:
    assert solution.ts is not None
    assert solution.ys is not None

    mask = jnp.isfinite(solution.ts)
    valid_times = solution.ts[mask]
    valid_states = solution.ys[mask]
    return (valid_times, valid_states)


def get_times_and_mapped_states(
    solution: diffrax.Solution,
    mapping: Callable[[Float[jax.Array, " 2"]], Shaped[jax.Array, " *dim"]],
) -> tuple[Float[jax.Array, " times"], Shaped[jax.Array, " times *dim"]]:
    valid_times, valid_states = get_times_and_states(solution)
    mapped_states = jax.vmap(mapping)(valid_states)
    return (valid_times, mapped_states)


def normalized(vector: Float[jax.Array, " dims"]) -> Float[jax.Array, " dims"]:
    norm = jnp.linalg.vector_norm(vector)
    norm = jnp.where(norm == 0.0, 1.0, norm)
    return vector / norm


def cost_adjusted(
    vector: Float[jax.Array, " dims"], costs: Float[jax.Array, " dims"]
) -> Float[jax.Array, " dims"]:
    return vector / costs
