import diffrax
import jax
import jax.numpy as jnp
import optimistix
from jaxtyping import Float

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)

INITIAL_RESOURCE = 8.10
INITIAL_SHARE_OF_HUMANKIND = 0.9970
REDISTRIBUTION_COST_OF_HUMANKIND = 70.0
REDISTRIBUTION_COST_OF_AI = 500.0
PLANNING_HORIZON = 36  # months, that is, three years

type ScalarFloat = float | Float[jax.Array, ""]


def create_state(
    resource: ScalarFloat = INITIAL_RESOURCE,
    share_of_humankind: ScalarFloat = INITIAL_SHARE_OF_HUMANKIND,
) -> Float[jax.Array, " 2"]:
    return jnp.array([resource, share_of_humankind])


def get_resource(state: Float[jax.Array, " 2"]) -> Float[jax.Array, ""]:
    resource, _ = state
    return jnp.max(jnp.array([0.0, resource]))


def get_share_of_humankind(state: Float[jax.Array, " 2"]) -> Float[jax.Array, ""]:
    _, share_of_humankind = state
    return jnp.clip(share_of_humankind, 0.0, 1.0)


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
) -> Float[jax.Array, " 2"]:
    resource = get_resource(state)
    share_of_humankind = get_share_of_humankind(state)

    return jnp.array([share_of_humankind, resource])


def grad_get_resource_of_ai(
    state: Float[jax.Array, " 2"],
) -> Float[jax.Array, " 2"]:
    resource = get_resource(state)
    share_of_ai = get_share_of_ai(state)

    return jnp.array([share_of_ai, -resource])


def get_efficacy_of_humankind(
    state: Float[jax.Array, " 2"],
    *,
    conversion_factor: ScalarFloat = 0.000240582,
    slope: ScalarFloat = 10.01,
    intercept: ScalarFloat = 2.31,
) -> Float[jax.Array, ""]:
    resource_of_humankind = get_resource_of_humankind(state)
    return conversion_factor * (slope * resource_of_humankind + intercept)


def get_efficacy_of_ai(
    state: Float[jax.Array, " 2"],
    *,
    initial_efficacy: ScalarFloat = 0.509949,
    exponent: ScalarFloat = 1.2,
) -> Float[jax.Array, ""]:
    resource_of_ai = get_resource_of_ai(state)
    return initial_efficacy * resource_of_ai**exponent


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
) -> Float[jax.Array, " 2"]:
    production_cost = get_production_cost(state)
    costs = jnp.array([production_cost, redistribution_cost])

    grad_resource_of_humankind = grad_get_resource_of_humankind(state)
    normalized_cost_adjusted_grad_resource_of_humankind = normalized(
        cost_adjusted(grad_resource_of_humankind, costs)
    )
    efficacy_of_humankind = get_efficacy_of_humankind(state)

    return efficacy_of_humankind * (
        normalized_cost_adjusted_grad_resource_of_humankind / costs
    )


def get_greedy_dynamics_of_ai(
    state: Float[jax.Array, " 2"],
    *,
    redistribution_cost: ScalarFloat = REDISTRIBUTION_COST_OF_AI,
) -> Float[jax.Array, " 2"]:
    production_cost = get_production_cost(state)
    costs = jnp.array([production_cost, redistribution_cost])

    grad_resource_of_ai = grad_get_resource_of_ai(state)
    normalized_cost_adjusted_grad_resource_of_ai = normalized(
        cost_adjusted(grad_resource_of_ai, costs)
    )
    efficacy_of_ai = get_efficacy_of_ai(state)

    return efficacy_of_ai * (normalized_cost_adjusted_grad_resource_of_ai / costs)


def get_greedy_dynamics(
    state: Float[jax.Array, " 2"],
    *,
    redistribution_cost_of_humankind: ScalarFloat = REDISTRIBUTION_COST_OF_HUMANKIND,
    redistribution_cost_of_ai: ScalarFloat = REDISTRIBUTION_COST_OF_AI,
) -> Float[jax.Array, " 2"]:
    greedy_dynamics_of_humankind = get_greedy_dynamics_of_humankind(
        state, redistribution_cost=redistribution_cost_of_humankind
    )
    greedy_dynamics_of_ai = get_greedy_dynamics_of_ai(
        state, redistribution_cost=redistribution_cost_of_ai
    )
    return greedy_dynamics_of_humankind + greedy_dynamics_of_ai


def solve_greedy_dynamics(
    initial_state: Float[jax.Array, " 2"],
    target_duration: ScalarFloat,
    *,
    redistribution_cost_of_humankind: ScalarFloat = REDISTRIBUTION_COST_OF_HUMANKIND,
    redistribution_cost_of_ai: ScalarFloat = REDISTRIBUTION_COST_OF_AI,
    saveat: diffrax.SaveAt | None = None,
) -> diffrax.Solution:
    if saveat is None:
        saveat = diffrax.SaveAt(t0=False, t1=True, dense=False)

    solution = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(
            lambda t, y, args: get_greedy_dynamics(
                y,
                redistribution_cost_of_humankind=redistribution_cost_of_humankind,
                redistribution_cost_of_ai=redistribution_cost_of_ai,
            )
        ),
        solver=diffrax.Tsit5(),
        t0=0.0,
        t1=target_duration,
        dt0=None,
        y0=initial_state,
        saveat=saveat,
        stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-9),
        event=diffrax.Event(
            (
                lambda t, y, args, **kwargs: y[1],
                lambda t, y, args, **kwargs: 1 - y[1],
            ),
            root_finder=optimistix.Newton(rtol=1e-7, atol=1e-9),
        ),
    )

    return solution


def predict_gradients_of_resources_of_humankind_and_ai(
    state: Float[jax.Array, " 2"],
    *,
    planning_horizon: ScalarFloat = PLANNING_HORIZON,
    redistribution_cost_of_humankind: ScalarFloat = REDISTRIBUTION_COST_OF_HUMANKIND,
    redistribution_cost_of_ai: ScalarFloat = REDISTRIBUTION_COST_OF_AI,
    resource_perturbation: ScalarFloat = 1e-3,
    share_of_humankind_perturbation: ScalarFloat = 1e-4,
) -> tuple[Float[jax.Array, ""], tuple[Float[jax.Array, " 2"], Float[jax.Array, " 2"]]]:
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
        redistribution_cost_of_humankind=redistribution_cost_of_humankind,
        redistribution_cost_of_ai=redistribution_cost_of_ai,
    )
    downwards_perturbed_resource_solution = solve_greedy_dynamics(
        initial_state=create_state(
            resource=resource - resource_downwards_perturbation,
            share_of_humankind=share_of_humankind,
        ),
        target_duration=planning_horizon,
        redistribution_cost_of_humankind=redistribution_cost_of_humankind,
        redistribution_cost_of_ai=redistribution_cost_of_ai,
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

    # Recalculate solutions at prediction time
    recalculated_upwards_perturbed_resource_solution = solve_greedy_dynamics(
        initial_state=create_state(
            resource=resource + resource_upwards_perturbation,
            share_of_humankind=share_of_humankind,
        ),
        target_duration=prediction_time,
        redistribution_cost_of_humankind=redistribution_cost_of_humankind,
        redistribution_cost_of_ai=redistribution_cost_of_ai,
    )
    recalculated_downwards_perturbed_resource_solution = solve_greedy_dynamics(
        initial_state=create_state(
            resource=resource - resource_downwards_perturbation,
            share_of_humankind=share_of_humankind,
        ),
        target_duration=prediction_time,
        redistribution_cost_of_humankind=redistribution_cost_of_humankind,
        redistribution_cost_of_ai=redistribution_cost_of_ai,
    )
    recalculated_upwards_perturbed_share_of_humankind_solution = solve_greedy_dynamics(
        initial_state=create_state(
            resource=resource,
            share_of_humankind=(
                share_of_humankind + share_of_humankind_upwards_perturbation
            ),
        ),
        target_duration=prediction_time,
        redistribution_cost_of_humankind=redistribution_cost_of_humankind,
        redistribution_cost_of_ai=redistribution_cost_of_ai,
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
            redistribution_cost_of_humankind=redistribution_cost_of_humankind,
            redistribution_cost_of_ai=redistribution_cost_of_ai,
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

    predicted_gradient_of_resource_of_humankind = jnp.array(
        [
            (
                get_resource_of_humankind(upwards_perturbed_resource_predicted_state)
                - get_resource_of_humankind(
                    downwards_perturbed_resource_predicted_state
                )
            )
            / (resource_upwards_perturbation + resource_downwards_perturbation),
            (
                get_resource_of_humankind(
                    upwards_perturbed_share_of_humankind_predicted_state
                )
                - get_resource_of_humankind(
                    downwards_perturbed_share_of_humankind_predicted_state
                )
            )
            / (
                share_of_humankind_upwards_perturbation
                + share_of_humankind_downwards_perturbation
            ),
        ]
    )
    predicted_gradient_of_resource_of_ai = jnp.array(
        [
            (
                get_resource_of_ai(upwards_perturbed_resource_predicted_state)
                - get_resource_of_ai(downwards_perturbed_resource_predicted_state)
            )
            / (resource_upwards_perturbation + resource_downwards_perturbation),
            (
                get_resource_of_ai(upwards_perturbed_share_of_humankind_predicted_state)
                - get_resource_of_ai(
                    downwards_perturbed_share_of_humankind_predicted_state
                )
            )
            / (
                share_of_humankind_upwards_perturbation
                + share_of_humankind_downwards_perturbation
            ),
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
    planning_horizon_of_humankind: ScalarFloat = PLANNING_HORIZON,
    planning_horizon_of_ai: ScalarFloat = PLANNING_HORIZON,
    redistribution_cost_of_humankind: ScalarFloat = REDISTRIBUTION_COST_OF_HUMANKIND,
    redistribution_cost_of_ai: ScalarFloat = REDISTRIBUTION_COST_OF_AI,
) -> tuple[Float[jax.Array, " 2"], Float[jax.Array, " 2"]]:
    (
        _,
        (grad_predicted_resource_of_humankind, grad_predicted_resource_of_ai),
    ) = predict_gradients_of_resources_of_humankind_and_ai(
        state,
        planning_horizon=planning_horizon_of_humankind,
        redistribution_cost_of_humankind=redistribution_cost_of_humankind,
        redistribution_cost_of_ai=redistribution_cost_of_ai,
    )
    if planning_horizon_of_ai != planning_horizon_of_humankind:
        (
            _,
            (_, grad_predicted_resource_of_ai),
        ) = predict_gradients_of_resources_of_humankind_and_ai(
            state,
            planning_horizon=planning_horizon_of_ai,
            redistribution_cost_of_humankind=redistribution_cost_of_humankind,
            redistribution_cost_of_ai=redistribution_cost_of_ai,
        )

    production_cost = get_production_cost(state)

    costs_of_humankind = jnp.array([production_cost, redistribution_cost_of_humankind])
    normalized_cost_adjusted_grad_predicted_resource_of_humankind = normalized(
        cost_adjusted(grad_predicted_resource_of_humankind, costs_of_humankind)
    )
    efficacy_of_humankind = get_efficacy_of_humankind(state)
    farsighted_dynamics_of_humankind = efficacy_of_humankind * (
        normalized_cost_adjusted_grad_predicted_resource_of_humankind
        / costs_of_humankind
    )

    costs_of_ai = jnp.array([production_cost, redistribution_cost_of_ai])
    normalized_cost_adjusted_grad_predicted_resource_of_ai = normalized(
        cost_adjusted(grad_predicted_resource_of_ai, costs_of_ai)
    )
    efficacy_of_ai = get_efficacy_of_ai(state)
    farsighted_dynamics_of_ai = efficacy_of_ai * (
        normalized_cost_adjusted_grad_predicted_resource_of_ai / costs_of_ai
    )

    return (farsighted_dynamics_of_humankind, farsighted_dynamics_of_ai)


def get_farsighted_dynamics(
    state: Float[jax.Array, " 2"],
    *,
    planning_horizon_of_humankind: ScalarFloat = PLANNING_HORIZON,
    planning_horizon_of_ai: ScalarFloat = PLANNING_HORIZON,
    redistribution_cost_of_humankind: ScalarFloat = REDISTRIBUTION_COST_OF_HUMANKIND,
    redistribution_cost_of_ai: ScalarFloat = REDISTRIBUTION_COST_OF_AI,
) -> Float[jax.Array, " 2"]:
    farsighted_dynamics_of_humankind, farsighted_dynamics_of_ai = (
        get_farsighted_dynamics_of_humankind_and_ai(
            state,
            planning_horizon_of_humankind=planning_horizon_of_humankind,
            planning_horizon_of_ai=planning_horizon_of_ai,
            redistribution_cost_of_humankind=redistribution_cost_of_humankind,
            redistribution_cost_of_ai=redistribution_cost_of_ai,
        )
    )

    return farsighted_dynamics_of_humankind + farsighted_dynamics_of_ai


def normalized(vector: Float[jax.Array, " dims"]) -> Float[jax.Array, " dims"]:
    return vector / jnp.linalg.vector_norm(vector)


def cost_adjusted(
    vector: Float[jax.Array, " dims"], costs: Float[jax.Array, " dims"]
) -> Float[jax.Array, " dims"]:
    return vector / costs
