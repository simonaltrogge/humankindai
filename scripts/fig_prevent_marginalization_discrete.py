import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import diffrax
    import jax
    import jax.numpy as jnp
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    import humankindai as hai
    from humankindai.plotting import colored_line, zoom_effect

    return diffrax, hai, jax, jnp, mpl, plt


@app.cell
def _():
    PLANNING_HORIZON = 5 * 12  # months, that is, five years
    TARGET_DURATION = 50 * 12  # months, that is, 50 years
    TIME_STEP = 12 / (365 * 24)  # one hour
    return PLANNING_HORIZON, TARGET_DURATION, TIME_STEP


@app.cell
def _(PLANNING_HORIZON, TARGET_DURATION, TIME_STEP, diffrax, hai):
    solution = hai.solve_farsighted_dynamics(
        initial_state=hai.create_state(
            hai.INITIAL_RESOURCE, hai.INITIAL_SHARE_OF_HUMANKIND
        ),
        target_duration=TARGET_DURATION,
        planning_horizon_of_humankind=PLANNING_HORIZON,
        planning_horizon_of_ai=PLANNING_HORIZON,
        solver=diffrax.Euler(),
        dt0=TIME_STEP,
        stepsize_controller=diffrax.ConstantStepSize(),
        max_steps=16**5,
        saveat=diffrax.SaveAt(t0=True, t1=True, steps=True),
        event=None,
        account_for_edge_behavior=True,
        # performance_over_memory=True,
        progress_meter=diffrax.TextProgressMeter(),
    )
    assert solution.ys is not None
    return (solution,)


@app.cell
def _(solution):
    solution
    return


@app.cell
def _(solution):
    solution.stats
    return


@app.cell
def _(hai, solution):
    times_in_months = hai.get_times(solution)
    times_in_years = times_in_months / 12
    resources = hai.map_states(solution, hai.get_resource)
    resources_of_humankind = hai.map_states(solution, hai.get_resource_of_humankind)
    resources_of_ai = hai.map_states(solution, hai.get_resource_of_ai)
    shares_of_humankind = hai.map_states(solution, hai.get_share_of_humankind)
    efficacies_of_humankind = hai.map_states(solution, hai.get_efficacy_of_humankind)
    efficacies_of_ai = hai.map_states(solution, hai.get_efficacy_of_ai)
    return (
        efficacies_of_ai,
        efficacies_of_humankind,
        resources,
        resources_of_ai,
        resources_of_humankind,
        shares_of_humankind,
        times_in_years,
    )


@app.cell
def _(PLANNING_HORIZON, hai, solution):
    prediction_times = hai.map_states(
        solution,
        lambda state: hai.predict_gradients_of_resources_of_humankind_and_ai(
            state,
            planning_horizon=PLANNING_HORIZON,
            early_return_prediction_time_only=True,
        ),
    )
    return (prediction_times,)


@app.cell
def _(PLANNING_HORIZON, prediction_times, times_in_years):
    time_of_kink_in_years = times_in_years[prediction_times < PLANNING_HORIZON][0]
    return (time_of_kink_in_years,)


@app.cell
def _(
    efficacies_of_ai,
    efficacies_of_humankind,
    mpl,
    plt,
    resources,
    resources_of_ai,
    resources_of_humankind,
    shares_of_humankind,
    time_of_kink_in_years,
    times_in_years,
):
    plt.style.use("petroff6")
    mpl.rcParams["lines.linewidth"] = 2
    mm_per_inch = 25.4
    inch_per_mm = 1 / mm_per_inch

    fig, axd = plt.subplot_mosaic(
        "ab",
        figsize=(183 * inch_per_mm, 90 * inch_per_mm),
        layout="constrained",
        sharex=True,
    )

    axd["a"].plot(times_in_years, resources_of_humankind, color="C0")
    axd["a"].plot(times_in_years, resources_of_ai, color="C2")
    axd["a"].plot(times_in_years, resources, color="C1")
    axd["a"].set_xlabel("Years from 2024")
    axd["a"].set_ylabel("Resource production (EJ/month)")
    axd["a"].set_xlim(0, 50)
    axd["a"].set_ylim(0, 50)
    axd["a"].spines[["top", "right"]].set_visible(False)

    ax_a_inset = axd["a"].inset_axes(
        (0.55, 0.25, 0.4, 0.4), xlim=(0, 50), ylim=(0.895, 1.005)
    )
    ax_a_inset.axvline(
        float(time_of_kink_in_years),
        color="C0",
        linewidth=1.5,
        linestyle="dashed",
        alpha=0.5,
    )
    ax_a_inset.plot(times_in_years, shares_of_humankind, color="C0")
    ax_a_inset.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
    ax_a_inset.xaxis.set_major_locator(mpl.ticker.MultipleLocator(50))
    ax_a_inset.set_yticks([0.9, 1])
    ax_a_inset.set_xlabel("Years from 2024")
    ax_a_inset.set_ylabel("Share of\nhumankind", labelpad=-20)


    axd["b"].plot(times_in_years, efficacies_of_humankind, color="C0")
    axd["b"].set_ylabel("Efficacy of humankind")
    axd["b"].set_ylim(0, 0.12)
    axd["b"].set_xlabel("Years from 2024")
    axd["b"].yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.04))
    axd["b"].spines[["top", "right"]].set_visible(False)

    ax_b_inset = axd["b"].inset_axes((0.55, 0.25, 0.4, 0.4), xlim=(0, 50), ylim=(0, 0.325))
    ax_b_inset.axvline(
        float(time_of_kink_in_years),
        color="C0",
        linewidth=1.5,
        linestyle="dashed",
        alpha=0.5,
    )
    ax_b_inset.plot(times_in_years, efficacies_of_ai, color="C2")
    ax_b_inset.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
    ax_b_inset.xaxis.set_major_locator(mpl.ticker.MultipleLocator(50))
    ax_b_inset.set_yticks([0.0, 0.3])
    ax_b_inset.set_xlabel("Years from 2024")
    ax_b_inset.set_ylabel("Efficacy of AI  ", labelpad=-15)

    plt.savefig("../results/fig_prevent_marginalization_discrete_plots.svg")
    plt.show()
    return (inch_per_mm,)


@app.cell
def _(PLANNING_HORIZON, hai, jax, jnp):
    Rs_left = jnp.linspace(8, 25, 16)
    as_left = jnp.linspace(0.955, 1.0, 16)
    grid_left = jnp.stack(jnp.meshgrid(Rs_left, as_left), axis=-1)
    field_left = jax.vmap(
        jax.vmap(
            lambda state: hai.get_farsighted_dynamics(
                state,
                account_for_edge_behavior=True,
                planning_horizon_of_ai=PLANNING_HORIZON,
                planning_horizon_of_humankind=PLANNING_HORIZON,
            )
        )
    )(grid_left)
    norm_left = jnp.linalg.vector_norm(field_left, axis=-1)
    return field_left, grid_left, norm_left


@app.cell
def _(PLANNING_HORIZON, hai, jax, jnp):
    Rs_right = jnp.linspace(hai.INITIAL_RESOURCE, 100, 16)
    as_right = jnp.linspace(0.955, 1.0, 16)
    grid_right = jnp.stack(jnp.meshgrid(Rs_right, as_right), axis=-1)
    field_right = jax.vmap(
        jax.vmap(
            lambda state: hai.get_farsighted_dynamics(
                state,
                account_for_edge_behavior=True,
                planning_horizon_of_ai=PLANNING_HORIZON,
                planning_horizon_of_humankind=PLANNING_HORIZON,
            )
        )
    )(grid_right)
    norm_right = jnp.linalg.vector_norm(field_right, axis=-1)
    return field_right, grid_right, norm_right


@app.cell
def _(
    field_left,
    field_right,
    grid_left,
    grid_right,
    hai,
    inch_per_mm,
    norm_left,
    norm_right,
    plt,
    resources,
    shares_of_humankind,
):
    fig_vec_field, axs = plt.subplots(
        1, 2, figsize=(183 * inch_per_mm, 90 * inch_per_mm), layout="constrained"
    )

    quiver = axs[1].quiver(
        grid_right[..., 0],
        grid_right[..., 1],
        field_right[..., 0] / norm_right,
        field_right[..., 1] / norm_right,
        # jnp.log1p(norm_right),
        angles="xy",
        pivot="middle",
    )
    _, vmax = quiver.get_clim()
    quiver.set_clim(0.0, vmax)
    axs[1].plot(resources, shares_of_humankind, color="C4")
    axs[1].set_xlim(
        hai.INITIAL_RESOURCE - 0.025 * (100 - hai.INITIAL_RESOURCE),
        100 + 0.025 * (100 - hai.INITIAL_RESOURCE),
    )
    axs[1].set_ylim(0.955 - 0.025 * 0.045, 1.000 + 0.025 * 0.045)
    # axs[1].set_yticks([0, 1])
    axs[1].set_xlabel(r"Resource production $R$ (EJ/month)")

    axs[0].quiver(
        grid_left[..., 0],
        grid_left[..., 1],
        field_left[..., 0] / norm_left,
        field_left[..., 1] / norm_left,
        # jnp.log1p(norm_left),
        angles="xy",
        pivot="middle",
        clim=(0, vmax),
    )
    axs[0].plot(resources, shares_of_humankind, color="C4")
    axs[0].set_xlim(8 - 0.025 * 17, 25 + 0.025 * 17)
    axs[0].set_ylim(0.955 - 0.025 * 0.045, 1.000 + 0.025 * 0.045)
    axs[0].set_xlabel(r"Resource production $R$ (EJ/month)")
    axs[0].set_ylabel(r"Share of humankind $a$")
    # axs[1].set_ylabel(r"Share of humankind $a$", labelpad=-10)

    # zoom_effect(axs[0], axs[1], alpha=0.2)

    plt.savefig("../results/5y_farsighted_discrete_vector_field_adjusted.svg")
    plt.show()
    return


@app.cell
def _(PLANNING_HORIZON, hai, solution):
    farsighted_dynamics = hai.map_states(
        solution,
        lambda state: hai.get_farsighted_dynamics_of_humankind_and_ai(
            state,
            account_for_edge_behavior=True,
            planning_horizon_of_ai=PLANNING_HORIZON,
            planning_horizon_of_humankind=PLANNING_HORIZON,
        ),
    )
    return (farsighted_dynamics,)


@app.cell
def _(farsighted_dynamics):
    farsighted_dynamics.shape
    return


@app.cell
def _(farsighted_dynamics, jnp):
    farsighted_dynamics_of_humankind, farsighted_dynamics_of_ai = jnp.moveaxis(
        farsighted_dynamics, 1, 0
    )
    return farsighted_dynamics_of_ai, farsighted_dynamics_of_humankind


@app.cell
def _(
    farsighted_dynamics_of_ai,
    farsighted_dynamics_of_humankind,
    hai,
    inch_per_mm,
    jnp,
    plt,
    resources,
    shares_of_humankind,
    times_in_years,
):
    fig3, axs3 = plt.subplots(
        2, 2, figsize=(183 * inch_per_mm, 136 * inch_per_mm), layout="constrained"
    )

    C0_petroff8 = plt.style.library["petroff8"]["axes.prop_cycle"].by_key()["color"][0]
    C5_petroff10 = plt.style.library["petroff10"]["axes.prop_cycle"].by_key()["color"][5]

    axs3[0, 0].axhline(0, linewidth=0.8, color="k")
    axs3[0, 0].scatter(
        times_in_years, farsighted_dynamics_of_humankind[:, 0], color="C0", marker=".", s=2
    )
    axs3[0, 0].scatter(
        times_in_years, farsighted_dynamics_of_ai[:, 0], color="C2", marker=".", s=2
    )
    axs3[0, 0].set_xlim(left=0, right=50)
    axs3[0, 0].set_ylim(bottom=-0.025, top=0.25)
    # axs3[0, 0].xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
    axs3[0, 0].set_xlabel(r"time $t$ (years)")
    axs3[0, 0].set_ylabel(r"$\Delta_{H/A} R / \Delta t$ (EJ/month²)")

    # inset_a = axs3[0, 0].inset_axes(
    #     (0.15, 0.675, 0.4, 0.325), xlim=(0.0, 9.5), ylim=(-0.1, 0.25)
    # )
    # inset_a.axhline(0, linewidth=0.8, color="k")
    # inset_a.plot(times_in_years, farsighted_dynamics_of_humankind[:, 0], color="C0")
    # inset_a.plot(times_in_years, farsighted_dynamics_of_ai[:, 0], color="C2")
    # inset_a.xaxis.set_major_locator(mpl.ticker.MultipleLocator(4))
    # inset_a.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(2))
    # inset_a.set_xlabel(r"$t$ (years)", labelpad=2)

    axs3[0, 1].axhline(0, linewidth=0.8, color="k")
    axs3[0, 1].scatter(
        times_in_years, farsighted_dynamics_of_humankind[:, 1], color="C0", marker=".", s=2
    )
    axs3[0, 1].scatter(
        times_in_years, farsighted_dynamics_of_ai[:, 1], color="C2", marker=".", s=2
    )
    axs3[0, 1].set_xlim(left=0, right=50)
    axs3[0, 1].set_ylim(bottom=-0.002, top=0.002)
    # axs3[0, 1].xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
    axs3[0, 1].set_xlabel(r"time $t$ (years)")
    axs3[0, 1].set_ylabel(r"$\Delta_{H/A} a/\Delta t$ (1/month)")

    # inset_b = axs3[0, 1].inset_axes(
    #     (0.3, 0.225, 0.4, 0.325), xlim=(0.0, 9.5), ylim=(-0.001, 0.001)
    # )
    # inset_b.plot(times_in_years, farsighted_dynamics_of_humankind[:, 1], color="C0")
    # inset_b.plot(times_in_years, farsighted_dynamics_of_ai[:, 1], color="C2")
    # inset_b.xaxis.set_major_locator(mpl.ticker.MultipleLocator(4))
    # inset_b.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(2))
    # inset_b.set_xlabel(r"$t$ (years)", labelpad=2)

    axs3[1, 0].plot(times_in_years, resources, color="C1")
    axs3[1, 0].plot(
        jnp.linspace(0, 1, 100),
        jnp.full(100, 8.2173),
        color=C5_petroff10,
        linestyle="dashed",
    )
    axs3[1, 0].plot(
        jnp.linspace(1, 2, 100),
        jnp.full(100, 8.4597),
        color=C5_petroff10,
        linestyle="dashed",
    )
    axs3[1, 0].plot(
        jnp.linspace(0, 3, 500),
        hai.INITIAL_RESOURCE * jnp.exp(0.029072 * jnp.linspace(0, 3, 500)),
        color=C5_petroff10,
        linestyle="dashed",
    )
    axs3[1, 0].set_xticks([0, 1, 2, 3], [])
    axs3[1, 0].set_xticks([0.5, 1.5, 2.5], ["2024", "2025", "2026"], minor=True)
    axs3[1, 0].xaxis.set_tick_params(
        which="minor", tick1On=False, tick2On=False, labelbottom=True
    )
    axs3[1, 0].set_xlim(0, 3)
    axs3[1, 0].set_ylim(8, 9)
    axs3[1, 0].set_ylabel(r"Resource $R$ (EJ/month)")

    axs3[1, 1].plot(times_in_years, shares_of_humankind, color="C0")
    axs3[1, 1].plot(
        jnp.linspace(0, 1, 100),
        jnp.full(100, 1 - 0.030 / 8.2173),
        color=C0_petroff8,
        linestyle="dashed",
    )
    axs3[1, 1].plot(
        jnp.linspace(1, 2, 100),
        jnp.full(100, 1 - 0.045 / 8.4597),
        color=C0_petroff8,
        linestyle="dashed",
    )
    axs3[1, 1].plot(
        jnp.linspace(0, 3, 500),
        1
        - (0.02432790648648986 * jnp.exp(jnp.log(1.5) * jnp.linspace(0, 3, 500)))
        / (hai.INITIAL_RESOURCE * jnp.exp(0.029072 * jnp.linspace(0, 3, 500))),
        color=C0_petroff8,
        linestyle="dashed",
    )
    axs3[1, 1].set_xticks([0, 1, 2, 3], [])
    axs3[1, 1].set_xticks([0.5, 1.5, 2.5], ["2024", "2025", "2026"], minor=True)
    axs3[1, 1].xaxis.set_tick_params(
        which="minor", tick1On=False, tick2On=False, labelbottom=True
    )
    axs3[1, 1].set_xlim(0, 3)
    axs3[1, 1].set_ylim(0.99, 0.998)
    axs3[1, 1].set_ylabel(r"Share of humankind $a$")

    for row in range(2):
        for col in range(2):
            axs3[row, col].spines[["top", "right"]].set_visible(False)

    plt.savefig("../results/5y_farsighted_discrete_further_details.png", dpi=300)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
