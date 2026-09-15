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

    return diffrax, hai, jax, jnp, mpl, plt, zoom_effect


@app.cell
def _():
    PLANNING_HORIZON = 3 * 12  # months, that is, three years
    TARGET_DURATION = 13.7 * 12  # months, that is, 13.7 years
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
    times_in_months_into_year_13 = (times_in_years - 13) * 12
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
        times_in_months_into_year_13,
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
def _(efficacies_of_ai, times_in_months_into_year_13, times_in_years):
    time_of_singularity_in_years = times_in_years[efficacies_of_ai > 1e15][0]
    time_of_singularity_in_months_into_year_13 = times_in_months_into_year_13[
        efficacies_of_ai > 1e15
    ][0]
    return (time_of_singularity_in_months_into_year_13,)


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
    time_of_singularity_in_months_into_year_13,
    times_in_months_into_year_13,
    times_in_years,
):
    plt.style.use("petroff6")
    mpl.rcParams["lines.linewidth"] = 2
    mm_per_inch = 25.4
    inch_per_mm = 1 / mm_per_inch


    inset_bounds = (0.175, 0.5, 0.4, 0.5)


    fig = plt.figure(layout="constrained", figsize=(183 * inch_per_mm, 136 * inch_per_mm))
    axd = fig.subplot_mosaic(".b;cd", sharex=True)

    LEFT_BOUND = 5.4
    RIGHT_BOUND = 7.6

    axd["b"].plot(times_in_months_into_year_13, resources_of_ai, color="C2")
    axd["b"].plot(times_in_months_into_year_13, resources_of_humankind, color="C0")
    axd["b"].plot(times_in_months_into_year_13, resources, color="C1")
    axd["b"].axvline(
        float(time_of_singularity_in_months_into_year_13),
        color="C2",
        linewidth=1.5,
        linestyle="dashed",
        alpha=0.5,
    )
    axd["b"].set_xlabel("2037")
    axd["b"].set_ylabel("Resource production (EJ/month)")
    axd["b"].set_xlim(LEFT_BOUND, RIGHT_BOUND)
    axd["b"].set_ylim(bottom=0, top=2500)
    axd["b"].set_xticks([6, 7], [])
    axd["b"].set_xticks([5.5, 6.5, 7.5], ["Jun", "Jul", "Aug"], minor=True)
    axd["b"].spines[["right", "top"]].set_visible(False)
    axd["b"].xaxis.set_tick_params(
        which="minor", tick1On=False, tick2On=False, labelbottom=True
    )

    ax_b_inset = axd["b"].inset_axes(inset_bounds, xlim=(0, 15), ylim=(0, 1.05))
    ax_b_inset.axvline(
        float(time_of_kink_in_years),
        color="C0",
        linewidth=1.5,
        linestyle="dashed",
        alpha=0.5,
    )
    ax_b_inset.plot(times_in_years, shares_of_humankind, color="C0")
    ax_b_inset.xaxis.minorticks_on()
    ax_b_inset.set_yticks([0, 1])
    ax_b_inset.set_xlabel("Years from 2024", labelpad=2)
    ax_b_inset.set_ylabel("Share of\nhumankind ", labelpad=-10)


    axd["c"].plot(times_in_months_into_year_13, efficacies_of_humankind, color="C0")
    axd["c"].set_ylim(bottom=0, top=3.0)
    axd["c"].set_xlabel("2037")
    axd["c"].set_ylabel("Efficacy of humankind")
    axd["c"].spines[["right", "top"]].set_visible(False)
    axd["c"].tick_params(axis="x", which="minor", tick1On=False, tick2On=False)

    # ax_c_inset = axd["c"].inset_axes(inset_bounds, xlim=(0.0, 15), ylim=(0.03, 0.05))
    # ax_c_inset.axvline(
    #     float(time_of_kink_in_years),
    #     color="C0",
    #     linewidth=1.5,
    #     linestyle="dashed",
    #     alpha=0.5,
    # )
    # ax_c_inset.plot(times_in_years, efficacies_of_humankind, color="C0")
    # ax_c_inset.xaxis.minorticks_on()
    # # ax_c_inset.set_yticks([0.0, 1.5])
    # ax_c_inset.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
    # ax_c_inset.set_xlabel("Years from 2024", labelpad=2)
    # ax_c_inset.set_ylabel("Efficacy of\nhumankind", labelpad=-23)

    axd["d"].plot(
        times_in_months_into_year_13, efficacies_of_ai, color="C2"
    )  # , clip_on=False)
    axd["d"].axvline(
        float(time_of_singularity_in_months_into_year_13),
        color="C2",
        linewidth=1.5,
        linestyle="dashed",
        alpha=0.5,
    )
    axd["d"].set_xlabel("2037")
    axd["d"].set_ylabel("Efficacy of AI")
    axd["d"].set_ylim(bottom=0, top=1000)
    axd["d"].spines[["right", "top"]].set_visible(False)
    axd["d"].tick_params(axis="x", which="minor", tick1On=False, tick2On=False)

    # axd["d"].set_xlim(7.217, 7.254)
    # print(f"{(7.254 - 7.217) * 30}")

    ax_d_inset = axd["d"].inset_axes(inset_bounds, xlim=(0.0, 15), ylim=(0, 1.5))
    ax_d_inset.axvline(
        float(time_of_kink_in_years),
        color="C0",
        linewidth=1.5,
        linestyle="dashed",
        alpha=0.5,
    )
    ax_d_inset.plot(times_in_years, efficacies_of_ai, color="C2")
    ax_d_inset.xaxis.minorticks_on()
    ax_d_inset.set_yticks([0.0, 1.5])
    ax_d_inset.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
    ax_d_inset.set_xlabel("Years from 2024", labelpad=2)
    ax_d_inset.set_ylabel("Efficacy of AI", labelpad=-15)

    fig.align_ylabels()
    plt.savefig("../results/fig_marginalization_of_humankind_discrete_plots.svg")
    plt.show()
    return (inch_per_mm,)


@app.cell
def _(PLANNING_HORIZON, hai, jax, jnp):
    Rs_left = jnp.linspace(8, 60, 16)
    as_left = jnp.linspace(0.94, 1.0, 16)
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
    Rs_right = jnp.linspace(hai.INITIAL_RESOURCE, 10000, 16)
    as_right = jnp.linspace(0.0, 1.0, 16)
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
    zoom_effect,
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
        hai.INITIAL_RESOURCE - 0.025 * (10000 - hai.INITIAL_RESOURCE),
        10000 + 0.025 * (10000 - hai.INITIAL_RESOURCE),
    )
    axs[1].set_ylim(-0.025, 1.025)
    axs[1].set_yticks([0, 1])
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
    axs[0].set_xlim(8 - 0.025 * 52, 60 + 0.025 * 52)
    axs[0].set_ylim(0.94 - 0.025 * 0.06, 1.0 + 0.025 * 0.06)
    axs[0].set_xlabel(r"Resource production $R$ (EJ/month)")
    axs[0].set_ylabel(r"Share of humankind $a$")
    # axs[1].set_ylabel(r"Share of humankind $a$", labelpad=-10)

    zoom_effect(axs[0], axs[1], alpha=0.2)

    plt.savefig("../results/3y_farsighted_discrete_vector_field.svg")
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
    efficacies_of_ai,
    farsighted_dynamics_of_ai,
    farsighted_dynamics_of_humankind,
    hai,
    inch_per_mm,
    jnp,
    mpl,
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
    axs3[0, 0].plot(
        times_in_years[efficacies_of_ai <= 1e15],
        farsighted_dynamics_of_ai[:, 0][efficacies_of_ai <= 1e15],
        color="C2",
    )
    axs3[0, 0].plot(times_in_years, farsighted_dynamics_of_humankind[:, 0], color="C0")
    axs3[0, 0].set_xlim(left=0, right=14)
    axs3[0, 0].set_ylim(bottom=-4, top=5)
    axs3[0, 0].xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
    axs3[0, 0].set_xlabel(r"time $t$ (years)")
    axs3[0, 0].set_ylabel(r"$\Delta_{H/A} R / \Delta t$ (EJ/month²)")

    inset_a = axs3[0, 0].inset_axes(
        (0.2, 0.675, 0.4, 0.325), xlim=(0, 12), ylim=(-0.0375, 0.0625)
    )
    inset_a.axhline(0, linewidth=0.8, color="k")
    inset_a.plot(times_in_years, farsighted_dynamics_of_ai[:, 0], color="C2")
    inset_a.plot(times_in_years, farsighted_dynamics_of_humankind[:, 0], color="C0")
    inset_a.xaxis.set_major_locator(mpl.ticker.MultipleLocator(4))
    inset_a.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(2))
    inset_a.set_xlabel(r"$t$ (years)", labelpad=2)

    axs3[0, 1].axhline(0, linewidth=0.8, color="k")
    axs3[0, 1].plot(times_in_years, farsighted_dynamics_of_ai[:, 1], color="C2")
    axs3[0, 1].plot(times_in_years, farsighted_dynamics_of_humankind[:, 1], color="C0")
    axs3[0, 1].set_xlim(left=0, right=14)
    axs3[0, 1].set_ylim(bottom=-0.1, top=0.05)
    axs3[0, 1].xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
    axs3[0, 1].yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
    axs3[0, 1].set_xlabel(r"time $t$ (years)")
    axs3[0, 1].set_ylabel(r"$\Delta_{H/A} a/\Delta t$ (1/month)")

    inset_b = axs3[0, 1].inset_axes(
        (0.3, 0.225, 0.4, 0.325), xlim=(0, 12), ylim=(-0.0015, 0.0015)
    )
    inset_b.axhline(0, linewidth=0.8, color="k")
    inset_b.plot(times_in_years, farsighted_dynamics_of_ai[:, 1], color="C2")
    inset_b.plot(times_in_years, farsighted_dynamics_of_humankind[:, 1], color="C0")
    inset_b.xaxis.set_major_locator(mpl.ticker.MultipleLocator(4))
    inset_b.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(2))
    inset_b.set_xlabel(r"$t$ (years)", labelpad=2)

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

    plt.savefig("../results/3y_farsighted_discrete_further_details.svg")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
