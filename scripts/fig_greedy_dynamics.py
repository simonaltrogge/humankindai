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
    TARGET_DURATION = 10 * 12  # months, that is, ten years
    MAXIMUM_TIME_STEP = 1  # month
    return MAXIMUM_TIME_STEP, TARGET_DURATION


@app.cell
def _(MAXIMUM_TIME_STEP, TARGET_DURATION, diffrax, hai):
    eliminated_ai_solution = hai.solve_greedy_dynamics(
        initial_state=hai.create_state(
            hai.INITIAL_RESOURCE,
            hai.INITIAL_SHARE_OF_HUMANKIND,
        ),
        target_duration=TARGET_DURATION,
        dtmax=MAXIMUM_TIME_STEP,
        saveat=diffrax.SaveAt(t0=True, t1=True, steps=True),
        event=None,
        account_for_edge_behavior=True,
    )
    assert eliminated_ai_solution.ys is not None
    return (eliminated_ai_solution,)


@app.cell
def _(MAXIMUM_TIME_STEP, TARGET_DURATION, diffrax, hai):
    marginalized_humankind_solution = hai.solve_greedy_dynamics(
        initial_state=hai.create_state(
            resource=hai.INITIAL_RESOURCE,
            share_of_humankind=0.97,
        ),
        target_duration=TARGET_DURATION,
        dtmax=MAXIMUM_TIME_STEP,
        saveat=diffrax.SaveAt(t0=True, t1=True, steps=True),
        event=None,
        account_for_edge_behavior=True,
        throw=False,
    )
    assert marginalized_humankind_solution.ys is not None
    return (marginalized_humankind_solution,)


@app.cell
def _(marginalized_humankind_solution):
    marginalized_humankind_solution.stats
    return


@app.cell
def _(mpl, plt):
    plt.style.use("petroff6")
    mpl.rcParams["lines.linewidth"] = 2
    mm_per_inch = 25.4
    inch_per_mm = 1 / mm_per_inch
    return (inch_per_mm,)


@app.cell
def _(eliminated_ai_solution, hai, inch_per_mm, plt):
    fig1, axs1 = plt.subplots(
        2,
        2,
        figsize=(183 * inch_per_mm, 136 * inch_per_mm),
        layout="constrained",
        sharex=True,
    )

    times_e = hai.get_times(eliminated_ai_solution)
    resources_e = hai.map_states(eliminated_ai_solution, hai.get_resource)
    resources_of_humankind_e = hai.map_states(
        eliminated_ai_solution, hai.get_resource_of_humankind
    )
    resources_of_ai_e = hai.map_states(eliminated_ai_solution, hai.get_resource_of_ai)
    shares_of_humankind_e = hai.map_states(
        eliminated_ai_solution, hai.get_share_of_humankind
    )
    efficacies_of_humankind_e = hai.map_states(
        eliminated_ai_solution, hai.get_efficacy_of_humankind
    )
    efficacies_of_ai_e = hai.map_states(eliminated_ai_solution, hai.get_efficacy_of_ai)

    axs1[0, 0].plot(times_e / 12, resources_of_ai_e, color="C2")
    axs1[0, 0].plot(times_e / 12, resources_of_humankind_e, color="C0")
    axs1[0, 0].plot(times_e / 12, resources_e, color="C1", linestyle="dashed")
    axs1[0, 0].set_xlabel(r"time $t$ (years)")
    axs1[0, 0].set_ylabel(
        r"Resource production $R$"
        "\n(EJ/month)"
    )
    axs1[0, 0].set_xlim(0, 10)
    axs1[0, 0].set_ylim(0, 12)
    axs1[0, 0].xaxis.set_tick_params(which="major", labelbottom=True)

    ax1_t_inset = axs1[0, 0].inset_axes(
        (0.55, 0.25, 0.35, 0.4), xlim=(0, 12), ylim=(0, 0.03)
    )
    ax1_t_inset.plot(times_e / 12, resources_of_ai_e, color="C2")
    # ax1_t_inset.plot(times_e / 12, resources_of_humankind_e, color="C0")
    # ax1_t_inset.plot(times_e / 12, resources_e, color="C1", linestyle="dashed")
    ax1_t_inset.set_xlabel(r"$t$ (years)", labelpad=2)
    # ax_t_inset.set_ylabel(r"$R$ (EJ/month)")
    # ax1_t_inset.set_xticks([7.65, 7.75])

    axs1[0, 1].plot(times_e / 12, shares_of_humankind_e, color="C0")
    axs1[0, 1].set_xlabel(r"time $t$ (years)")
    axs1[0, 1].set_ylabel(r"Share of humankind $a$")
    axs1[0, 1].set_ylim(0.9965, 1.0005)
    axs1[0, 1].xaxis.set_tick_params(which="major", labelbottom=True)

    axs1[1, 0].plot(times_e / 12, efficacies_of_humankind_e, color="C0")
    axs1[1, 0].set_xlabel(r"time $t$ (years)")
    axs1[1, 0].set_ylabel(r"Efficacy of humankind $H$")
    axs1[1, 0].set_ylim(0, 0.035)

    # ax1_b_inset = axs1[1, 0].inset_axes(
    #     (0.2, 0.55, 0.35, 0.4), xlim=(7.65, 7.75), ylim=(0, 3)
    # )
    # ax1_b_inset.plot(times_e / 12, efficacies_of_humankind_e, color="C0")
    # ax1_b_inset.set_xlabel(r"$t$ (years)", labelpad=2)
    # # ax_b_inset.set_ylabel(r"$H$")
    # ax1_b_inset.set_xticks([7.65, 7.75])

    axs1[1, 1].plot(times_e / 12, efficacies_of_ai_e, color="C2")
    axs1[1, 1].set_xlabel(r"time $t$ (years)")
    axs1[1, 1].set_ylabel(r"Efficacy of AI $A$")
    axs1[1, 1].set_ylim(0, 0.008)

    for row_e in range(2):
        for col_e in range(2):
            axs1[row_e, col_e].spines[["top", "right"]].set_visible(False)

    # fig1.align_labels()
    plt.savefig("../results/fig_greedy_elimination_of_ai.svg")
    plt.show()
    return resources_e, shares_of_humankind_e


@app.cell
def _(hai, inch_per_mm, marginalized_humankind_solution, plt):
    fig2, axs2 = plt.subplots(
        2,
        2,
        figsize=(183 * inch_per_mm, 136 * inch_per_mm),
        layout="constrained",
        sharex=True,
    )

    times_m = hai.get_times(marginalized_humankind_solution)
    resources_m = hai.map_states(marginalized_humankind_solution, hai.get_resource)
    resources_of_humankind_m = hai.map_states(
        marginalized_humankind_solution, hai.get_resource_of_humankind
    )
    resources_of_ai_m = hai.map_states(
        marginalized_humankind_solution, hai.get_resource_of_ai
    )
    shares_of_humankind_m = hai.map_states(
        marginalized_humankind_solution, hai.get_share_of_humankind
    )
    efficacies_of_humankind_m = hai.map_states(
        marginalized_humankind_solution, hai.get_efficacy_of_humankind
    )
    efficacies_of_ai_m = hai.map_states(
        marginalized_humankind_solution, hai.get_efficacy_of_ai
    )
    time_of_singularity_in_years_m = times_m[efficacies_of_ai_m > 1e15][0] / 12

    axs2[0, 0].plot(times_m / 12, resources_of_ai_m, color="C2")
    axs2[0, 0].plot(times_m / 12, resources_of_humankind_m, color="C0")
    axs2[0, 0].plot(times_m / 12, resources_m, color="C1", linestyle="dashed")
    axs2[0, 0].set_xlabel(r"time $t$ (years)")
    axs2[0, 0].set_ylabel(
        r"Resource production $R$"
        "\n(EJ/month)"
    )
    axs2[0, 0].set_xlim(0, 9)
    axs2[0, 0].set_ylim(0, 80)
    axs2[0, 0].xaxis.set_tick_params(which="major", labelbottom=True)

    ax2_t_inset = axs2[0, 0].inset_axes(
        (0.2, 0.55, 0.35, 0.4), xlim=(7.65, 7.75), ylim=(0, 1500)
    )
    ax2_t_inset.plot(times_m / 12, resources_of_ai_m, color="C2")
    ax2_t_inset.plot(times_m / 12, resources_of_humankind_m, color="C0")
    ax2_t_inset.plot(times_m / 12, resources_m, color="C1", linestyle="dashed")
    ax2_t_inset.set_xlabel(r"$t$ (years)", labelpad=2)
    # ax_t_inset.set_ylabel(r"$R$ (EJ/month)")
    ax2_t_inset.set_xticks([7.65, 7.75])

    axs2[0, 1].plot(times_m / 12, shares_of_humankind_m, color="C0")
    axs2[0, 1].set_xlabel(r"time $t$ (years)")
    axs2[0, 1].set_ylabel(r"Share of humankind $a$")
    axs2[0, 1].set_ylim(0, 1.05)
    axs2[0, 1].xaxis.set_tick_params(which="major", labelbottom=True)

    axs2[1, 0].plot(times_m / 12, efficacies_of_humankind_m, color="C0")
    axs2[1, 0].set_xlabel(r"time $t$ (years)")
    axs2[1, 0].set_ylabel(
        r"Efficacy of humankind $H$"
        # "\n"
    )
    axs2[1, 0].set_ylim(0, 3)

    ax2_b_inset = axs2[1, 0].inset_axes(
        (0.2, 0.55, 0.35, 0.4), xlim=(7.65, 7.75), ylim=(0, 3)
    )
    ax2_b_inset.plot(times_m / 12, efficacies_of_humankind_m, color="C0")
    ax2_b_inset.set_xlabel(r"$t$ (years)", labelpad=2)
    # ax_b_inset.set_ylabel(r"$H$")
    ax2_b_inset.set_xticks([7.65, 7.75])

    axs2[1, 1].plot(times_m / 12, efficacies_of_ai_m, color="C2")
    axs2[1, 1].axvline(
        time_of_singularity_in_years_m,
        color="C2",
        linestyle="dashed",
        linewidth=1.5,
        alpha=0.5,
    )
    axs2[1, 1].set_xlabel(r"time $t$ (years)")
    axs2[1, 1].set_ylabel(r"Efficacy of AI $A$")
    axs2[1, 1].set_ylim(0, 50)

    for row_m in range(2):
        for col_m in range(2):
            axs2[row_m, col_m].spines[["top", "right"]].set_visible(False)

    # fig2.align_labels()
    plt.savefig("../results/fig_greedy_marginalization_of_humankind.svg")
    plt.show()
    return resources_m, shares_of_humankind_m


@app.cell
def _(hai, jax, jnp):
    Rs_left = jnp.linspace(8, 11.5, 16)
    as_left = jnp.linspace(0.965, 1.0, 16)
    grid_left = jnp.stack(jnp.meshgrid(Rs_left, as_left), axis=-1)
    field_left = jax.vmap(
        jax.vmap(
            lambda state: hai.get_greedy_dynamics(state, account_for_edge_behavior=True)
        )
    )(grid_left)
    norm_left = jnp.linalg.vector_norm(field_left, axis=-1)
    return field_left, grid_left, norm_left


@app.cell
def _(hai, jax, jnp):
    Rs_right = jnp.linspace(1, 6500, 16)
    as_right = jnp.linspace(0.0, 1.0, 16)
    grid_right = jnp.stack(jnp.meshgrid(Rs_right, as_right), axis=-1)
    field_right = jax.vmap(
        jax.vmap(
            lambda state: hai.get_greedy_dynamics(state, account_for_edge_behavior=True)
        )
    )(grid_right)
    norm_right = jnp.linalg.vector_norm(field_right, axis=-1)
    return field_right, grid_right, norm_right


@app.cell
def _(
    eliminated_ai_solution,
    field_left,
    field_right,
    grid_left,
    grid_right,
    inch_per_mm,
    marginalized_humankind_solution,
    norm_left,
    norm_right,
    plt,
    resources_e,
    resources_m,
    shares_of_humankind_e,
    shares_of_humankind_m,
    zoom_effect,
):
    fig, axs = plt.subplots(
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
    axs[1].plot(resources_e, shares_of_humankind_e, color="C4")
    axs[1].plot(resources_m, shares_of_humankind_m, color="C4")
    # colored_line(
    #     *eliminated_ai_solution.ys.T,
    #     times_e,
    #     ax=axs[1],
    #     cmap="plasma",
    #     clim=(0, 120),
    # )
    # colored_line(
    #     *marginalized_humankind_solution.ys.T,
    #     times_m,
    #     ax=axs[1],
    #     cmap="plasma",
    #     clim=(0, 120),
    # )
    axs[1].set_xlim(-0.025 * 6499, 6500 + 0.025 * 6499)
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
    axs[0].plot(*eliminated_ai_solution.ys.T, color="C4")
    axs[0].plot(*marginalized_humankind_solution.ys.T, color="C4")
    # colored_line(
    #     *eliminated_ai_solution.ys.T,
    #     times_e,
    #     ax=axs[0],
    #     cmap="plasma",
    #     clim=(0, 120),
    # )
    # colored_line(
    #     *marginalized_humankind_solution.ys.T,
    #     times_m,
    #     ax=axs[0],
    #     cmap="plasma",
    #     clim=(0, 120),
    # )
    axs[0].set_xlim(7.875, 11.625)
    axs[0].set_ylim(0.964, 1.001)
    axs[0].set_xlabel(r"Resource production $R$ (EJ/month)")
    axs[0].set_ylabel(r"Share of humankind $a$")
    # axs[1].set_ylabel(r"Share of humankind $a$", labelpad=-10)

    zoom_effect(axs[0], axs[1], alpha=0.2)

    plt.savefig("../results/greedy_vector_field.svg")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
