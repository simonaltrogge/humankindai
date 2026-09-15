import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    import humankindai as hai

    return hai, jax, jnp, mpl, plt


@app.cell
def _(hai):
    def unpack_derivative_of_resources_of_humankind_wrt_share_of_humankind(
        share_of_humankind,
        planning_horizon,
        *,
        rtol=hai.RELATIVE_TOLERANCE,
        atol=hai.ABSOLUTE_TOLERANCE,
    ):
        (
            prediction_time,
            ((_, derivative_of_resources_of_humankind_wrt_share_of_humankind), _),
        ) = hai.predict_gradients_of_resources_of_humankind_and_ai(
            hai.create_state(hai.INITIAL_RESOURCE, share_of_humankind),
            planning_horizon=planning_horizon,
            account_for_edge_behavior=True,
            rtol=rtol,
            atol=atol,
            # performance_over_memory=True,
        )
        return prediction_time, derivative_of_resources_of_humankind_wrt_share_of_humankind

    return (
        unpack_derivative_of_resources_of_humankind_wrt_share_of_humankind,
    )


@app.cell
def _(
    jax,
    jnp,
    unpack_derivative_of_resources_of_humankind_wrt_share_of_humankind,
):
    shares_of_humankind = jnp.linspace(0.995, 1.0, 1000)
    prediction_times_3y, derivative_wrt_share_of_humankind_3y = jax.vmap(
        lambda a: unpack_derivative_of_resources_of_humankind_wrt_share_of_humankind(
            a,
            3 * 12,  # months, that is, three years
        )
    )(shares_of_humankind)
    prediction_times_5y, derivative_wrt_share_of_humankind_5y = jax.vmap(
        lambda a: unpack_derivative_of_resources_of_humankind_wrt_share_of_humankind(
            a,
            5 * 12,  # months, that is, five years
        )
    )(shares_of_humankind)
    return (
        derivative_wrt_share_of_humankind_3y,
        derivative_wrt_share_of_humankind_5y,
        shares_of_humankind,
    )


@app.cell
def _(
    derivative_wrt_share_of_humankind_3y,
    derivative_wrt_share_of_humankind_5y,
    hai,
    mpl,
    plt,
    shares_of_humankind,
):
    plt.style.use("petroff6")
    mpl.rcParams["lines.linewidth"] = 2
    mm_per_inch = 25.4
    inch_per_mm = 1 / mm_per_inch

    C5_petroff8 = plt.style.library["petroff8"]["axes.prop_cycle"].by_key()["color"][5]

    fig, ax = plt.subplots(
        figsize=(89 * inch_per_mm, 77 * inch_per_mm), layout="constrained"
    )

    ax.axhline(0, color="k", linewidth=0.8)
    ax.plot(shares_of_humankind, derivative_wrt_share_of_humankind_3y, color="C0")
    ax.plot(
        shares_of_humankind,
        derivative_wrt_share_of_humankind_5y,
        color=C5_petroff8,
        linestyle="dashed",
    )
    ax.axvline(
        hai.INITIAL_SHARE_OF_HUMANKIND,
        color="C4",
        linewidth=1.5,
        linestyle="dashed",
        alpha=0.5,
    )
    ax.set_xlim(0.995, 1.0)
    ax.set_ylim(bottom=-50, top=10)
    ax.set_xlabel(r"$a(0)$")
    ax.set_ylabel(r"$\mathrm{d}a(T_h)R(T_h)/\mathrm{d}a(0)$")
    ax.set_xticks(
        [0.995, hai.INITIAL_SHARE_OF_HUMANKIND, 1.000], ["0.995", r"$a_0$", "1.000"]
    )

    inset = ax.inset_axes((0.6, 0.175, 0.4, 0.4), xlim=(0.99825, 0.99875), ylim=(-1, 5))
    inset.axhline(0, color="k", linewidth=0.8)
    inset.plot(shares_of_humankind, derivative_wrt_share_of_humankind_3y, color="C0")
    inset.plot(
        shares_of_humankind,
        derivative_wrt_share_of_humankind_5y,
        color=C5_petroff8,
        linestyle="dashed",
    )
    inset.set_xticks([0.99825, 0.99875])
    # inset.set_xlabel(r"a(0)", labelpad=-9)

    ax.indicate_inset_zoom(inset)

    ax.spines[["top", "right"]].set_visible(False)

    # ax.scatter(
    #     shares_of_humankind[prediction_times_3y < 3 * 12],
    #     jnp.full_like(shares_of_humankind[prediction_times_3y < 3 * 12], 10),
    #     color="C0",
    # )
    # ax.scatter(
    #     shares_of_humankind[prediction_times_5y < 5 * 12],
    #     jnp.full_like(shares_of_humankind[prediction_times_5y < 5 * 12], -50),
    #     color=C5_petroff8,
    # )

    plt.savefig("../results/daR_da.svg")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
