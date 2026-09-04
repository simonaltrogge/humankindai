import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import diffrax
    import matplotlib.pyplot as plt

    import humainkind as hai

    return diffrax, hai, plt


@app.cell
def _(diffrax, hai):
    marginalized_ai_solution = hai.solve_greedy_dynamics(
        initial_state=hai.create_state(),
        target_duration=120,
        rtol=1e-7,
        atol=1e-9,
        dtmax=0.01,
        saveat=diffrax.SaveAt(t1=True, steps=True),
        event=None,
        max_steps=16**4,
    )
    return (marginalized_ai_solution,)


@app.cell
def _(diffrax, hai):
    marginalized_humankind_solution = hai.solve_greedy_dynamics(
        initial_state=hai.create_state(
            resource=hai.INITIAL_RESOURCE,
            share_of_humankind=hai.INITIAL_SHARE_OF_HUMANKIND - 0.2,
        ),
        target_duration=120,
        rtol=1e-7,
        atol=1e-9,
        dtmax=0.01,
        saveat=diffrax.SaveAt(t1=True, steps=True),
        event=None,
        max_steps=16**4,
    )
    return (marginalized_humankind_solution,)


@app.cell
def _(hai, marginalized_ai_solution, marginalized_humankind_solution, plt):
    plt.style.use("petroff6")

    fig, axs = plt.subplots(2, 4, figsize=(8, 4))

    for row, (solution, color) in enumerate(
        [[marginalized_ai_solution, "C0"], [marginalized_humankind_solution, "C2"]]
    ):
        times = hai.get_times(solution)
        resources = hai.map_states(solution, hai.get_resource)
        shares_of_humankind = hai.map_states(solution, hai.get_share_of_humankind)
        efficacies_of_humankind = hai.map_states(solution, hai.get_efficacy_of_humankind)
        efficacies_of_ai = hai.map_states(solution, hai.get_efficacy_of_ai)

        axs[row, 0].scatter(times / 12, resources, c=color, marker=".")
        axs[row, 1].scatter(times / 12, shares_of_humankind, c=color, marker=".")
        axs[row, 2].scatter(times / 12, efficacies_of_humankind, c=color, marker=".")
        axs[row, 3].scatter(times / 12, efficacies_of_ai, c=color, marker=".")

    for column in range(4):
        axs[1, column].set_xlim(0, 1.5)

    plt.show()
    return


if __name__ == "__main__":
    app.run()
