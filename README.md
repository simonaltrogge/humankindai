# humankindai

*Dynamical systems modeling of the interaction between humankind and
artificial intelligence (AI).*

This repository contains code to model future interactions between humankind an AI around a shared pool of resources, as presented in the preprint [*A basic computational model of humankind's marginalization by AI*](https://doi.org/10.48565/bonndoc-982) (Memmesheimer & Altrogge, 2026).

## Installation

This project is managed with [uv](https://docs.astral.sh/uv/) and uses
[marimo notebooks](https://marimo.io/). To install the project:

 1. If you have not already, [install uv](https://docs.astral.sh/uv/getting-started/installation/).

 2. Clone (or download) this repository.
    ```bash
    git clone https://github.com/simonaltrogge/humankindai.git
    ```

 3. Enter the directory and create the project's virtual Python
    environment.
    ```bash
    cd humankindai
    uv sync
    ```

## Usage

If you followed the [installation instructions](#installation) above, uv
has created a virtual environment in the `.venv` directory under the
projects main folder. In this virtual environment, the `humankindai`
package as well as all its dependencies (including marimo) are already
installed. More information on uv-managed projects can be found in the [uv
docs](https://docs.astral.sh/uv/guides/projects/).

### Interactive Python session

To enter an interactive Python session using the project's virtual
environment, execute from somewhere within the project's directory
structure
```bash
uv run python
```
You can then import the package using
```pycon
>>> import humankindai
```

### Marimo notebooks

This project includes marimo notebooks (an alternative to Jupyter
notebooks) in its `scripts` folder. You can start up a marimo notebook, for example, `fig_greedy_dynamics.py`, by executing from within the project's main folder
```bash
uv run marimo edit ./scripts/fig_greedy_dynamics.py
```

More information on marimo and its integration with uv can be found in
the [marimo docs](https://docs.marimo.io/) and [uv
docs](https://docs.astral.sh/uv/guides/projects/), respectively. You can also use marimo from within some popular editors, such as Visual Studio Code and PyCharm, see [here](https://docs.marimo.io/getting_started/quickstart/#editor-extensions).

## Citation 

If you use this code in your research, please cite our preprint

```bibtex
@unpublished{memmesheimer2026basic
    handle: 20.500.11811/14518,
    doi: https://doi.org/10.48565/bonndoc-982,
    author = {{Raoul-Martin Memmesheimer} and {Simon Altrogge}},
    title = {A basic computational model of humankind's marginalization by AI},
    publisher = {Universitäts- und Landesbibliothek Bonn},
    year = 2026,
    month = sep,
    url = {https://hdl.handle.net/20.500.11811/14518}
}
```
