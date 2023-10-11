# Revelio

[![PyPI](https://img.shields.io/pypi/v/revelio?style=flat-square)](https://pypi.python.org/pypi/revelio/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/revelio?style=flat-square)](https://pypi.python.org/pypi/revelio/)
[![PyPI - License](https://img.shields.io/pypi/l/revelio?style=flat-square)](https://pypi.python.org/pypi/revelio/)
[![Coookiecutter - Wolt](https://img.shields.io/badge/cookiecutter-Wolt-00c2e8?style=flat-square&logo=cookiecutter&logoColor=D4AA00&link=https://github.com/woltapp/wolt-python-package-cookiecutter)](https://github.com/woltapp/wolt-python-package-cookiecutter)


---

**Documentation**: [https://ndido98.github.io/revelio](https://ndido98.github.io/revelio)

**Source Code**: [https://github.com/ndido98/revelio](https://github.com/ndido98/revelio)

**PyPI**: [https://pypi.org/project/revelio/](https://pypi.org/project/revelio/)

---

A declarative framework for Morphing Attack Detection experiments

## Installation

```sh
git clone git@github.com:ndido98/revelio
cd revelio
poetry install
poetry run revelio
```

## Reproducing the experiments

Using the uploaded configuration file in the `experiments/` directory inside the repository, it is possible to reproduce the experiments reported in the paper.

The following experiments are available at the moment:
* [Inception-Resnet V1, Table 6](https://github.com/ndido98/revelio/blob/master/experiments/inception-resnet.yml)

This list is currently a work in progress.

## Development

* Clone this repository
* Requirements:
  * [Poetry](https://python-poetry.org/)
  * Python 3.10+
* Create a virtual environment and install the dependencies

```sh
poetry install
```

* Activate the virtual environment

```sh
poetry shell
```

### Testing

```sh
pytest
```

### Documentation

The documentation is automatically generated from the content of the [docs directory](./docs) and from the docstrings
 of the public signatures of the source code. The documentation is updated and published as a [Github project page
 ](https://pages.github.com/) automatically as part each release.

### Pre-commit

Pre-commit hooks run all the auto-formatters (e.g. `black`, `isort`), linters (e.g. `mypy`, `flake8`), and other quality
 checks to make sure the changeset is in good shape before a commit/push happens.

You can install the hooks with (runs for each commit):

```sh
pre-commit install
pre-commit install -t commit-msg
```

Or if you want e.g. want to run all checks manually for all files:

```sh
pre-commit run --all-files
```

---

This project was generated using the [wolt-python-package-cookiecutter](https://github.com/woltapp/wolt-python-package-cookiecutter) template.
