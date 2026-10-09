# twpasolver
Coupled Mode Equations solver for Traveling Wave Parametric Amplifiers.

## Documentation

You can find the latest documentation [here](https://twpalab.github.io/twpasolver/index.html).

## Installation

Clone the repository with:

```bash
git clone https://github.com/twpalab/twpasolver
```

then to install it in normal mode:

```bash
pip install .
```

Use poetry to install the latest version in developer mode, remember to also
install the pre-commits!

```bash
poetry install --with docs,analysis
pre-commit install
```

The coupled-mode-equation solver is JIT-compiled with Numba and cached to
disk, but that cache is only populated once something actually calls it. Run
the warmup command once after installing to pre-compile it, instead of
paying that cost (several minutes) the first time you run an analysis:

```bash
twpasolver-warmup
```

## License

twpasolver is licensed under the [Apache License 2.0](LICENSE). See the [LICENSE](LICENSE) file for details.
