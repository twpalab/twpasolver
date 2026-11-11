# twpasolver

Simulation of traveling-wave parametric amplifiers (TWPAs): circuit models of the line, linear response, phase
matching and gain from coupled mode equations (CMEs).

> **Scope:** this release simulates **kinetic-inductance TWPAs (KI-TWPAs)**. The cells of the line can be any linear
> circuit, but the only nonlinearity is that of a kinetic inductance (three-wave mixing with a dc bias, four-wave
> mixing without). Josephson-junction and SNAIL TWPAs are not supported yet.

- **Circuit models**: lumped and distributed elements, composed into unit cells, supercells and segmented devices;
  ABCD/S/Z parameters of any network, and user-defined models.
- **Linear response**: Bloch wavenumber, attenuation and impedance, stopbands, transmission.
- **Two gain models**:
  - `simple`: the traditional forward-only CMEs of a homogeneous line; fast, for quick sweeps and comparison with
    the literature.
  - `bloch`: forward and backward waves in the exact Bloch-mode basis of the loaded line, with reflections at the
    ends, connectors, junctions between segments and modes close to stopband edges; port-resolved gains
    (S21, S12, S11, S22).
- **Mode sets** with any number of modes and independent tones (pump harmonics, frequency conversion, dual pumps),
  with the rotating-wave terms found exactly.

## Documentation

Tutorials, physics notes and the API reference: <https://twpalab.github.io/twpasolver/>.

## Installation

```bash
pip install twpasolver
```

The CME solvers are compiled with Numba and cached to disk on first use. Run the warmup once after installing to
compile them up front (a few minutes), instead of the first time you run an analysis:

```bash
twpasolver-warmup
```

For development, clone the repository and install with Poetry and the pre-commit hooks:

```bash
git clone https://github.com/twpalab/twpasolver
cd twpasolver
poetry install --with docs,analysis
pre-commit install
```

## Example

The gain of the kinetic-inductance TWPA used in the tutorials, at the pump that best phase matches the 4-12 GHz band:

```python
import numpy as np
from twpasolver import TWPAnalysis

an = TWPAnalysis(twpa="docs/source/notebooks/imskit_v3_holotype.json", freqs=(0.2, 40, 2e-3))
pm = an.phase_matching(signal_freqs=(4, 12, 0.1), pump_freqs=(14, 18, 0.05))
result = an.gain(np.linspace(4, 12, 41), pump=pm.optimal_pump_freq, model="bloch")
print(f"pump {pm.optimal_pump_freq:.2f} GHz, mean gain {result.bandwidth()['mean_gain']:.1f} dB")
result.plot()
```

## Citing

If you use twpasolver in published work, please cite the repository: <https://github.com/twpalab/twpasolver>.

## License

twpasolver is licensed under the [Apache License 2.0](LICENSE).
