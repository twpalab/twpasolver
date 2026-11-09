# Changelog

## 0.1.0

First release on PyPI. The analysis interface was redesigned; code written for 0.0.x needs porting (see the
tutorials for the new workflow).

### New

- `bloch` gain model: forward-backward CMEs in the exact Bloch-mode basis, with reflections, connectors, segmented
  devices (exact junction conditions), a fixed edge basis for modes close to stopband edges, and port-resolved
  gains `gain_db` (S21), `gain_db_12`, `gain_db_11`, `gain_db_22` with the external currents `a_in`, `a_out`,
  `b_in`, `b_out`.
- `ModeSet`: modes and frequency relations with any number of independent tones (`tones=` in `gain` and
  `phase_matching`); exact RWA terms on integer frequency vectors.
- `GainResult` and `PhaseMatchingResult`: the same fields for every model, `save`/`load` to HDF5, `plot`,
  `GainResult.bandwidth()`, `plot_currents()`, `mode(label)`.
- `TWPAnalysis.linear` (`LinearData`): Bloch parameters per segment, all stopbands in the grid, `k_star`; computed
  on demand and refreshed when the model or the grid changes. `k` follows the winding of the mode, so it is
  continuous across resonator notches and negative in left-handed bands.
- `Ports(input, output, Z_ext)`: connector models at both ends (`bloch` only); `TWPAnalysis.response()` gives the
  linear two-port response including them.
- User models: any `TwoPortModel` subclass that declares a `name` is accepted in cell lists and JSON files, also
  when defined after import; custom cells support the `bloch` model through `bloch_nl_elements()`.
- `nl_scale` on `LCLfBaseCell` and `StubBaseCell`: nonlinear fraction of the line inductance (`bloch` couplings).
- `RKSolverConfig` and `BlochSolverConfig` for the integrator and the forward-backward solver; the
  `twpasolver-warmup` command compiles the kernels once.

### Changed (breaking)

- `TWPAnalysis(twpa, freqs, unit, ports)`: `freqs` is an arange tuple or a list/array (replaces `f_arange`/`f_list`
  and the `Frequencies` class); `ports=Ports(...)` replaces `port`, `port_out`, `port_Z_ext`.
- Analysis methods return results and store nothing: `data`, `data_file`, `save_data`, `load_data`,
  `update_base_data`, the auto-saving decorator and `parameter_sweep` are removed (save results with
  `result.save(path)`; a sweep is a loop).
- `gain(...)` returns a `GainResult`; `modes=` (a `ModeSet` or `"basic_3wm"`/`"basic_4wm"`) replaces
  `mode_array_config` and the mode-array registry (`get_mode_array`, `add_mode_array`); `idler_mode` is removed from
  `gain`; `pump` is required (no automatic pump estimate); `I_tuples_bwd` is on the physical position grid.
- `phase_matching(signal_freqs, pump_freqs, ...)` returns a `PhaseMatchingResult` with `optimal_pump_freq` (NaN
  when no pump is valid); its 4WM Kerr term uses `2 chi k_p`; pumps inside a stopband give NaN.
- The gain model `general_no_reflections` is now `simple`; the `general` forward-backward model and the `quantum`
  package are removed.
- `ModeArray`, `ModeArrayFactory`, `RWAAnalyzer` and `Mode` are replaced by `ModeSet`; sympy is no longer a
  dependency.
- Bandwidth and plots are methods of the results (`result.bandwidth()`, `result.plot()`); `TWPAnalysis` keeps
  `plot_response()`.
- `TwoPortCell.interpolate` defaults to `polar=False`; `ABCDArray` products and powers use NumPy
  (`matmul_2x2`/`matpow_2x2` removed); `model * n` validates `N`.
- The `twpasolver` logger no longer propagates to the root logger.
- Requires Python >= 3.10.

### Fixed

- `TWPAnalysis(twpa=...)`, `TWPA.from_file`, `read_file` and `save_to_file` accept `pathlib.Path` file names.
- Base data no longer goes stale after editing the model or the grid; `gain(Ip0=...)` no longer changes the model.
- Stopband detection independent of where the frequency grid ends; HDF5 saves no longer destroy the previous file
  on failure; complex `Z0` handled consistently in S-parameters; NaN results reported instead of passed silently.
- Phase matching of segmented devices (length-weighted per-segment mismatch with each segment's nonlinearity).
- Mode frequencies outside the grid of the linear response raise instead of using extrapolated
  wavenumbers.
- Mode currents along the line (`I_fwd`, `I_bwd`) come from the integrator's fourth-order dense output instead of
  a linear interpolation between its adaptive steps.
