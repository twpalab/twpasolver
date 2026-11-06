# twpasolver: clean-up and release plan

Working document for preparing twpasolver for publication (GitHub `twpalab/twpasolver`, Sphinx docs on
gh-pages, possibly PyPI). Written 2026-11-06 from a full review of the `edge-basis` branch: CME solvers,
`analysis.py`, and the rest of the package, tests, packaging, docs and branches. Any chat can pick up a work
package (WP) from here. Tick boxes and add notes as you go, and keep this file current.

**Goal:** a package that is correct, tested, documented and as small as possible to maintain. Prefer deleting
and merging code over adding options.

## How to work on this

- **Repo:** `/home/pietro/work/phd/twpas/twpasolver`. **Work branch: `bloch-cmes`**, created 2026-11-06 from
  `edge-basis`. `edge-basis` holds the state before the clean-up and contains every other local branch (see WP7.8).
  Commit only when Pietro asks.
- **Python:** `~/.virtualenvs/twpaenv/bin/python`. Poetry is broken on this machine; use pip/pytest directly.
- **Tests:** `~/.virtualenvs/twpaenv/bin/python -m pytest tests -q -p no:cacheprovider` takes about 30 s with a
  warm numba cache. A cold compile takes minutes: uniform `bloch` about 55 s, segmented about 80 s, `general`
  about 30 s.
- **Static checks:** `python -m mypy src` and `python -m pylint src/twpasolver`.
- **Scratch files** go in `/tmp/claude-1000/...`, never in the repo.
- **Line numbers** below refer to `edge-basis` on 2026-11-06 plus the uncommitted edge-basis fix. They drift, so
  grep for the symbol.
- **Regression reference for every refactor:** record gains before and after on the test devices (uniform and
  segmented holotype, `ext2` mode array, edge basis on). Bloch results must agree to ≤1e-9 dB unless the WP says
  otherwise.
- **Downstream users of the API:** `notebooks/contra_qpm`, `notebooks/kitwpac-article/run7`,
  `notebooks/wideiso/sim/run7`, and the thesis figure scripts in `docs/source/notebooks/*_canon.py`. A breaking
  change goes into the changelog in WP7.

## Current state (2026-11-06)

- **Size:** about 14.8k lines in `src`. The largest files are `analysis.py` (2.6k), `cmes/cmes_fb.py` (1.66k),
  `cmes/bloch_fb_segmented.py` (1.57k), `cmes/bloch_fb.py` (1.35k), `modes_rwa.py` (1.14k), `cmes/bloch.py`
  (0.87k), and `quantum/` (1.8k).
- **Gain models (`GainModel`):**

  | model | what it is |
  |---|---|
  | `general_no_reflections` | homogeneous-line CMEs, forward only (`cmes/base.py`) |
  | `general` | homogeneous-line forward–backward with a perturbative boundary update (`cmes/cmes_fb.py`) |
  | `bloch_no_reflections` | Bloch couplings, forward only, uniform devices only, heuristic guards (`cmes/bloch.py`) |
  | `bloch` | exact Bloch forward–backward with edge basis and Anderson acceleration; uniform (`bloch_fb.py`) and segmented (`bloch_fb_segmented.py`) |

- **Tests:** 89 pass, 9 fail; causes in WP1.
- **Static checks:** mypy reports 37 errors; pylint scores 7.68/10, with 803 invalid-name warnings from physics
  symbols.
- **Uncommitted edge-basis fix** (2026-11-04) in `analysis.py`, `cmes/bloch.py` and
  `tests/test_bloch_edge_basis.py`:
  - **What it fixes:** a mode at a zone-folding point of an unfolded supercell (splitting 0, no gap) is no longer
    put in the fixed edge basis. Without the fix, contra-directional QPM conversion collapsed from 0.85 to 0.02.
  - **New code:** `passband_mask`, `in_stopband`, `STOPBAND_TOL`, an `outside` entry list, and 4 new tests.
- **Uncommitted edits by Pietro** to `docs/source/notebooks/imskit_v3_holotype.json` (L 50.5→55.2 pH, Lf
  0.02→0.039 nH) and to several docs notebooks and scripts.

---

## WP0: Decisions needed from Pietro (blocking WP3 to WP5 and WP7)

- [x] **D1. `general` forward–backward model** (`cmes_fb.py` plus its `analysis.py` branch, about 1,850 lines).
  - **Against keeping it:** it is the least accurate model, has no convergence flags, no NaN detection and no
    tests, uses clamped perturbative boundary updates, and has a Manley–Rowe-breaking cutoff (WP2).
  - **What only it provides:** segmentations whose segments are not whole supercells.
  - **Options:** remove it, or deprecate it for one release, then remove it.
  **Decided 2026-11-06: remove it.**
- [x] **D2. `bloch_no_reflections`.**
  - **Against keeping it:** it covers uniform devices only, depends on the heuristic guard stack, and exists only
    as a fast Bloch-coupling estimate.
  - **Options:** remove it, or merge it into one forward-only kernel shared with `general_no_reflections`
    (Bloch coefficient tables as input).
  **Decided 2026-11-06: remove it.** The final package has two models:
  - **`simple`** (renamed from `general_no_reflections`): the traditional homogeneous-line CMEs, forward-only and
    fast. Keep both of its kernels: the uniform one is 1.4× faster than the segmented one, and the segmented one
    interpolates linearly between segment midpoints.
  - **`bloch`:** the segmented forward–backward kernel only; uniform devices are the one-segment case.
- [x] **D3. `quantum/`.** It has about 1.8k lines, 0 % coverage and no API docs, and only the untracked
  `quantum.ipynb` uses it. Options: leave it out of the first release, or ship it as
  `twpasolver.experimental.quantum` with a warning and a smoke test. If it ships, it still needs
  `edge_guard_mode_weights` (see WP3.4).
  **Decided 2026-11-06 (revised the same day):** `quantum/` is removed from `bloch-cmes` together with its
  reactive-angle guard (`edge_guard_mode_weights`, `BLOCH_EDGE_GUARD_DEG`). It stays on `edge-basis` and comes back
  in a later update, adapted to the new API: it used the uniform `bloch_fb_cmes_solve`, `bloch_mode_tables`,
  `process_weights` and the private `_prepare_cme_inputs`/`_build_port_tables`.
- [x] **D4. `simulate_stopband_modes=False`** (the guard path superseded by the edge basis; it only buys about
  2.5× speed). Keep it as a fast option, or remove it. **Decided 2026-11-06: keep it** (large speed-up).
- [x] **D5. Holotype JSON.** Is the uncommitted edit of `docs/source/notebooks/imskit_v3_holotype.json` intended
  for the docs? Either way the tests get frozen fixtures (WP1.2).
  **Decided 2026-11-06:** yes, the edited JSON (unloaded cells L 55.2 pH, Lf 0.039 nH) is the docs model,
  committed on `edge-basis`. For WP1.2 the tests should use a frozen fixture of this model; re-derive the
  hard-coded edge frequencies of the 6 older edge-basis tests for it.
- [x] **D6. Thesis scripts and data in `docs/source/notebooks/`** (`*_canon.py/json/pkl`,
  `edge_basis_thesis_figure*.py`, `chapter_figures_canon.py`, `fig_segmented_random_L_canon.py`, `figures/`,
  `figures2/`, `bloch_data/`, `edge_basis_data/`, `benchmark_models*`, `loss_sweep.pkl`, `quantum_vs_loss.*`).
  Where should they move: a folder in the thesis repo, or `notebooks/twpasolver_thesis/` in the workspace?
  **Done 2026-11-06:** everything except notebooks 01-04, `Capacitor.json` and `imskit_v3_holotype.json` moved to
  `/home/pietro/work/phd/twpas/notebooks/amplification/twpasolver_thesis/` (see Log).
- [ ] **D7. Release target:** GitHub plus docs only, or also PyPI? Version number (suggested `0.1.0`)? Python
  versions to support (suggested ≥3.10, with no upper bound)?
- [ ] **D8. Public API naming.** Is it acceptable to break `gain()` return keys and `FBSolverConfig` options
  (see WP4.2)? Downstream notebooks would need updating.

## WP1: Baseline: green tests and a clean tree (do first)

- [ ] **1.1 Commit the edge-basis fold-point fix** as its own commit, after Pietro confirms. Its tests:
  `test_passband_mask_ignores_touching_bands`, `test_folding_point_*`, `test_contra_qpm_survives_edge_basis`.
- [x] **1.2 Frozen fixtures.** Copy the holotype JSON into `tests/data/` and point `HOLOTYPE` there
  (`tests/test_bloch.py:510`; about 15 tests use it, and today they `skipif` silently if the docs file moves).
  - The current test file needs **both** JSON versions: the 6 older edge-basis tests pass only with the HEAD
    version, the new fold-point and contra tests only with the edited one.
  - Either keep two fixtures (`holotype_v3_head.json`, `holotype_v3_newsim.json`), or re-derive the hard-coded
    edge frequencies of one set for a single fixture. Re-deriving is preferred.
- [x] **1.3 Stale tests.**
  - Delete `test_segmented_not_supported`: segmented devices are supported now.
  - Rewrite or delete `test_edge_guard_removes_spurious_conversion_gain`: the guard was replaced by the edge basis.
- [x] **1.4 `test_mismatch_filter_keeps_partner_terms_together[general]`** is a real bug; fix it in WP2.1, or
  mark it xfail until then.
- [x] **1.5 `.gitignore`:** add `Claude outputs/`, `docs/source/api-reference/` (regenerated by `conf.py`),
  `*.pkl`, `*.npz`. Move `Claude outputs/` out of the repo after asking Pietro.
- [x] **1.6 Move thesis and scratch files out of `docs/source/notebooks/`** (per D6). Done 2026-11-06, see Log.
  - Delete `04_generalized_cmes copy.ipynb`; delete or move `bloch_test.ipynb`, `test_ports.ipynb` and
    `models_benchmark.ipynb`.
  - Delete `src/**/.ipynb_checkpoints` (ignored by git, but they clutter the source tree).
- [x] **1.7 Notebook outputs:** strip outputs from tracked notebooks 04 (1.07 MB) and 06 (1.4 MB), and add an
  nbstripout pre-commit hook (or decide on stored outputs, see WP7.4).

**Acceptance:** pytest is all green, and `git status` shows no untracked junk.

## WP2: Bug fixes, no restructuring (low risk)

Each fix gets a test.

### Physics and solvers

- [x] **2.1 Asymmetric cutoff in `cmes_fb._filter_wm_terms`** (`cmes_fb.py:~111`). It normalises the mismatch
  by the target mode's k, so partner terms of one process are kept or dropped separately, which breaks
  Manley–Rowe. Use the process-wide `k_proc` as `_filter_wm_terms_bloch` does. (Moot if D1 removes the model.)
- [ ] **2.2 4WM Kerr factor in `phase_matching`** (`analysis.py:~1672`, `~1699-1705`). `pump_coeff=2` is applied
  only to the linear k_p, so |Δk_Kerr| = 3χk_p on a flat line; the CMEs give 2χk_p (weak tones k(1+2χ), pump
  k(1+χ), χ = ξI_p²/8). `notebooks/amplification/uniform_line.py` has a reference implementation.
- [ ] **2.3 k unwrap above resonator notches.** `_bloch_params_for_cell` (`analysis.py:~528`) unwraps 2·Im γ and
  halves it, so k is fixed only modulo π per supercell. Above an LC notch (cos kd passes +1), k lands on
  k + π/d.
  - Reference fix: `notebooks/amplification/dispersion_strategies.py::fix_k` picks 2πm ± arccos((A+D)/2) closest
    to the low-frequency linear dispersion.
  - Test it on a cell with a series/shunt LC resonator (the Malnou 2026 4WM rpm line).
- [ ] **2.4 `m_max` and the mismatch cutoff are counted per supercell** (2π/D, π/D) and `kd_pi` uses the
  supercell `n_cells` (`analysis.py:~1221-1235`). An unfolded supercell (q repeated periods) silently loses the
  base period's harmonics with `m_max='auto'`: the 102-cell holotype differs by dB from the 34-cell one.
  - **Fix:** detect the primitive period of the linear cells, or at least document this and scale the auto base
    by q.
  - The auto window also falls back to a hardcoded 0.10 when `edge_threshold=None` (`~1234`).
- [x] **2.5 Silent NaN.** The rk45 step cap (`rk45.py:~182`) produces NaN; the no-reflections and `general`
  models return NaN gain without a warning. Add a warning, or a `converged`/`finite` flag on every model.
  - **Done 2026-11-06 (branch `audit-fixes`):** the cap is an `rk45_solve` argument (`max_steps_per_cell`).
    `bloch` keeps 1 per cell (`SOLVER_MAX_STEPS_PER_CELL`); `simple` uses `SIMPLE_MAX_STEPS_PER_CELL = 10`,
    because it keeps every mixing term unfiltered and strongly mismatched terms need a few steps per cell.
    Case: uniform LCLf, N=2000, `create_extended_3wm(n_pump_harmonics=2, nfc=1)`, pump 18 GHz, signals 8.5/8.8
    GHz gave `[nan, nan]` with cap 1. `no_reflections_cmes_solve` now logs a warning for any NaN point.
  - **Superseded with 2.6:** one `RKSolverConfig.max_steps_per_cell = 10` for both models. Cap 1 vs 10 is
    bit-identical (max |ΔG| = 0) on the holotype `ext2` set: uniform edge basis, native, `simulate_stopband_modes
    =False`, segmented, and `simple`; same convergence and NaN pattern (native edge points still blow up).
- [x] **2.6 Global solver constants are frozen into cached numba kernels.** Changing `SOLVER_MAX_STEPS_PER_CELL`
  and similar has no effect until `__pycache__` is wiped. Remove `configure_solver`/`get_solver_config`
  (`solver_config.py:81-115`), which are never called and misleading, and pass tolerances as arguments.
  - **Audit 2026-11-06: wider than constants.** Numba re-checks only the caller's own file, so editing a callee
    in another module (`rk45.py`, `bloch_common.py`, `solver_config.py`) also keeps the stale kernels in
    `base.py`/`bloch_fb.py` (reproduced with a toy two-file kernel). Fix: pass constants as arguments, and set
    `numba.config.CACHE_DIR` to a folder keyed by a hash of `cmes/*.py` in `cmes/__init__.py`.
  - **Done 2026-11-06 (branch `audit-fixes`):**
    - `RKSolverConfig` (`atol`, `rtol`, `first_step`, `max_steps_per_cell`) replaces the `SOLVER_*` constants. It
      is passed to every kernel as an argument (`rk` tuple; `gain(rk_config=...)`), so changes act at once. The
      dead `max_step` plumbing of `bloch_fb` is gone (its slot carries `rk`).
    - `twpasolver/__init__.py::_drop_stale_numba_cache` hashes the modules with `@njit` decorators and deletes
      their `__pycache__/*.nb[ic]` when the hash changes (stamp `__pycache__/numba_sources.sha1`). A user-set
      `NUMBA_CACHE_DIR` is not managed. Test: `tests/test_numba_cache.py`.
    - `FBSolverConfig` is renamed `BlochSolverConfig` and `gain(fb_config=)` is `gain(bloch_config=)`, with no
      alias (the Bloch CMEs were never published).

### API bugs (confirmed by running them)

- [x] **2.7 2-D `initial_amplitudes_*` crash `general` and `bloch`.** `y0_fwd[signal_idx]` at
  `analysis.py:~2030` and `~2458` picks a row instead of a column; use `y0_*_broadcast[:, signal_idx]`.
  - Done 2026-11-06 (audit-fixes): `bloch` takes the signal column of the broadcast ICs. Test `test_bloch_2d_initial_amplitudes`.
- [x] **2.8 HDF5 save fails for every forward–backward result:** `gains_all_passes` is a list of dicts. Store it
  as arrays.
  - Done 2026-11-06 (audit-fixes) with 2.29. Test `test_twpanalysis_end_to_end`.
- [ ] **2.9 `phase_matching` on segmented devices raises `IndexError`** (k is 2-D there). It also fills
  `np.empty` with garbage when `gamma` is None (`~1663`).
- [x] **2.10 `k_star` fit guard:** `len(freqs_mask) > 2` (`~715`) should be `freqs_mask.sum() > 2`.
  - Done 2026-11-06 (audit-fixes). Test `test_k_star_guard_with_coarse_grid`.
- [ ] **2.11 Stopband detection from argmin/argmax of `diff(S21_db)`** (`~665-670`) can return start > end; it
  gave `[19.13, 14.07]` on the device in `tests/test_bloch.py`. That reversed pair feeds the default ranges and
  the pump estimate. Use the supercell half-trace (`passband_mask`/`in_stopband` from the edge-basis fix).
  - **Audit 2026-11-06:** the result depends on where the grid ends. Same device: `f_arange` to 20 GHz gives
    `[19.06, 14.16]`, to 10 GHz `[6.79, 7.50]`, to 40 GHz `[39.55, 39.54]`. `_estimate_optimal_pump` then works
    on the single point `freqs[idx:-1]`.
- [ ] **2.12 Base data is never computed automatically.** `gain()` on a fresh `TWPAnalysis` raises
  `KeyError 'optimal_pump_freq'`, and edits to `twpa` leave stale data without warning. Call `update_base_data()`
  from the decorator, as it did before 6097a93. Exclude `Ip0` from the state check.
- [ ] **2.13 `gain(Ip0=...)` permanently sets `self.twpa.Ip0`** (`~1816`). Use a local value instead.
- [ ] **2.14 Shared mutable default** `fb_config=FBSolverConfig()` (`~1738`): use `None`.
- [x] **2.15 `parameter_sweep` `log.warning` format** has an argument but no `%s` (`~414`).
  - Done 2026-11-06 (audit-fixes) with 2.32. Test `test_parameter_sweep_target_in_result_keys`.
- [x] **2.16 The decorator's overwrite message** checks `function_name` instead of `save_name` (`~368`); line
  `~375` is dead.
  - Done 2026-11-06 (audit-fixes). Test `test_overwrite_message_checks_save_name`.
- [x] **2.17 `bandwidth()` with no stored gain** raises a `TypeError`; with kwargs it overwrites `data["gain"]`.
  - Done 2026-11-06 (audit-fixes): no stored gain and no kwargs raises `ValueError`; all-NaN gain raises; `nanargmax` for partial NaN. The kwargs path still re-runs `gain()` (unchanged). Test `test_bandwidth_errors`.
- [ ] **2.18 `thin`** is a point count in `gain` (zero points if `N_tot < thin`) but a stride in
  `phase_matching`. Make both a stride, or rename one of them.
- [x] **2.19 `TWPAnalysis(freqs_arange=...)`** (a typo) is silently ignored and gives an empty grid. Set
  `extra="forbid"` on the pydantic models. The unused fixture at `tests/test_analysis.py:30` has exactly this typo.
  - Done 2026-11-06 (audit-fixes): `extra="forbid"` on `TWPAnalysis` and `Frequencies` only (circuit models unchanged); dead fixture removed. Test `test_twpanalysis_rejects_unknown_fields`.
- [ ] **2.20 Likely, not run:** in segmented `general` forward–backward runs, the backward carrier reverses
  `gammas` but not the segment midpoints (`~1968-1988`). This affects only the phase of `I_tuples_bwd`. Moot if
  D1 removes the model.

### Models and modes

- [x] **2.21 `TWPA` slicing drops parameters** (`modelarray.py:53`): `TWPA(Istar=1e-3, N=7)[0:2]` returns
  default `Istar` and `N=1`.
  - **Audit 2026-11-06:** also `OnePortArray(parallel=True, twoport_parallel=True)[0:2]` comes back as a series
    array. Fix: `return self.model_copy(update={"cells": self.cells[indices]})`.
  - Done 2026-11-06 (audit-fixes): `model_copy` keeps every field (incl. `N`) except `cells`. Test `test_twpa_slice_keeps_parameters`.
- [x] **2.22 `validate_nested_cells` mutates the caller's list** (`modelarray.py:39`).
  - Done 2026-11-06 (audit-fixes). Test `test_nested_cells_do_not_mutate_input`.
- [x] **2.23 Small fixes:** `matrices_arrays.py:99` (`and` should be `or`), `bonus_types.py:41` (missing `f`
  prefix), and `Frequencies.f` silently prefers `f_arange` when `f_list` is also set (raise instead).
  - Done 2026-11-06 (audit-fixes). Tests `test_abcd_setter_rejects_wrong_length`, `test_impedance_must_have_positive_real_part`, `test_frequencies_validation`.
- [ ] **2.24 `modes_rwa._validate_modes` checks single characters** (`~553`), so multi-character mode names on
  the right-hand side fail ("Mode 2 not found" for `p2-s`).
- [x] **2.25 Cyclic relations** are accepted silently (empty expressions); `_compute_substitutions` (`~315`) can
  loop forever. Detect cycles and raise.
  - Done 2026-11-06 (audit-fixes): networkx cycle check in `RWAAnalyzer._compute_substitutions`. Test `test_cyclic_relations_raise`.
- [ ] **2.26 `Mode.__eq__`/`__hash__`:** eq ignores the label but hash uses it (`~204/216`).
- [x] **2.27 `twpa_cells.py:94,103`:** complex `C` is passed where the annotation says float (mypy).
  - Done 2026-11-06 (audit-fixes): `LCLf_abcd(C: complex)`; mypy 22 -> 20.

### Audit 2026-11-06 (new findings, `/ponytail-audit` on `bloch-cmes` daa4246)

Confirmed by running unless marked "read". Line numbers refer to `bloch-cmes` daa4246.

- [x] **2.28 NaN iterates flagged converged with `anderson_depth=0`** (`bloch_common._convergence_step`):
  `nan > n_fwd` is False, so NaN passes counted as stable and `converged=True` after 3 passes. The Anderson
  branch caught NaN (`_relative_residual` returns inf) but kept iterating NaN for every remaining pass.
  - **Done 2026-11-06 (branch `audit-fixes`):** the plain iteration is removed (`anderson_depth` must be >= 1,
    `_convergence_step` deleted). A non-finite residual ends the iteration at once with `converged=False`, so
    the existing warning reports it and no time is spent on NaN passes. Tests:
    `test_nan_iterate_stops_unconverged`, `test_result_does_not_depend_on_anderson_depth` (depth 1 vs 5).
- [x] **2.29 A failed HDF5 save destroys the previous file** (`file_utils.save_to_hdf5`: mode `"w"` empties it
  first). Besides 2.8, these also fail half-way: `phase_matching` results (`triplets["k"]` mixes the (1,)
  `pump_k` with scalars, ragged array) and `parameter_sweep` over `gain` (lists of `mode_info` dicts; notebook 03
  does this). Int dict keys reload as strings, so `plot_mode_currents` loses its labels after `load_data`.
  Fix: write to `name + ".tmp"` then `os.replace`; save lists of dicts as index-keyed groups; `float(pump_k)`.
  - Done 2026-11-06 (audit-fixes): tmp file + `os.replace`; lists of dicts, strings or ragged items saved as index groups (attr `list`), all-int keys marked (attr `int_keys`); `pump_k` stored as a scalar. Tests `test_twpanalysis_end_to_end`, `test_parameter_sweep_over_gain_saves`, `test_digit_keyed_dict_stays_dict`, `test_failed_save_keeps_previous_file`.
- [x] **2.30 `update_base_data` ignores grid, unit and port changes** (`analysis.py:650-656`, state is only
  `twpa.model_dump()`). After changing `f_arange`, `data["freqs"]` keeps the old 1995 points while `a.f` has 995.
  Fix: include `(f_arange, f_list, unit, port)` in the state.
  - Done 2026-11-06 (audit-fixes): state is `(twpa, f_list, f_arange, unit)`; ports are read at `gain()` time and are not part of base data. Test `test_update_base_data_follows_grid`.
- [x] **2.31 `phase_matching` range arguments** (`analysis.py:1358-1378`): `pump_arange=[13, 14, 15]` gives only
  `[13]` (treated as an arange); an explicit `(5, 10, 0.1)` is thinned by `thin=20` to 3 points; `pump_Ip0=0.0`
  is falsy, so Kerr stays on. Fix: only a tuple is an arange, do not thin explicit ranges, `is not None`.
  - Done 2026-11-06 (audit-fixes). Tests `test_phase_matching_explicit_ranges`, `test_phase_matching_zero_pump_is_linear`.
- [x] **2.32 `parameter_sweep` `KeyError`** (`analysis.py:412-430`) when the swept name is also a result key
  (`parameter_sweep("gain", "signal_freqs", ...)`): it appends to `<name>_sweeped`, which is never created.
  Fix with 2.15: `results = {}`, then `results[target_savename] = []` after the collision check.
  - Done 2026-11-06 (audit-fixes). Test `test_parameter_sweep_target_in_result_keys`.
- [x] **2.33 `__array__` is not NumPy-2 compliant** (`matrices_arrays.py:39-41`):
  `np.asarray(cell.abcd, dtype=complex)` raises `TypeError`; `np.array(cell.abcd)` warns. Fix:
  `def __array__(self, dtype=None, copy=None): return np.array(self._matarray, dtype=dtype, copy=copy)`.
  - Done 2026-11-06 (audit-fixes). Test `test_abcd_array_protocol`.
- [x] **2.34 `.s` vs `.to_network().s` disagree for complex Z0** (`mathutils.a2s` uses pseudo-waves, skrf
  power waves): 0.53 apart for `Z0_ref=30+20j`, 1.6e-16 for real Z0. Fix: `s_def="pseudo"` in `to_network`.
  - Done 2026-11-06 (audit-fixes): `to_network` passes our pseudo-wave S with `s_def="pseudo"` (skrf converts `a=` before `z0` is set); `.z` agrees too, skrf's own `.a` still assumes power waves. Test `test_network_s_matches_for_complex_z0`.
- [ ] **2.35 Small ones:**
  - `N=0` and `model * 0` crash on a bare `assert` in `matpow_2x2` (`N` is `NonNegativeInt`).
  - `Z0=0` passes the validator (`bonus_types.py:40`) and then fails the bare assert in `a2s`/`s2a`.
  - `Resistance.Z` truncates R on integer frequency arrays (`np.full_like`, `oneport.py:119`).
  - `TwoPortCell.interpolate` defaults to polar interpolation, which is wrong for real sign-changing entries
    (lossless TL: error 0.16 vs 0.0099 for `polar=False`).
  - `Frequencies(f_arange=(1, 2, 0))` divides by zero (step typed `NonNegativeFloat`).
  - A list-valued `port` fills only `Zout`; its ABCD stays the identity (`analysis.py:643-646`), read only.
  - CI venv cache key (`analysis.yml:26-27`) ignores `matrix.python-version`. **Done:** key has OS and Python.
  - `warmup` with `passes=1` logs a "did not converge" warning every time. **Done** with 3.8.
  - Done 2026-11-06 (audit-fixes) for `N=0` (identity), `Z0=0`, `Resistance.Z`, arange step > 0 (tests `test_zero_repetitions_is_identity`, `test_impedance_must_have_positive_real_part`, `test_resistance_on_integer_freqs`, `test_frequencies_validation`). Still open: polar `interpolate`, list-valued `port`, CI cache key, warmup warning.
- [x] **2.36 Test for `TWPAnalysis`:** no test runs `phase_matching`, `bandwidth`, stopband detection or a save and
  reload of a real result; the `twpanalysis_instance` fixture asks for a `twpa_model` fixture that does not exist.
  One test on the 3-cell device (`gain` simple and bloch, `phase_matching`, `save_data`/`load_data`, stopband
  start < end) would have caught 2.11, 2.29-2.32.
  - Done 2026-11-06 (audit-fixes): `test_twpanalysis_end_to_end` (simple and bloch gain, `phase_matching`, `bandwidth`, save/load). Stopband start < end is left to 2.11.

**Acceptance:** each item has a test; the full suite is green; mypy errors go down.

## WP3: Solver consolidation (largest saving, about 1,100–1,500 lines)

Order: 3.1 → 3.2 → 3.3, then 3.4/3.5 per the decisions.

- [x] **3.1 Drop the uniform Bloch forward–backward kernel; route uniform devices through the segmented solver
  as n_seg = 1.**
  - **Evidence (2026-11-06):** the holotype uniform vs `_segmented_holotype(scale=(1.0,))` agree to 3e-11 dB at
    8 points from 4 to 12 GHz, and to 3e-12 dB at the p2s edge (9.204–9.25 GHz). Runtime is equal (0.52 vs
    0.53 s, 0.75 vs 0.66 s). This also saves about 55 s of cold compile.
  - **Duplicated pairs to collapse:**
    - `bloch_fb_rhs`/`_edge` (bloch_fb.py:322-475) and `bloch_fb_segment_rhs`/`_edge` (seg:239-402);
    - `_edge_sources` and `_edge_sources_segmented`;
    - `_edge_bvp_solve` and `_edge_bvp_solve_segmented`;
    - the pass loop and Anderson/convergence block (bloch_fb.py:858-1118 and seg:1093-1320);
    - the wrapper closures `_table`/`_apcz_tof`/`_weights` (bloch_fb.py:1238-1278, seg:1453-1493, and
      `cmes_fb.py:1545-1585` as `_as_nfm`).
  - **Ring buffer:** the uniform kernel changes the Anderson ring-buffer `depth` every pass (`bloch_fb.py:893`);
    the segmented one keeps it constant. Keep the segmented behaviour.
  - Move the shared numba helpers (stencils, mixing-term filter, row coefficients, expm, Anderson step, residual)
    into `cmes/bloch_common.py`.
  - The quantum background (`quantum/background.py`) calls the uniform pieces; it needs an adapter (or see D3).
- [x] **3.2 `analysis.py` uniform/segmented merge.**
  - `_prepare_bloch_coefficients` (`~1071`) → `_prepare_bloch_segmented` (`~1505`) with n_seg = 1.
  - `_apply_edge_basis` (`~1273`) → `_apply_edge_basis_segmented` (`~1376`).
  - Stop computing the unused `coeffs_3wm`/`coeffs_4wm` on the forward–backward path (rebuilt at `~1356-1360`
    and never used by the kernel).
- [ ] **3.3 One boundary-table helper, one forward–backward post-processing helper** (per-pass split, port gains,
  prompt reflections, `mode_info`), shared by every forward–backward driver.
- [x] **3.4 Guard systems** (per D2/D3/D4). Mostly done 2026-11-06: the `bloch_no_reflections` guard stack is
  removed and `process_weights` is the single N-d version. With `quantum/` gone (D3), the only guard left is the
  `simulate_stopband_modes=False` path (kept, D4). Leftover: the 45° reactive-angle threshold is repeated in
  `analysis.py`; make it one constant (small, do it with WP4).
  - **What exists:**
    - `_stopband_guard_weights` and `_process_weights_nd` serve `simulate_stopband_modes=False`;
    - `_bloch_mode_weights`, `_bloch_edge_guard`, `edge_proximity_mode_weights`, `cell_phase_mask` and the
      `BLOCH_EDGE_GUARD*` constants serve `bloch_no_reflections`;
    - `edge_guard_mode_weights` serves the quantum model;
    - `process_weights` (bloch.py:730) duplicates `_process_weights_nd` (`analysis.py:149`).
  - **Target:** at most one guard-weight mechanism, or none. Remove the `pw3`/`pw4` kernel plumbing if nothing
    needs it.
  - The 45° reactive-angle threshold is repeated at 4 places; make it one constant.
- [x] **3.5 Forward-only kernels:** `bloch_no_reflections_cmes` (bloch.py:775) is `no_reflections_cmes`
  (base.py:63) with the (α+jκ) prefactor folded into the coefficients. Keep one kernel with per-frequency
  coefficient tables (per D2).
- [x] **3.6 `general` model** (per D1). If removed, delete `cmes_fb.py`, `_compute_forward_backward_gain`, the
  `general` branches and options (`update_rate`, `BOUNDARY_CORR_MAX`) and `_filter_wm_terms`; also
  `_convergence_step`/`_store_pass` once `bloch_fb` no longer imports them.
- [ ] **3.9 Dead options and near-copies in `cmes/` (audit 2026-11-06, about 260 lines):**
  - Never set by any caller: `edge_substeps` (`max_step` and `SOLVER_MAX_STEPS` removed with 2.6); rk45 `rtols`/`atols` arrays, the
    automatic `first_step` block, `y_size == 0` and `t_eval_size == 0` branches; `FBSolverConfig.m_max_fixed`.
  - Test-only: `bloch_fb_cmes_solve` (3 uses in `test_bloch.py`).
  - Forward and backward sweeps in `bloch_fb.py` (~90 lines each) differ only in index direction: one sweep with
    a direction flag.
  - `bloch.py`: `bloch_mode_tables` re-implements `supercell_matrix`/`_intracell_currents`; the edge walk is
    repeated in `edge_reference_frequency` and `stopband_edges`.
- [ ] **3.7 Unify defaults:** `m_max` (config 2 vs wrapper 5), `anderson_depth` (10 vs 5), and the `general`
  defaults (`passes` 3 vs 10, `kappa_cutoff` 0.1 vs 0.2, ...). Use `FBSolverConfig` as the single source.
- [x] **3.8 `warmup.py`:** also compile the Bloch kernels (today it compiles only the `general` models). Add the
  `twpasolver-warmup` script entry to `pyproject` (the README advertises it, but it exists only in old install
  metadata). Fix its 13 mypy errors (`**dict[str, object]` passed into `TWPA`, lines 100-101).
  - **Done 2026-11-06 (audit-fixes):** `[tool.poetry.scripts] twpasolver-warmup` (checked in a built wheel); the
    warmup silences the logger during its one-pass calls, so it no longer prints "did not converge".

**Acceptance:** results identical to the pre-refactor reference (≤1e-9 dB for `bloch`); the full suite is green;
cold compile time measured and noted here; the line count noted here.

## WP4: `analysis.py` restructuring and API consistency

- [ ] **4.1 Split into a package `twpasolver/analysis/`**, with `__init__` re-exporting `TWPAnalysis`,
  `GainModel` and `Analyzer`:

  | module | content |
  |---|---|
  | `base.py` | `Analyzer`, `analysis_function`, `parameter_sweep` |
  | `linear.py` | Bloch parameters, segment ABCD cache, stopbands, `k_star`, pump estimate |
  | `phase_matching.py` | `phase_matching` |
  | `cme_inputs.py` | initial states, carrier, port tables, relation coefficients (computed once per call, not three times as now) |
  | `stopbands.py` | harmonic limits, edge basis, `_fb_stopbands` |
  | `gain.py` | `GainModel`, drivers, result builder |
  | `core.py` | thin `TWPAnalysis` |

  - Make `_prepare_cme_inputs` and `_build_port_tables` public: `quantum/background.py:148,212` calls them.
- [ ] **4.2 Consistent result dict across models** (per D8):
  - every model returns the same keys (`I_tuples_fwd/bwd` with zeros for forward-only models, `gain_db`,
    `gain_db_11/12/22`, `converged`);
  - one naming scheme (per-pass `gain_db_fwd/bwd` against top-level `gain_db/gain_db_12`; `mode_info` keys
    `signal`/`pump` in `phase_matching` against `signal_mode`/`pump_mode` in `gain`);
  - NaN-safe dB everywhere.
- [ ] **4.3 Docstrings:**
  - `Is0` is documented as 1e-6 but defaults to 1e-7 (private drivers 1e-6);
  - `fb_config` is described as "GENERAL model only" (also in the `FBSolverConfig` docstring);
  - the comment at `~1929` says `Zout = twpa.Z0_ref`, but the code uses `port_Z_ext`;
  - the `_edge_basis_entries` Returns section needs a tidy (the `outside` edit);
  - `_bloch_mode_weights` claims segmented shapes it never receives;
  - the `__init__.py` package docstring lists MINIMAL_3WM/GENERAL and omits BLOCH.
- [ ] **4.4 Remove ignored parameters or make them work:**
  - `idler_mode` only goes into metadata;
  - `update_rate` is ignored by `bloch`, and `m_max`/`edge_threshold`/`anderson_depth` by `general`;
  - `Z0` and `Ze` are both `port_Z_ext`;
  - `max_step` and `edge_substeps` are never exposed.
- [ ] **4.5 Dead code:**
  - `self._n_distinct_segments` (`~743`);
  - the unreachable `threshold is None` branches (`~1300`, `~1402`);
  - `gap` is always True (`~1250`);
  - the `edge_basis_threshold` alias in `FBSolverConfig` (only a test uses it);
  - the unused `twpanalysis_instance` fixture;
  - `mathutils.compute_phase_matching` (unused, and it has the wrong 4WM Kerr factor);
  - `FrequencyLinspace` and `nb_complex3d` (unused types).
- [ ] **4.6 Units:** user-unit frequencies, with Hz only internally. Hz constants (`1e9/4e9/8e9` spans) sit next
  to user-unit logs; express the spans in user units, or document them.
- [ ] **4.7 Residual metric:** document in the result that the forward–backward residual is relative and
  dominated by nearly empty backward envelopes (floor 1e-3 to 1e-1 is common; observables stable). Better:
  weight it by mode power. With `anderson_depth=0` the convergence test (`_convergence_step`) differs from the
  reported metric; make them one.

## WP5: models, modes and logging

- [ ] **5.1 `AnyModel` blocks user models** (`models/twoportarrays.py:167-168`). The union is built from
  `all_subclasses` at import, so a user `TwoPortModel` subclass defined later fails validation. Yet
  `docs/source/implementing_models.rst:11` promises it works, and `notebooks/contra_qpm/cqpm.py` works around it
  with `tw.cells[:] = ...`.
  - **Fix:** a registry with a rebuildable union (`model_rebuild` on subclass creation via
    `__init_subclass__`), or a custom validator dispatching on `name`.
  - Also remove the `# mypy: ignore-errors` headers (`twoportarrays.py:9`, `plotting.py:3`); the opaque union
    causes 8 mypy errors in analysis/bloch.
- [ ] **5.2 First-class nonlinear-element hook.** The Bloch code accepts `bloch_nl_elements()` on cells; document
  it, and consider an `nl_scale` field on `LCLfBaseCell`/`StubBaseCell`. That would remove the subclass in
  `contra_qpm` and allow nonlinearity-only gratings in JSON models.
- [ ] **5.3 `modes_rwa`:**
  - Replace the two parsers (`RWAAnalyzer._parse_expression` `~269`, `ModeArray._parse_expression_for_propagation`
    `~425`) and the two resolvers (sympy substitution, dict linear combination) with integer coefficient vectors.
    This drops the sympy dependency (current build: 18 modes, 1363 terms, 0.12 s).
  - Remove `n_sidebands` (`~1066`) and the unused `edge_colors`/`edge_styles`.
  - `get_dependent_modes` has no docstring and no caller.
  - No `plt.show()` in library code (`~914`); deduplicate `plot_mode_relations`/`print_symbolic_expressions`.
- [ ] **5.4 Logging:** `logger.py:39-40` sets INFO and adds a handler at import, so messages print twice under
  pytest. A library should add only a `NullHandler`; keep a helper (`twpasolver.logger.enable(level)`) for users.
- [ ] **5.7 Dead code outside `cmes/` (audit 2026-11-06, grepped incl. docs and notebooks):**
  `mathutils.compute_phase_matching` (~58 lines, also crashes when `pump_f/2 < freqs[0]`), `TwoPortCell.from_z`,
  `bonus_types.FrequencyLinspace` and `nb_complex3d`, `ModeArray.get_dependent_modes`,
  `RWAAnalyzer.update_relations`, `print_rwa_terms`, `edge_colors`/`edge_styles`, `print_symbolic_expressions`
  (repeats `plot_mode_relations`), `_n_distinct_segments`, and `Mode.__eq__`/`__hash__` (fixes 2.26 by deletion).
  `_validate_modes` can go too (fixes 2.24): `RWAAnalyzer` already raises on unknown names. The `mode_info` dict
  is built twice in `analysis.py`. `twpa_cells.py` repeats the qp-loss half-inductance block in `StubBaseCell`
  and `LCLfBaseCell`.
- [ ] **5.5 `matrices_arrays.py`:** the 12 duplicated property pairs on ABCD/S/Z arrays become a small descriptor.
  `TwoPortModel.__mul__` should use `model_copy(update=...)`.
- [ ] **5.6 `quantum/`** (per D3). Dead functions: `gaussian.expm_batch`, `magnus4_omega`,
  `transfer_to_scattering`. The Python `star`/`cascade` duplicate the numba versions in `kernels.py`;
  `linearize.py:146` uses `col` possibly before assignment.

## WP6: Packaging, CI, tooling

- [ ] **6.1 `pyproject.toml`:**
  - real description; version per D7; license as SPDX `Apache-2.0`; author email; URLs; classifiers;
  - drop the `<3.13` upper bound;
  - `networkx = "<=3.*"` resolves to 3.0: use `>=3`;
  - sympy is listed in both main and analysis (or dropped, per 5.3);
  - declare `typing_extensions` (imported);
  - drop the `CyRK` mypy override;
  - add `[tool.poetry.scripts] twpasolver-warmup = ...`;
  - `[tool.pylint]` `good-names` for physics symbols (removes most of the 803 invalid-name warnings);
  - a `[tool.pycln]` section, or fix the pre-commit argument;
  - refresh `poetry.lock` (poetry is broken locally: fix it or do this in CI).
- [ ] **6.2 CI (`.github/workflows/analysis.yml`, `deploy_docs.yml`):**
  - bump checkout/setup-python/gh-pages from v3 (deprecated Node);
  - include the Python version in the venv cache key (3.10–3.12 currently share one `.venv`);
  - run on pull requests;
  - let pylint fail on errors only;
  - mypy must pass;
  - docs: drop the pinned 3.10.4 and the apt sphinx; deploy on tags or releases only.
- [ ] **6.3 pre-commit:** isort is pinned to `9.0.0b1` (beta); the nbqa isort 5.6.4 and pyupgrade 2.7.3 pins are
  very old; add nbstripout (WP1.7).
- [ ] **6.4 Coverage:** 40 % overall. Plotting 11 %, warmup 0 %, quantum 0 %; numba code reads low because
  coverage cannot trace jitted functions. Add a CI job with `NUMBA_DISABLE_JIT=1` for a real number, if
  feasible.
- [ ] **6.5 mypy to zero:** 13 errors fixed by 3.8, 8 by 5.1, 2 by 2.27; numba types used as annotations
  (`cmes/base.py:29`) need `TYPE_CHECKING` aliases; plus about 6 ndarray shape assignments.

## WP7: Documentation and release

- [ ] **7.1 Toctree** (`docs/source/tutorials.rst`/`index.rst`) has only notebooks 01-04. 05/06 and `quantum` were moved
  out on 2026-11-06 (thesis material). New docs notebooks for segmented devices and the edge basis can be written
  later from them, without thesis dependencies.
- [ ] **7.2 Notebooks** must use the post-refactor API, and must not depend on thesis files (`palette.py`,
  `prl_aps.mplstyle`, `edge_basis_sweep.py` → keep only what the docs need).
- [ ] **7.3 API reference:** `run_apidoc` in `conf.py` writes into the repo, so either ignore its output (1.5) or
  generate it at build time only. It currently lacks bloch*, rk45, warmup and quantum.
- [ ] **7.4 Execution policy:** `nbsphinx_execute` is unset (auto), so stripped notebooks run at every build,
  which is slow given numba compile times. Either commit outputs for heavy notebooks and set `never` for them, or
  cache them. Pandoc is needed for the build and is not installed locally.
- [ ] **7.5 README:** what twpasolver is, the models and when to use which, install, warmup, a minimal example,
  citation, license.
- [ ] **7.6 Physics notes for users:** document that the forward-only `general_no_reflections` (future `simple`)
  ignores `port`, `port_out` and `port_Z_ext`: its gain is output over input inside the line, with no connector
  losses or end reflections. Either say so or warn when connectors are set. the edge-basis derivation (`edge_basis_physics.tex`, cited by
  `cmes/bloch.py`) now lives in `notebooks/amplification/twpasolver_thesis/claude_outputs/`; adapt it for the docs.
  - the edge basis, and when to set `edge_threshold`;
  - `m_max` per supercell (2.4);
  - the convergence/residual meaning (4.7);
  - the nonlinear-element hook (5.2).
- [ ] **7.7 Changelog of breaking changes** (return keys, removed models and options).
- [ ] **7.8 Branches.**
  - Reconcile `origin/main` (4 commits, the simple_4wm PR: `create_basic(three_wave=)`, `basic_4wm` array,
    placeholder relations; already present in substance on `edge-basis`). `git merge-tree` shows conflicts in
    `analysis.py` and `modes_rwa.py`; resolve by keeping the `edge-basis` side.
  - `edge-basis` has never been pushed.
  - The one commit on `edge-basis-with-guard` not in `edge-basis` (4e1a164, a docs example comparing the guard
    with the edge basis) can be dropped.
  - Delete the merged local branches (`bloch_model`, `forward-backward`, `new_cme_models`,
    `position-dependent`, `quantum_solver`) after Pietro confirms, and decide whether the stale remote branches
    go too.
- [ ] **7.9 Tag the release**; PyPI per D7.

---

## Suggested order and size

| order | WP | size | risk | depends on |
|---|---|---|---|---|
| 1 | WP1 baseline | small | low | D5, D6 |
| 2 | WP2 bug fixes | medium | low | (2.1/2.20 moot if D1 = remove) |
| 3 | WP3.1–3.3, 3.7, 3.8 | large | low–medium | WP1 |
| 4 | WP0 decisions → WP3.4–3.6 | medium | medium | D1–D4 |
| 5 | WP4 | large | medium | WP3, D8 |
| 6 | WP5 | medium | low–medium | — (5.3 is independent) |
| 7 | WP6, WP7 | medium | low | everything above |

Expected net reduction: about 1,500 lines from WP3.1–3.5 and 4.x with no physics loss, plus about 1,850 if the
`general` forward–backward model goes (D1), and about 1,800 if `quantum/` is held back (D3).

## Log

Record what was done, by which chat or branch, and anything learned (numbers, surprises).

- 2026-11-06: plan written from three parallel reviews (solvers, `analysis.py`, rest/packaging). No code changed
  except the uncommitted edge-basis fold-point fix of 2026-11-04.
- 2026-11-06, docs clean-up (uncommitted):
  - **What stays:** `docs/source/notebooks/` now holds only `01_models`, `02_twpas`, `03_analysis`,
    `04_generalized_cmes`, `Capacitor.json` and `imskit_v3_holotype.json`.
  - **What moved:** everything else, 28 MB including 05/06, quantum, bloch_test, test_ports, models_benchmark, the
    `*_canon*` thesis scripts, figures and data, to `notebooks/amplification/twpasolver_thesis/`.
  - **Moved files in git:** four were tracked (`06_edge_basis.ipynb`, `edge_basis_diagram.py`, `edge_basis_sweep.py`,
    `edge_basis_thesis_figure.py`) and now show as deleted.
  - **Changes in the moved folder:** a copy of `imskit_v3_holotype.json` (current, edited version) sits next to the
    moved scripts, and `THESIS = HERE.parents[2] / "tesi_phd" ...` was fixed in `edge_basis_thesis_figure*.py`.
  - **Notebook 04:** removed the stale `mode_directions=` argument (source only, outputs untouched).
  - **Check:** 01-04 all execute in a temporary copy.
  - **Not touched:** `tesi_phd*/tools/ch5_review/quantum_*.py` hard-code the old notebooks path. They were already
    broken (they import `quantum_loss_sweep_nb`, which no longer exists).
- 2026-11-06, WP1.2-1.4 and 2.1 on `bloch-cmes` (uncommitted): **test suite green, 98 passed, 0 skipped.**
  - **Fixture:** `tests/data/imskit_v3_holotype.json` is a frozen copy of the new docs model; `HOLOTYPE` in
    `tests/test_bloch.py` points there, and all `skipif(not HOLOTYPE.exists())` decorators were removed.
  - **Edge frequencies:** the p2s edge (pump 16 GHz) moved from 41.2109 to 41.1949 GHz. Every hard-coded signal
    near it was shifted by -16 MHz; the "splitting 0.099" point was recomputed (9.2481 GHz).
  - **Outdated tests:**
    - `test_segmented_not_supported` → `test_segmented_not_supported_without_reflections`, since only
      `bloch_no_reflections` still rejects segmented devices.
    - `test_edge_guard_removes_spurious_conversion_gain` → `test_edge_guard_switches_off_edge_conversion`. With
      the new parameters the unguarded run no longer shows the spurious gain (≤0.08 dB over 9.10-9.22 GHz), so
      only the guard itself is checked.
  - **Bug fix, `cmes_fb._filter_wm_terms`:** the threshold is now `cutoff * max|kappa|` over the modes of the
    process (as in the Bloch filter), so partner terms are kept together. This changes `general`-model results
    slightly where the target is not the largest-|k| mode of a process (more terms are kept).
- 2026-11-06, WP1.5-1.7 on `bloch-cmes`:
  - `.gitignore` now covers `docs/source/api-reference/`, `*.pkl`, `*.npz` and `Claude outputs/`.
  - `Claude outputs/` moved to `notebooks/amplification/twpasolver_thesis/claude_outputs/`; the comment in
    `cmes/bloch.py` that cited it was updated.
  - Stale `.ipynb_checkpoints` folders deleted from the repo root, `src/` and `tests/`.
  - Notebooks 01-04 carry no outputs, and nbstripout already runs in pre-commit.
  - **WP1 complete:** green suite, clean tree.
- 2026-11-06, benchmark for WP3.1 (warm numba, best of 3, uniform holotype vs a 1-segment segmented copy; script:
  `/tmp/claude-1000/bench/seg_vs_uni.py`):
  - **Speed:** segmented/uniform time ratio is 1.00-1.04 for 9 modes: 10 or 200 signals, fixed 20 passes or the
    default config, fine x-grid (`thin=10`), and the p2s edge with the edge basis. Only tiny runs show fixed
    setup overhead: 3 modes and 10 points take 10 ms against 7 ms (ratio 1.5). The interpolation in the
    segmented kernel costs nothing measurable.
  - **Results:** identical (≤5e-10 dB) wherever the iteration converges. With the 3-mode `basic` array they
    differ by up to 0.18 dB only at non-converged points (fs 3.0-3.6 GHz, idler 12.4-13 GHz just below the first
    gap, residuals 0.05-0.4), because the uniform kernel reindexes its Anderson history every pass. Keep the
    segmented behaviour.
  - **Side finding:** `basic` (p, s, i) fails to converge there with the default config. Check this after WP3.1.
- 2026-11-06, benchmark of the models (holotype, `ext2`, 200 signals, warm numba; scripts
  `/tmp/claude-1000/bench/models.py`, `gnr_seg.py`):

  | model | time | vs `bloch` (median / max \|ΔG\|) |
  |---|---|---|
  | `general_no_reflections` | 0.09 s | 0.8 / 4.2 dB |
  | `bloch_no_reflections` | 0.13 s | 4.1 / 6.6 dB |
  | `general` | 2.9 s | 5.2 / 8.3 dB |
  | `bloch` | 3.9 s | reference |

  - **Downstream use:** `bloch_no_reflections` is used nowhere. `general` appears only in the Ch. 6 CME-layer figure
    (`plot_cme_layers*.py`) and in `kitwpac_fc.ipynb`.
  - **Leaning (Pietro, not final):** keep `general_no_reflections`, as the "traditional" homogeneous-line CMEs
    (textbook baseline, fast, forward-only, any segmentation; couplings are not exact), and `bloch`. Drop `general`
    and `bloch_no_reflections`.
  - **`general_no_reflections` segmented vs uniform:** the segmented kernel is 35-40 % slower (0.091 → 0.125 s at
    200 signals, 0.48 → 0.67 s at 1000), independently of the number of segments. The cause is the per-step
    interpolation of α, κ, couplings and phase in `no_reflections_cmes_segmented`. Results are identical for 1 or 4
    segments.
  - **Keep both kernels for this model:** the uniform path is about 85 lines and buys a 1.4× speed-up on the fast
    model (unlike Bloch, where the uniform solver is 850 lines with no speed gain).
  - **Physics difference:** the segmented `general_no_reflections` interpolates k and couplings linearly between
    segment midpoints (a smooth taper), while segmented `bloch` uses piecewise-constant segments with exact
    junctions. Document this.
  - **Helper caveat:** `_segmented_holotype` uses `N // n_seg`, so 36 segments make a line 1 % shorter (1188
    supercells).
- 2026-11-06, WP3.1 on `bloch-cmes`:
  - **What changed:** the uniform Bloch forward–backward kernel is deleted. `cmes/bloch_fb.py` is now the former
    segmented module (renamed with `git mv`, with a merged module docstring), plus a thin wrapper
    `bloch_fb_cmes_solve` that adds a one-segment axis for uniform devices. Its extra parameters are keyword-only.
    The shared numba helpers moved to the new `cmes/bloch_common.py`, together with `_convergence_step` and
    `_store_pass` taken from `cmes_fb.py`.
  - **Unchanged callers:** `analysis.py` (uniform path), `quantum/background.py` and the tests still call
    `bloch_fb_cmes_solve`. The tests that used the uniform RHS or the uniform edge BVP now call the segmented
    versions with one segment, which is mathematically identical.
  - **Regression** (`/tmp/claude-1000/wp31/regress.py`):
    - uniform holotype: default, fixed-pass and guard configurations agree to ≤1e-9 dB;
    - the 4-segment device and the quantum response are bit-identical;
    - the native-basis run differs by 2e-7 dB and contra-QPM by 7e-4 relative, both only at non-converged points
      (the Anderson iteration path differs).
  - **Size and compile time:** `cmes/` lost about 810 lines. A cold numba compile of the Bloch forward–backward
    model now takes 76 s and covers uniform and segmented devices; before, the two kernels compiled separately
    (about 55 s and 80 s).
  - **Next:** WP3.2 (one preparation path in `analysis.py`). Then remove `general` and `bloch_no_reflections`
    (WP3.4–3.6) and rename `general_no_reflections` to `simple`.
- 2026-11-06, WP3.2 on `bloch-cmes`:
  - **One preparation path:** the Bloch forward–backward driver now has a single path. `_bloch_segments()` returns
    `[self.twpa]` with bounds `[0, N_tot]` for a uniform device, the k table gets a length-1 segment axis, and
    `_prepare_bloch_segmented`, `_apply_edge_basis_segmented` and `_fb_stopbands` run unchanged.
  - **Removed:** `_apply_edge_basis` (uniform) and the uniform branches of `_fb_stopbands` and the driver.
  - **Result dict:** `_drop_segment_axis` removes the length-1 axis, so uniform devices keep their shapes. The dict
    loses `coeffs_3wm`, `coeffs_4wm` and `gamma_exact` (unused by the forward–backward kernel; no downstream reader)
    and gains `edge_mode` and `seg_bounds`.
  - **Numerics:** identical to WP3.1 on the whole regression set.
  - **Still to remove:** `_prepare_bloch_coefficients` stays only for `bloch_no_reflections` and the RHS energy
    tests; it goes with that model (WP3.4–3.6), and the energy tests then use the segmented preparation.
- 2026-11-06, connectors test: `test_fb_linear_limit_with_connectors` (uniform and 4-segment holotype) checks that
  the unpumped Bloch forward–backward model with different lossy connectors at the two ends reproduces S21, S11,
  S12 and S22 of the cascade connector–device–reversed connector.
  - **Agreement:** S21/S12 within 6e-5 dB, S11/S22 within 5e-3 dB.
  - **Sensitivity:** ignoring the connectors would be off by 0.5 dB on S21 and 12 dB on S11/S22; reversing the
    output connector by 0.1 dB and 3-15 dB.
  - **No test covered connectors before.** Finding: `general_no_reflections` ignores connectors (WP7.6).
- 2026-11-06, WP3.4–3.6 and the rename, on `bloch-cmes`:
  - **Two models left:** `simple` (renamed from `general_no_reflections`; the old string still works with a
    deprecation warning, via `GainModel._missing_`) and `bloch`. Default model: `simple`. `gain(fb_config=None)`
    replaces the shared mutable default.
  - **Removed models:**
    - `general`: `cmes/cmes_fb.py` (1,628 lines), its driver, `update_rate`, `BOUNDARY_CORR_MAX`.
    - `bloch_no_reflections`: its kernel and `bloch_coefficients`, `cell_phase_mask`,
      `edge_proximity_mode_weights`, `_bloch_mode_weights`, `_bloch_edge_guard`, `_prepare_bloch_coefficients`,
      `BLOCH_EDGE_GUARD`, `BLOCH_CELL_PHASE_CUTOFF`.
  - **Also removed:** the never-called `configure_solver`/`get_solver_config`, the `edge_basis_threshold` alias,
    `as_solver_args`, `_process_weights_nd` (now `bloch.process_weights`, N-d), and the dead `minimal_3wm` branch
    of `plot_mode_currents`.
  - **Kept:** `edge_guard_mode_weights` and `BLOCH_EDGE_GUARD_DEG`, used by `quantum/`.
  - **Other files:**
    - `cmes/bloch.py` is now pure NumPy (linear tables, edge basis, weights).
    - `warmup.py` compiles `simple` (uniform and segmented) and `bloch`; built with `TWPA.model_validate`, which
      removes its 13 mypy errors.
    - The solver's non-convergence warning no longer says "Segmented" for uniform devices.
  - **Tests:** 95 pass. Removed: `test_energy_conservation_lossless` (no-reflections), the segmented-rejection test,
    two no-reflections guard tests, and the `general` case of the mismatch-filter test. The energy tests now build
    uniform tables with `_uniform_bloch_tables` (segmented preparation plus `_drop_segment_axis`).
  - **Notebook 04:** the model section is rewritten for `simple` and `bloch`, the Floquet text now describes exact
    junctions, and `FBSolverConfig` uses 40 passes and a 1e-2 threshold (the old 10 passes and 1e-4 left every
    `bloch` call unconverged). Notebooks 01-04 execute without warnings.
  - **Regression:** bit-identical for `simple` (uniform and segmented, `basic_3wm` and `ext2`), `bloch` (uniform and
    segmented, edge basis and guard option), the default model and `quantum/`.
  - **Size and checks:** `src/` is 11,330 lines (14,820 at the start of the clean-up); mypy is down from 37 to 24
    errors.
  - **Downstream:** scripts using `model="general"` (Ch. 6 `plot_cme_layers*.py`, `kitwpac_fc.ipynb`) need the
    `edge-basis` branch.
- 2026-11-06, `quantum/` removed (D3 revised) on `bloch-cmes`: `src/twpasolver/quantum/` (1,859 lines),
  `edge_guard_mode_weights` and `BLOCH_EDGE_GUARD_DEG`. `test_edge_guard_weights` became `test_reactive_angle`.
  - **Checks:** 95 tests pass; `simple` and `bloch` are bit-identical on the regression set.
  - **Kept:** the uniform wrapper `bloch_fb_cmes_solve` (used by the tests, and a convenient entry point for uniform
    devices).

- 2026-11-06, `/ponytail-audit` of `bloch-cmes` (daa4246): new items 2.28-2.36, 3.9, 5.7, and notes on 2.6,
  2.11, 2.21. Branch `audit-fixes` (from daa4246) fixes 2.5 and 2.28:
  - **2.28:** plain iteration (`anderson_depth=0`) removed; a NaN/inf residual stops the point unconverged.
  - **2.5:** step cap is an `rk45_solve` argument; `simple` uses 10 per cell, `bloch` keeps 1; NaN warning in
    `no_reflections_cmes_solve`.
  - **Regression:** finite results are unchanged by construction (converged `bloch` runs take the same path; the
    cap only acts on runs that hit it, which were NaN before).

- 2026-11-06, 2.6 on `audit-fixes`: `RKSolverConfig` passed to every kernel, shared step cap 10 per cell,
  `FBSolverConfig`/`fb_config` renamed `BlochSolverConfig`/`bloch_config` (no alias), and the cached kernels are
  dropped when any `@njit` module changes. Cap 1 vs 10 is bit-identical on the regression set; mypy 22 errors.

- 2026-11-06, WP2 API batch on `audit-fixes`: 2.7, 2.8, 2.10, 2.15-2.17, 2.19, 2.21-2.23, 2.25, 2.27,
  2.29-2.34, most of 2.35, and 2.36; tests in `tests/test_analysis_api.py`. Gains bit-identical on the
  regression set; mypy 22 -> 20.

## Where to pick up (end of session 2026-11-06)

`bloch-cmes` holds WP1, WP3.1-3.6 and the connector test; nothing is pushed. Next, roughly in order:
1. **WP3.7:** one source of defaults (`m_max` 2 in the config vs 5 in the solver signature, `anderson_depth` 10 vs
   5). **WP3.8:** add the `twpasolver-warmup` script entry to `pyproject`.
2. **WP2:** the bug fixes that still apply. 2.1 and 2.20 are moot; 2.6 and 2.14 are done; 2.5 (silent NaN) now
   concerns `simple` only.
3. **WP4:** split `analysis.py` (now about 1,900 lines; many WP4 items shrank with the model removal), a
   consistent result dict, and the docstrings (Is0 default, `fb_config`).
4. **WP5 to WP7:** models and modes, packaging and CI, docs; then reconcile with `origin/main` and push.
