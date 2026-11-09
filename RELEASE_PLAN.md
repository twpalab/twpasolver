# twpasolver: clean-up and release plan

Working document for preparing twpasolver for publication (GitHub `twpalab/twpasolver`, Sphinx docs on
gh-pages, possibly PyPI). Written 2026-11-06 from a full review of the `edge-basis` branch: CME solvers,
`analysis.py`, and the rest of the package, tests, packaging, docs and branches. Any chat can pick up a work
package (WP) from here. Tick boxes and add notes as you go, and keep this file current.

**Goal:** a package that is correct, tested, documented and as small as possible to maintain. Prefer deleting
and merging code over adding options.

## How to work on this

- **Repo:** `/home/pietro/work/phd/twpas/twpasolver`. **Main branch of the clean-up: `bloch-cmes`**, created
  2026-11-06 from `edge-basis`. `edge-basis` holds the state before the clean-up and contains every other local
  branch (see WP7.8). Work happens in worktree branches (e.g. `audit-fixes`) that commit freely and never push;
  Pietro fast-forwards `bloch-cmes` with `git merge --ff-only <branch>`.
- **Python:** `~/.virtualenvs/twpaenv/bin/python`; from a worktree set `PYTHONPATH=src` (the editable install
  points to the main checkout). Poetry: `~/.virtualenvs/twpaenv/bin/poetry` (the one on PATH is broken).
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

## State at the first review (2026-11-06; see "Where to pick up" for the current state)

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
  2.5× speed). Keep it as a fast option, or remove it. **Decided 2026-11-06: keep it** (large speed-up). Reconfirmed 2026-11-07 for WP4.
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
- [x] **D7. Release target:** GitHub plus docs only, or also PyPI? Version number (suggested `0.1.0`)? Python
  versions to support (suggested ≥3.10, with no upper bound)?
  - **Pietro, 2026-11-06:** PyPI, `0.1.0`, Python ≥3.10. Set in `pyproject.toml` (`>=3.10,<4.0`: poetry will not
    resolve an open bound because the docs dependencies declare `<4.0`); lock refreshed.
- [x] **D8. Public API naming.** Is it acceptable to break `gain()` return keys and `FBSolverConfig` options
  (see WP4.2)? Downstream notebooks would need updating.
  - **Pietro, 2026-11-06:** any API break is allowed before 0.1.0.

## WP1: Baseline: green tests and a clean tree (do first)

- [x] **1.1 Commit the edge-basis fold-point fix** as its own commit, after Pietro confirms. Its tests:
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
- [x] **2.2 4WM Kerr factor in `phase_matching`** (`analysis.py:~1672`, `~1699-1705`). `pump_coeff=2` is applied
  only to the linear k_p, so |Δk_Kerr| = 3χk_p on a flat line; the CMEs give 2χk_p (weak tones k(1+2χ), pump
  k(1+χ), χ = ξI_p²/8). `notebooks/amplification/uniform_line.py` has a reference implementation.
  - Done 2026-11-06 (audit-fixes): Kerr bracket uses `pump_coeff * k_p`; a flat line gives -2 chi k_p (4WM) and -chi k_p (3WM), as `uniform_line.py`. Test `test_phase_matching_kerr_mismatch_on_flat_line`.
- [x] **2.3 k unwrap above resonator notches.** Done 2026-11-06 (audit-fixes): k is now the winding of the
  forward-wave amplitude a = V + Z0·I of the forward Bloch mode, carried back through every series/shunt element of
  the supercell (`cmes/bloch.py::bloch_wavenumber`), snapped to arg λ. Fixes k + π/d above notches, the sign of k in
  left-handed bands and k on grids starting above a stopband; equal to the old k to 1e-16 on the holotype devices.
  Old text: `_bloch_params_for_cell` (`analysis.py:~528`) unwraps 2·Im γ and
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
  - **Documented 2026-11-08** in `BlochSolverConfig.m_max` (build the supercell from one period, or scale m_max
    and kappa_cutoff by q); the automatic fix stays open for after the release.
  - The auto window also falls back to a hardcoded 0.10 when `edge_threshold=None` (`~1234`).
  - **Experiment 2026-11-06 (snapshot 95a9250, scripts and logs in the job tmp `mmax/`):** scaling *both* the
    harmonic limit and the π/D cap by q (102 cells: m=12, kc=0.6) matches the 34-cell run to 0.02–0.03 dB;
    scaling only m makes it worse (0.55 dB at 4 GHz). Auto vs converged reference (34 cells, m=17, kc=1,
    13× slower): 0.028 dB. q can be read from the cell list; the c_m spectrum also shows it (non-zero only at
    multiples of q, threshold ~1e-6).
  - **Deferred 2026-11-06 (keep `m_max='auto'` for now).** Branch `harmonic-filter` (bd46930, WIP): no
    `m_max`/cap/q; keep a term when its residual accumulates < `mismatch_cycles` cycles over the device and its
    process' Fourier component is >= `harmonic_tol` of its bound, both tested on canonically ordered participants
    (partners kept bit for bit). Accurate (holotype 100 cycles: 0.04 dB from the uncapped reference; 102-cell
    unfold = 34-cell without q) but ~4x slower than auto, and segmented copies stall (residual ~0.1) at 100
    cycles (fine at 30 and 300). The cost is **passes, not RK steps**: per-pass time is unchanged and
    `max_steps_per_cell=1` changes nothing, because the kernel only sees the residual `dk + G_m`; the extra
    high-m terms are forward-backward (Bragg-type) couplings that slow the fixed-point iteration. Next attempt:
    treat strong FB terms implicitly (as the edge basis does) or precondition Anderson with them. A residual
    threshold relative to k_proc without the pi/D cap was 25x slower (residuals up to ~0.6 rad/cell do cost RK
    steps).
  - **Rejected:** DFT weight rule w = |c_m|·min(L, 1/|Δk+G_m|) with one tolerance per process: 1.3–2 dB off at
    every tol (1e-1…3e-4), mostly unconverged at the edge. Exactly matched Kerr rows set W_ref (weight |c|·L) and
    drop pump-driven 4WM rows; 3WM chains are cut partially; a shared 3WM/4WM reference drops all 3WM (units).
    A viable weight needs field amplitudes (Ip for pump rows) and separate references. `kappa_cutoff`'s π/d cap
    is a validity bound, not a strength criterion, so it stays.
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
- [x] **2.9 `phase_matching` on segmented devices raises `IndexError`** (k is 2-D there). It also fills
  `np.empty` with garbage when `gamma` is None (`~1663`).
  - Done 2026-11-06 (audit-fixes): `delta` is the length-weighted mean of the per-segment mismatch (own k,
    gamma, and chi = Ip0² xi_s / 8 with the drive `twpa.Ip0`, as in `gain`); it used the outer `twpa.xi` before.
    Test `test_phase_matching_segmented_is_length_weighted_mean`.
- [x] **2.10 `k_star` fit guard:** `len(freqs_mask) > 2` (`~715`) should be `freqs_mask.sum() > 2`.
  - Done 2026-11-06 (audit-fixes). Test `test_k_star_guard_with_coarse_grid`.
- [x] **2.11 Stopband detection from argmin/argmax of `diff(S21_db)`** (`~665-670`) can return start > end; it
  gave `[19.13, 14.07]` on the device in `tests/test_bloch.py`. That reversed pair feeds the default ranges and
  the pump estimate. Use the supercell half-trace (`passband_mask`/`in_stopband` from the edge-basis fix).
  - **Audit 2026-11-06:** the result depends on where the grid ends. Same device: `f_arange` to 20 GHz gives
    `[19.06, 14.16]`, to 10 GHz `[6.79, 7.50]`, to 40 GHz `[39.55, 39.54]`. `_estimate_optimal_pump` then works
    on the single point `freqs[idx:-1]`.
  - Done 2026-11-06 (audit-fixes): lowest run of `in_stopband` on the lossless supercell (first segment for segmented devices); None plus a warning when the grid has none (`gain(pump=None)` and default `phase_matching` ranges then raise). The pump estimate searches the passband between that stopband and the next. Tests `test_stopband_does_not_depend_on_grid_end`, `test_no_stopband_needs_explicit_pump`.
  - **Pietro, 2026-11-06:** automatic pump estimate removed (it depends on the device and the mixing process).
    `data["stopbands"]` is an `(n, 2)` array of every stopband in the grid (`(0, 2)` without one); `gain` needs
    `pump`; `phase_matching` needs both ranges and returns `optimal_pump_freq` (argmin over `pump_freqs` of the
    mean `|delta|`; unfilled `delta` columns are NaN, no longer `np.empty` garbage). Tests
    `test_stopbands_do_not_depend_on_grid_end`, `test_no_stopband_gives_empty_stopbands`, `test_gain_needs_pump`,
    `test_phase_matching_optimal_pump`.
- [x] **2.12 Base data is never computed automatically.** `gain()` on a fresh `TWPAnalysis` raises
  `KeyError 'optimal_pump_freq'`, and edits to `twpa` leave stale data without warning. Call `update_base_data()`
  from the decorator, as it did before 6097a93. Exclude `Ip0` from the state check.
  - Rechecked 2026-11-06 (`888f01e`): still open; a fresh `gain()` now raises
    `ValueError: Mode array configuration 'basic_3wm' not found`.
  - Done 2026-11-06 (audit-fixes): `analysis_function` calls `update_base_data()` (a no-op when the state is
    unchanged). `Ip0` stays in the state check (only a recomputation cost). Test `test_base_data_computed_on_demand`.
- [x] **2.13 `gain(Ip0=...)` permanently sets `self.twpa.Ip0`** (`~1816`). Use a local value instead.
  - Done 2026-11-06 (audit-fixes): `Ip0` is passed down to the initial conditions. Test `test_gain_ip0_does_not_change_twpa`.
- [x] **2.14 Shared mutable default** `fb_config=FBSolverConfig()` (`~1738`): use `None`.
- [x] **2.15 `parameter_sweep` `log.warning` format** has an argument but no `%s` (`~414`).
  - Done 2026-11-06 (audit-fixes) with 2.32. Test `test_parameter_sweep_target_in_result_keys`.
- [x] **2.16 The decorator's overwrite message** checks `function_name` instead of `save_name` (`~368`); line
  `~375` is dead.
  - Done 2026-11-06 (audit-fixes). Test `test_overwrite_message_checks_save_name`.
- [x] **2.17 `bandwidth()` with no stored gain** raises a `TypeError`; with kwargs it overwrites `data["gain"]`.
  - Done 2026-11-06 (audit-fixes): no stored gain and no kwargs raises `ValueError`; all-NaN gain raises; `nanargmax` for partial NaN. The kwargs path still re-runs `gain()` (unchanged). Test `test_bandwidth_errors`.
- [x] **2.18 `thin`** is a point count in `gain` (zero points if `N_tot < thin`) but a stride in
  `phase_matching`. Make both a stride, or rename one of them.
  - Done 2026-11-06 (audit-fixes): `thin` removed from `phase_matching` (ranges used as given, default ranges unthinned); in `gain` it is the position spacing in unit cells, at least 2 points. Test `test_gain_position_grid_has_two_points_at_least`.
- [x] **2.19 `TWPAnalysis(freqs_arange=...)`** (a typo) is silently ignored and gives an empty grid. Set
  `extra="forbid"` on the pydantic models. The unused fixture at `tests/test_analysis.py:30` has exactly this typo.
  - Done 2026-11-06 (audit-fixes): `extra="forbid"` on `TWPAnalysis` and `Frequencies` only (circuit models unchanged); dead fixture removed. Test `test_twpanalysis_rejects_unknown_fields`.
- [x] **2.20 Likely, not run:** in segmented `general` forward–backward runs, the backward carrier reverses
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
- [x] **2.24 `modes_rwa._validate_modes` checks single characters** (`~553`), so multi-character mode names on
  the right-hand side fail ("Mode 2 not found" for `p2-s`).
  - Done 2026-11-06 (audit-fixes): `_validate_modes` deleted (5.7); unknown names raise `KeyError` in `RWAAnalyzer`. Test `test_mode_array_multichar_relation`.
- [x] **2.25 Cyclic relations** are accepted silently (empty expressions); `_compute_substitutions` (`~315`) can
  loop forever. Detect cycles and raise.
  - Done 2026-11-06 (audit-fixes): networkx cycle check in `RWAAnalyzer._compute_substitutions`. Test `test_cyclic_relations_raise`.
- [x] **2.26 `Mode.__eq__`/`__hash__`:** eq ignores the label but hash uses it (`~204/216`).
  - Done 2026-11-06 (audit-fixes): both deleted (5.7); `Mode` uses the dataclass `__eq__`.
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
- [x] **2.35 Small ones:**
  - `N=0` and `model * 0` crash on a bare `assert` in `matpow_2x2` (`N` is `NonNegativeInt`).
  - `Z0=0` passes the validator (`bonus_types.py:40`) and then fails the bare assert in `a2s`/`s2a`.
  - `Resistance.Z` truncates R on integer frequency arrays (`np.full_like`, `oneport.py:119`).
  - `TwoPortCell.interpolate` defaults to polar interpolation, which is wrong for real sign-changing entries
    (lossless TL: error 0.16 vs 0.0099 for `polar=False`).
  - `Frequencies(f_arange=(1, 2, 0))` divides by zero (step typed `NonNegativeFloat`).
  - A list-valued `port` fills only `Zout`; its ABCD stays the identity (`analysis.py:643-646`), read only.
  - CI venv cache key (`analysis.yml:26-27`) ignores `matrix.python-version`. **Done:** key has OS and Python.
  - `warmup` with `passes=1` logs a "did not converge" warning every time. **Done** with 3.8.
  - Done 2026-11-06 (audit-fixes) for `N=0` (identity), `Z0=0`, `Resistance.Z`, arange step > 0 (tests `test_zero_repetitions_is_identity`, `test_impedance_must_have_positive_real_part`, `test_resistance_on_integer_freqs`, `test_frequencies_validation`).
  - Done 2026-11-06 (audit-fixes): list-valued `port`/`port_out` removed. A one-port impedance fixes the reflection
    seen by the device but not the connector transmission, so drive, output coupling and prompt reflection were
    those of an identity connector (forward gain off by 20 log10(Ze/Z_l) dB for a real Z_l). Ports must be a
    `TwoPortModel`. Test `test_port_must_be_a_model`.
  - Done 2026-11-06 (audit-fixes): `TwoPortCell.interpolate` defaults to `polar=False`. Test `test_interpolate_default_handles_sign_changes`.
- [x] **2.36 Test for `TWPAnalysis`:** no test runs `phase_matching`, `bandwidth`, stopband detection or a save and
  reload of a real result; the `twpanalysis_instance` fixture asks for a `twpa_model` fixture that does not exist.
  One test on the 3-cell device (`gain` simple and bloch, `phase_matching`, `save_data`/`load_data`, stopband
  start < end) would have caught 2.11, 2.29-2.32.
  - Done 2026-11-06 (audit-fixes): `test_twpanalysis_end_to_end` (simple and bloch gain, `phase_matching`, `bandwidth`, save/load). Stopband start < end is left to 2.11.
- [x] **2.37 The segmented edge-basis `bloch` solve amplifies rounding by about 1e12** (found 2026-11-06 while
  checking the winding k, 77a7e00). On `_segmented_holotype((1, 1.02, 0.98, 1))` with the edge basis on (ext2,
  pump 16 GHz, passes 80, anderson_depth 15), multiplying `data["k"]` by (1 + 1e-15) moves `gain_db` by up to
  1.9e-3 dB (S11 more) at every signal, and it stays at 400 passes; the old (unwrap) code shows the same. Uniform
  devices agree to 3.6e-11 dB under the same perturbation. Suspect: the junction coefficients between segments when
  a mode is on the fixed edge basis (nearly singular basis change), or the per-segment edge-basis K. To do: find
  which input the gain is sensitive to (k vs Zb vs edge_K, per segment), check the conditioning of the junction
  matrices, and fix or at least document the accuracy limit.
  - **Narrowed 2026-11-06** (script `sens.py`, k x (1 + 1e-15), signals 2-12 GHz, ext2, pump 16 GHz, passes 80,
    anderson_depth 15): only segmented + edge basis is sensitive (max |dG21| 3.6e-3 dB, at every signal, largest at
    12 GHz). Segmented with `simulate_stopband_modes=False`: 4e-12 dB; segmented without stopband treatment
    (`edge_threshold=None`): 3e-11 dB; uniform in all three modes: <= 2e-11 dB. So the junctions and the Anderson
    iteration are fine; the culprit is the fixed edge basis across segment junctions (edge-mode junction step or
    per-segment `edge_K`). Start there.
  - **Done 2026-11-07** (ebbdf2d, cada7ee). Not an ill-conditioned junction: the iteration never converged. On
    that device `p4` (64 GHz) lies deep in the stopband of segment 2 only (|K01| L_seg = 218), is ~2000x below
    the pump, and jittered by a few % per pass, so the residual floor was ~1e-2, the default threshold: which pass
    stopped depended on rounding (k x (1 + 1e-15) changed the passes used at every point). Without `p4` the same
    device converges to 1e-10. Cause: `rk45_solve` linearly interpolated its adaptive steps onto `t_eval`, so the
    stored envelopes carried an O(h^2) error that depended discontinuously on the step placement (atol=1e-20
    gave the same floor drop, at 3-10x the cost). Fix: Dormand-Prince dense output. Floor 1e-2 -> 1e-4, all 9
    points converge (was 7/9) in 17-36 passes (was 21-80); at a tight threshold the rounding perturbation moves
    the gain by <= 5e-5 dB. The uniform holotype gain at 6 GHz moved by -0.023 dB (old value was interpolation
    error; matches an atol=1e-20 reference). EDGE_SUBSTEPS 4 -> 16 changes gains by <= 1.5e-4 dB: the edge grid is
    fine.

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
- [x] **3.3 One boundary-table helper, one forward–backward post-processing helper** (per-pass split, port gains,
  prompt reflections, `mode_info`), shared by every forward–backward driver. Folded into WP4.4 (2026-11-07).
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
- [x] **3.9 Dead options and near-copies in `cmes/` (audit 2026-11-06, about 260 lines):**
  - Never set by any caller: `edge_substeps` (`max_step` and `SOLVER_MAX_STEPS` removed with 2.6); rk45 `rtols`/`atols` arrays, the
    automatic `first_step` block, `y_size == 0` and `t_eval_size == 0` branches; `FBSolverConfig.m_max_fixed`.
  - Test-only: `bloch_fb_cmes_solve` (3 uses in `test_bloch.py`).
  - Forward and backward sweeps in `bloch_fb.py` (~90 lines each) differ only in index direction: one sweep with
    a direction flag.
  - `bloch.py`: `bloch_mode_tables` re-implements `supercell_matrix`/`_intracell_currents`; the edge walk is
    repeated in `edge_reference_frequency` and `stopband_edges`.
  - Done 2026-11-06 (audit-fixes): all of the above except `bloch_fb_cmes_solve` (undecided) and rk45
    `max_num_steps` (kept: `notebooks/ripples/kern.py` passes it); `edge_substeps` is the constant `EDGE_SUBSTEPS`.
- [x] **3.7 Unify defaults:** `m_max` (config 2 vs wrapper 5), `anderson_depth` (10 vs 5), and the `general`
  defaults (`passes` 3 vs 10, `kappa_cutoff` 0.1 vs 0.2, ...). Use `FBSolverConfig` as the single source.
  - Done 2026-11-06 (audit-fixes): `bloch_fb_segmented_cmes_solve` takes its defaults from `BlochSolverConfig()` (m_max 2, anderson_depth 10); `bloch_fb_cmes_solve` removed, tests call the segmented solver.
- [x] **3.8 `warmup.py`:** also compile the Bloch kernels (today it compiles only the `general` models). Add the
  `twpasolver-warmup` script entry to `pyproject` (the README advertises it, but it exists only in old install
  metadata). Fix its 13 mypy errors (`**dict[str, object]` passed into `TWPA`, lines 100-101).
  - **Done 2026-11-06 (audit-fixes):** `[tool.poetry.scripts] twpasolver-warmup` (checked in a built wheel); the
    warmup silences the logger during its one-pass calls, so it no longer prints "did not converge".

**Acceptance:** results identical to the pre-refactor reference (≤1e-9 dB for `bloch`); the full suite is green;
cold compile time measured and noted here; the line count noted here.

## WP4: `analysis.py` restructuring and API consistency

- Base data simplified 2026-11-06 (audit-fixes): one code path for uniform (one segment) and segmented
  devices; `_segment_supercell_abcds` and `_segment_bloch_params` merged into `update_base_data` and
  `_bloch_params_for_cell(cell, T=None)` (returns k, alpha, Zb, reflection_coeff only); `analysis.py` +55 -137
  (net -82; with the winding in `bloch.py` +83, src net +1). No data key removed (all are read somewhere).

**Design (agreed with Pietro 2026-11-07; any API break allowed, D8).** Today `TWPAnalysis` (~1,400 of 1,836
lines) mixes linear device data, a mutable mode-array registry, CME problem assembly, solver preparation (edge
basis, stopband guard, harmonic limits, relation coefficients: ~600 lines that belong in `cmes/`) and result
storage (`data` dict, auto-saving decorator), with two dualities everywhere: uniform vs segmented shapes and
`simple` vs `bloch` pipelines with different result keys. Target: pure functions over frozen data, a uniform
device is always the one-segment case, one problem builder and one result type for both models, a thin facade.

```
twpasolver/analysis/
  linear.py         LinearData (frozen): per-segment k, alpha, Zb±, segment bounds, S21/S11, stopbands, k_star
                    linear_data(twpa, freqs_hz) -> LinearData       # pure, cached on (twpa dump, grid)
  problem.py        CMEProblem: mode frequencies (n_modes, n_f), per-segment gammas/Zb, relations,
                    coefficients (n_rel, n_seg)                     # one builder for both models
  ports.py          Ports(input, output=None, Z_ext=50) -> boundary tables
  gain.py           solve(problem, inputs, model, ports, config) -> GainResult
  phase_matching.py from LinearData + modes (segment-weighted, as in 2.9)
  results.py        GainResult / PhaseMatchingResult dataclasses: same fields for every model, save/load, plot
  core.py           TWPAnalysis facade (~200 lines)
cmes/prep.py       edge basis, stopband guard, harmonic limits, relation coefficients (moved out of analysis)
```

```python
an = TWPAnalysis(twpa, freqs=(0.05, 20, 1e-3), unit="GHz", ports=Ports(conn, Z_ext=50))
an.linear.stopbands, an.linear.k                      # lazy, recomputed only when twpa/grid change
modes = ModeSet.three_wave(pump_harmonics=3, conversions=2)   # or "basic_3wm"
r = an.gain(signals, pump=8.0, modes=modes, model="bloch", Is0=1e-7, config=BlochSolverConfig())
r.gain_db, r.s11_db, r.I_fwd, r.I_bwd, r.converged, r.diagnostics
r.save("run.h5")
pm = an.phase_matching(signals, pumps, modes=modes, process="PA")
```

Decisions (Pietro, 2026-11-07):
- `simple` stays the textbook CMEs (k-based coupling); it is **not** rebuilt on the Bloch tables, so
  `cmes/base.py` stays. WP8.4 is withdrawn; WP8.5 (`phase_matching` from shared tables) is unaffected.
- The stopband guard (`simulate_stopband_modes=False`, D4) and `StubBaseCell(line=False)` stay.
- Dependencies: only `sympy` is removed (5.3); `networkx`, `skrf` and `matplotlib` stay core; no extras.
- Flatten the facade's base classes (4.3c).
- Skipped: slimming `BlochSolverConfig` (all options, `save_all_passes` included, stay) and making `skrf` an
  optional extra.

Items (each step keeps golden gains bit-identical on the holotype, the segmented holotype and the 3-cell device):
- [x] **4.1 Move solver preparation to `cmes/bloch_prep.py`:** `_edge_basis_entries`, `_harmonic_limits`,
  `_stopband_guard_weights`, `prepare_relations_coefficients`, `_fb_stopbands`, `_apply_edge_basis_segmented`,
  `_prepare_bloch_segmented`, `_log_edge_basis_skipped`. Mechanical, ~600 lines out of `analysis.py`.
  - Done 2026-11-07 (audit-fixes): module named `cmes/prep.py` (relation coefficients also feed `simple`);
    methods became functions taking `segments`, `seg_bounds`, `unit_multiplier`. `analysis.py` 1,836 → 1,275,
    `prep.py` 594. Golden reference (scratch `golden.py`: holotype edge/guard/simple, segmented holotype,
    3-cell simple/bloch, uniform and two-segment phase matching; 33 arrays) bit-identical; 140 tests.
- [x] **4.2 Uniform is the one-segment case everywhere:** drop the dual shapes in `_prepare_cme_inputs`, the
  two branches of `_apply_phase_carrier`, `_drop_segment_axis` and `_is_segmented` branching; the segment axis
  is internal and never in results.
  - Done 2026-11-07 (audit-fixes): `data["segments"]` always present (one entry for a uniform device);
    `_prepare_cme_inputs`, `_apply_phase_carrier`, `_bloch_segments` and `phase_matching` have one path; the
    `bloch` driver no longer adds the axis; `simple` always runs the segmented kernel (uniform kernel and
    `no_reflections_cmes` deleted, warmup compiles one `simple` kernel). Golden: `bloch` and phase matching
    bit-identical, uniform `simple` within 7e-11 dB (rounding of the cumulative phase); uniform `simple` is
    ~1/3 slower (0.18 → 0.24 s for 400 signals on the holotype). Public shapes unchanged on purpose:
    `data["k"]` stays 1-D for uniform devices until 4.3b, and `_drop_segment_axis` stays as the one squeeze
    of the uniform `bloch` diagnostics until 4.5b fixes the result layout. `analysis.py` 1,275 → 1,225,
    `cmes/base.py` 381 → 278.
  - **Found, open:** the segmented `simple` kernel interpolates the phase table linearly between output-grid
    points (`thin`), while the phase of the piecewise-linear κ is piecewise quadratic, so segmented `simple`
    results depend slightly on `thin`. Exact fix: evaluate the quadratic phase in the RHS, or put the
    segment midpoints into the phase grid. Not measured yet.
- [x] **4.3b `linear.py` + lazy `an.linear`:** replaces `update_base_data`, the `data` dict and the
  `analysis_function` decorator. `parameter_sweep` becomes a loop that stacks results.
- [x] **4.3c Flatten the facade's base classes:** `TWPAnalysis` stops inheriting from `Analyzer` (an abstract
  base with one subclass) and `Frequencies` (two ways to give the grid, `f_list` or `f_arange`). The facade
  takes `freqs` (array or arange tuple) and `unit`; `Analyzer` goes. `Frequencies` is used only as this base
  class but is exported from `twpasolver/__init__`: drop it, or keep it as a standalone grid helper the
  facade calls (decide when doing 4.3c).
  - Done 2026-11-07 (audit-fixes), with Pietro's choices (return only; per-segment shapes; drop
    `Frequencies`): frozen `LinearData` (`freqs`, `abcd`, `S21`, `S11`, `S21_db`, `stopbands`, `k`, `alpha`,
    `Zb`, `reflection_coeff` as (n_seg, n_freq), `k_star`, `seg_bounds`/`seg_midpoints`/`seg_N_tot`/
    `seg_epsilon`/`seg_xi`) from the pure `linear_data(twpa, f_hz, unit_multiplier)`; `an.linear` recomputes
    it when `twpa`, `freqs` or `unit` change and refreshes the mode arrays. Removed: `data`, `data_file`,
    `update_base_data`, `analysis_function`, `Analyzer`, `save_data`/`load_data`, `parameter_sweep`,
    `Frequencies` (module, export, tests, fixtures). `TWPAnalysis(twpa, freqs, unit, ports...)`: `freqs` is an
    arange tuple (validated: three values, positive step) or a list/array, stored as the explicit list so
    JSON round trips keep the grid. `bandwidth(gain_result)`, `plot_gain(result)`,
    `plot_mode_currents(result)`, `plot_phase_matching(result)` take the result; mode-array factories take
    `an.linear`. Golden unchanged from 4.2. Found on the way: `warmup()` crashed (its `gain` call had no
    `pump` since 2.11; the untyped decorator hid it from mypy); fixed, test `test_warmup_runs`. New tests
    `test_linear_follows_grid_and_unit`, `test_twpanalysis_json_round_trip_keeps_grid`. 133 tests, mypy 9.
  - **For 7.2:** docs notebooks and `implementing_models.rst` still use `f_arange`, `update_base_data`,
    `an.data[...]`, `parameter_sweep`, `save_data` and argument-less `plot_*`.
- [x] **4.4 `problem.py`: one builder for both models**, computing relations and coefficients once per call;
  absorbs 3.3 (shared port tables and post-processing). Ignored parameters go: `idler_mode` is derived from the
  relations; `Z0`/`Ze` duplicates removed.
  - **Ports, decided with Pietro 2026-11-07 (separate port fields):** the outgoing currents were split
    between the kernel (`_store_pass` overwrote the last profile point with the transmitted part, a leftover
    of the derivative-based boundary update), `_compute_bloch_gain` (prompt reflection from the connector
    ABCD) and a local `_gains`, and the backward profile was stored mirrored (`I_bwd[..., k]` at `L - x[k]`).
    Done 2026-11-07 (audit-fixes): the kernel returns plain envelopes; `bloch_fb_segmented_cmes_solve`
    applies the carriers, un-mirrors the backward profile and returns `I_fwd`, `I_bwd` (physical currents on
    `x`, true internal values at the ends) and the external currents `a_in`, `a_out`, `b_in`, `b_out`. One
    helper `_port_gains` gives `gain_db` (S21), `gain_db_12`, `gain_db_11`, `gain_db_22` for every model and
    every saved pass (per-pass keys now match the top-level ones); `simple` reports `a_in = y0`,
    `b_out = I(L)`, `a_out = 0`, `b_in = NaN`. Results gain a `ports` field. Golden: gains within 1e-14 dB,
    profile interiors bit-identical, last points now internal currents (reference re-recorded). This also
    fixes `plot_mode_currents(backward=True)`, which drew the mirrored profile.
- [x] **4.5b Result dataclasses** (`GainResult`, `PhaseMatchingResult`): every model has the same fields
  (`I_fwd`, `I_bwd` zeros for `simple`, `gain_db`, `gain_db_11/12/22` NaN for `simple`, `converged`,
  `diagnostics`, the optional per-pass history of `save_all_passes`), one naming scheme, NaN-safe dB;
  `save`/`load` through `file_utils`, `plot` through `plotting`.
- [x] **4.6 Units:** user unit at the facade, Hz everywhere inside; express the Hz spans (`1e9/4e9/8e9`) in the
  user unit or document them.
- [x] **4.7 Residual metric:** document on the result that the forward–backward residual is relative and
  dominated by nearly empty backward envelopes (floor 1e-3 to 1e-1 is common; observables stable); better,
  weight it by mode power. One convergence test, the same as the reported metric.
- [x] **4.8 Split into the package** above, `__init__` re-exporting `TWPAnalysis`, `GainModel`, `Ports`,
  `ModeSet` and the result types.
  - **Done 2026-11-07 (audit-fixes), 4.4 + 4.5b + 4.6 + 4.7 + 4.8 together with 5.3:** `twpasolver/analysis/` is
    `linear.py` (`LinearData`, `linear_data`, `ParameterInterpolator`, `segments_of`), `problem.py`
    (`build_problem` -> `CMEProblem`, defined in `cmes/prep.py`: mode frequencies signed in the analysis unit and
    `|f|` in Hz, per-segment gammas/reflections/Zb, relations and coefficients computed once), `ports.py`
    (`Ports(input, output, Z_ext)`, vectorised boundary tables), `gain.py` (`GainModel`, `solve_simple`,
    `solve_bloch`, `port_gains`), `phase_matching.py`, `results.py` (`GainResult`, `PhaseMatchingResult`:
    same fields for both models, `save`/`load` to HDF5, `plot`, `GainResult.bandwidth`, `plot_currents`,
    `mode(label)`) and `core.py` (`TWPAnalysis(twpa, freqs, unit, ports)` with `linear`, `problem(modes,
    tones)`, `gain(..., modes, tones)`, `phase_matching(signal_freqs, pump_freqs, ...)`, `plot_response`).
    Removed: the mode-array registry (`get_mode_array`, `add_mode_array`, `mode_array_config`), `idler_mode` of
    `gain` (metadata only; kept in `phase_matching`, where it picks the idler), `port`/`port_out`/
    `port_Z_ext`, the `_prepare_cme_inputs` dict. `prep.py` functions take the `CMEProblem`, so the solvers get
    Hz from one place (4.6; the remaining Hz constants are internal edge-search windows). 4.7: the residual is
    already one metric for the test and the report (per-mode change over the largest forward plus backward
    envelope; the `anderson_depth=0` route that differed went in 2.28); documented on `GainResult.diagnostics`.
    Extra independent tones (a second pump) go through `tones=` instead of mutating a mode array; a missing
    tone raises instead of counting as 0 Hz. Golden (scratch `golden.py`, self-contained): all 42 arrays
    bit-identical (gains, currents, phase mismatch). Tests migrated (`MIGRATION` notes in the job scratch);
    174 tests, mypy 9.
- [x] **4.9 Plain NumPy 2×2 algebra:** `matmul_2x2`/`matpow_2x2` duplicate `a @ b` and
  `np.linalg.matrix_power` on stacks; keep the `ABCDArray`/`SMatrixArray` accessors (`.A`, `.S21`) as thin views.
  - Done 2026-11-07 (audit-fixes): `ABCDArray.__matmul__` (same-length check kept, now `ValueError`) and
    `__pow__` (`**0` is the identity) and `supercell_matrix` use `np.matmul` / `np.linalg.matrix_power`;
    `mathutils.matmul_2x2`/`matpow_2x2` and their tests removed. Overflow to inf in stopbands is silenced as the
    Numba code did. Linear data of the holotype grid: 0.22 s -> 0.24 s. **Not bit-identical:** the product order
    of `matrix_power` differs at the 1e-16 level, which moves results within the solver tolerance (default
    `convergence_threshold` 1e-2 relative): `simple` and phase matching by <= 4e-12 dB, `bloch` S21 by
    <= 2e-4 dB (segmented holotype with edge basis, the 2.37 sensitivity) and 4e-6 dB elsewhere, S11 by
    <= 4e-3 dB. Golden reference re-recorded. 170 tests, mypy 9.

Estimate: `analysis.py` 1,836 lines → ~1,200 in the package (facade ~200); `src/` ~8,200 → ~6,500 together
with 5.3. Order: 4.1, 4.2, 4.3b with 4.3c, 4.4, 4.5b with 5.3, 4.6/4.7, 4.8; 4.9 any time.

History of the earlier WP4 items:
- [x] **4.3 Docstrings:**
  - `Is0` is documented as 1e-6 but defaults to 1e-7 (private drivers 1e-6);
  - `fb_config` is described as "GENERAL model only" (also in the `FBSolverConfig` docstring);
  - the comment at `~1929` says `Zout = twpa.Z0_ref`, but the code uses `port_Z_ext`;
  - the `_edge_basis_entries` Returns section needs a tidy (the `outside` edit);
  - `_bloch_mode_weights` claims segmented shapes it never receives;
  - the `__init__.py` package docstring lists MINIMAL_3WM/GENERAL and omits BLOCH.
- [x] **4.5 Dead code:**
  - `self._n_distinct_segments` (`~743`);
  - the unreachable `threshold is None` branches (`~1300`, `~1402`);
  - `gap` is always True (`~1250`);
  - the `edge_basis_threshold` alias in `FBSolverConfig` (only a test uses it);
  - the unused `twpanalysis_instance` fixture;
  - `mathutils.compute_phase_matching` (unused, and it has the wrong 4WM Kerr factor);
  - `FrequencyLinspace` and `nb_complex3d` (unused types).

## WP5: models, modes and logging

- [x] **5.1 `AnyModel` blocks user models** (`models/twoportarrays.py:167-168`). The union is built from
  `all_subclasses` at import, so a user `TwoPortModel` subclass defined later fails validation. Yet
  `docs/source/implementing_models.rst:11` promises it works, and `notebooks/contra_qpm/cqpm.py` works around it
  with `tw.cells[:] = ...`.
  - **Fix:** a registry with a rebuildable union (`model_rebuild` on subclass creation via
    `__init_subclass__`), or a custom validator dispatching on `name`.
  - Also remove the `# mypy: ignore-errors` headers (`twoportarrays.py:9`, `plotting.py:3`); the opaque union
    causes 8 mypy errors in analysis/bloch.
  - Done 2026-11-07 (audit-fixes): `TwoPortModel` keeps a `registry` filled in `__pydantic_init_subclass__` by
    every subclass that declares its own `name`; `ModelArray` validates each cell through
    `TwoPortModel.validate_cell` (dict by registered name, instance copied, nested list wrapped), checked
    against `cell_base` (`OnePortModel` for `OnePortArray`). User models defined after import work as
    instances, dicts and through JSON; an unnamed subclass (e.g. the contra_qpm cell with its own hook) keeps
    its type. `AnyModel`, `AnyOnePortModel` and `all_subclasses` removed; `cells` stays `list[Any]` because a
    `list[TwoPortModel]` field with `revalidate_instances="always"` would revalidate subclass instances as the
    base class and drop their fields. Both `# mypy: ignore-errors` headers removed (`computed_field` and the
    `name` override get the codes Pydantic documents; `plt.get_cmap`); mypy stays at 9. Tests
    `tests/test_models_registry.py`. Golden bit-identical; 176 tests.
- [x] **5.2 First-class nonlinear-element hook.** The Bloch code accepts `bloch_nl_elements()` on cells; document
  it, and consider an `nl_scale` field on `LCLfBaseCell`/`StubBaseCell`. That would remove the subclass in
  `contra_qpm` and allow nonlinearity-only gratings in JSON models.
  - Done 2026-11-07 (audit-fixes): `nl_scale` (default 1) on `LCLfBaseCell` and `StubBaseCell` scales their
    nonlinear elements in the bloch CMEs only (not the linear response, not `simple`); the
    `bloch_nl_elements()` hook and `nl_scale` are documented in `implementing_models.rst` (whose method name
    `get_single_abcd` was wrong: it is `single_abcd`). Test: `nl_scale=0.5` gives bit-identical bloch gains to
    a subclass whose hook returns half of `L`.
- [x] **5.3 `modes_rwa`:**
  - Replace the two parsers (`RWAAnalyzer._parse_expression` `~269`, `ModeArray._parse_expression_for_propagation`
    `~425`) and the two resolvers (sympy substitution, dict linear combination) with integer coefficient vectors.
    This drops the sympy dependency (current build: 18 modes, 1363 terms, 0.12 s).
  - **Verified exact 2026-11-07:** the parser admits only sums/differences of names, so every mode is an integer
    vector over the independent tones (any number: multiple pumps included) and sympy `==` on these linear forms
    is integer-vector equality. Prototype (`find_terms`, ~50 lines, scratch `rwa_int.py`) gives term lists
    identical to `RWAAnalyzer.find_rwa_terms` (order and coefficients) on 338 relation sets (all factory configs,
    300 random sets with 1-3 independent modes, a dual-pump 4WM set): ~50,000 terms. Keep that comparison as the
    regression test; reject a mode defined by two relations; cycles still raise.
  - Becomes `ModeSet` (WP4): frequencies are the integer matrix applied to the independent tones, no mutation.
    `plot_mode_relations` and `print_modes` stay as they are; `networkx` stays a core dependency.
  - Remove `n_sidebands` (`~1066`) and the unused `edge_colors`/`edge_styles`. Done 2026-11-06 (audit-fixes),
    together with the `general_no_reflections` alias of `GainModel`.
  - `get_dependent_modes` has no docstring and no caller.
  - **Done 2026-11-07 (audit-fixes):** `modes_rwa.py` is `ModeSet` (frozen: labels, relations, `independent`,
    integer `matrix`; `basic`, `extended_3wm`, `frequencies(tones)`, `rwa_terms(power)` cached, `dependents`
    (replaces `get_dependent_modes`), `expression`, `__str__`, `plot_relations` without `plt.show()`).
    `RWAAnalyzer`, `Mode`, `ModeArray`, `ModeArrayFactory` and sympy are gone (dependency dropped, lock
    refreshed). Regression: `tests/data/rwa_terms.json` freezes the sympy term lists of 39 mode sets (8,750 terms,
    all factory configs and a dual-pump set); `ModeSet` matches all of them exactly (`tests/test_modes.py`).
  - No `plt.show()` in library code (`~914`); deduplicate `plot_mode_relations`/`print_symbolic_expressions`.
- [x] **5.4 Logging:** `logger.py:39-40` sets INFO and adds a handler at import, so messages print twice under
  pytest. A library should add only a `NullHandler`; keep a helper (`twpasolver.logger.enable(level)`) for users.
  - Done 2026-11-06 (audit-fixes): `log.propagate = False` (own handler and INFO kept, so notebooks still print once). Test `test_log_does_not_reach_root_handler`.
- [x] **5.7 Dead code outside `cmes/` (audit 2026-11-06, grepped incl. docs and notebooks):**
  `mathutils.compute_phase_matching` (~58 lines, also crashes when `pump_f/2 < freqs[0]`), `TwoPortCell.from_z`,
  `bonus_types.FrequencyLinspace` and `nb_complex3d`, `ModeArray.get_dependent_modes`,
  `RWAAnalyzer.update_relations`, `print_rwa_terms`, `edge_colors`/`edge_styles`, `print_symbolic_expressions`
  (repeats `plot_mode_relations`), `_n_distinct_segments`, and `Mode.__eq__`/`__hash__` (fixes 2.26 by deletion).
  `_validate_modes` can go too (fixes 2.24): `RWAAnalyzer` already raises on unknown names. The `mode_info` dict
  is built twice in `analysis.py`. `twpa_cells.py` repeats the qp-loss half-inductance block in `StubBaseCell`
  and `LCLfBaseCell`.
  - Done 2026-11-06 (audit-fixes) except `get_dependent_modes` (kept: used in
    `notebooks/amplification/twpasolver_thesis/05_quantum_time.ipynb`). Test `test_stub_cell_line_abcd`.
- [x] **5.5 `matrices_arrays.py`:** the 12 duplicated property pairs on ABCD/S/Z arrays become a small descriptor.
  `TwoPortModel.__mul__` should use `model_copy(update=...)`.
  - Done 2026-11-06 (audit-fixes): descriptor `_Element`, same validation. Left: `__mul__`.
  - Done 2026-11-07 (audit-fixes): `__mul__` is a deep `model_copy` with a validated `N` assignment.
- [x] **5.6 `quantum/`** (per D3). Dead functions: `gaussian.expm_batch`, `magnus4_omega`,
  `transfer_to_scattering`. The Python `star`/`cascade` duplicate the numba versions in `kernels.py`;
  `linearize.py:146` uses `col` possibly before assignment.

## WP6: Packaging, CI, tooling

- [x] **6.1 `pyproject.toml`:**
  - real description; version per D7; license as SPDX `Apache-2.0`; author email; URLs; classifiers;
  - drop the `<3.13` upper bound; **done 2026-11-06** with version `0.1.0` (D7), lock refreshed;
  - `networkx = "<=3.*"` resolves to 3.0: use `>=3`;
  - sympy is listed in both main and analysis (or dropped, per 5.3);
  - declare `typing_extensions` (imported);
  - drop the `CyRK` mypy override;
  - add `[tool.poetry.scripts] twpasolver-warmup = ...`;
  - `[tool.pylint]` `good-names` for physics symbols (removes most of the 803 invalid-name warnings);
  - a `[tool.pycln]` section, or fix the pre-commit argument;
  - refresh `poetry.lock` (poetry is broken locally: fix it or do this in CI).
  - Done 2026-11-07 (audit-fixes): PEP 621 `[project]` (Poetry 2 deprecates the `[tool.poetry]` metadata):
    description, SPDX `Apache-2.0`, author `twpalab` and maintainer Pietro Campana with e-mail (poetry-core 2.1
    writes only the first author into the metadata), documentation/repository URLs, keywords, classifiers for
    3.10-3.13, `networkx>=3.0`, `typing-extensions` declared, `[project.scripts]` warmup entry, sympy gone, CyRK
    mypy override dropped, `[tool.pycln] all`, `[tool.pylint]` name regexes for physics symbols and line length
    100, `[tool.black]` py310, `[tool.isort]` black profile. Lock refreshed; `poetry build` gives a wheel with
    the `analysis/`, `cmes/`, `models/` packages and complete metadata.
- [x] **6.2 CI (`.github/workflows/analysis.yml`, `deploy_docs.yml`):**
  - bump checkout/setup-python/gh-pages from v3 (deprecated Node);
  - include the Python version in the venv cache key (3.10–3.12 currently share one `.venv`);
  - run on pull requests;
  - let pylint fail on errors only;
  - mypy must pass;
  - docs: drop the pinned 3.10.4 and the apt sphinx; deploy on tags or releases only.
  - Done 2026-11-07 (audit-fixes): `analysis.yml` on push and pull requests, actions v4/v5, matrix 3.10-3.13,
    pylint `--errors-only` (clean: Numba `prange` and one Pydantic field read get inline disables), mypy must
    pass, pytest with coverage. `deploy_docs.yml` builds on every push/PR with Python 3.12 and pandoc and
    deploys to gh-pages only for `v*` tags. New `publish.yml`: build and upload to PyPI on `v*` tags with
    trusted publishing (configure the trusted publisher on PyPI first).
- [x] **6.3 pre-commit:** isort is pinned to `9.0.0b1` (beta); the nbqa isort 5.6.4 and pyupgrade 2.7.3 pins are
  very old; add nbstripout (WP1.7).
  - Done 2026-11-07 (audit-fixes): isort 9.0.2 (was the 9.0.0b1 beta); nbQA hooks no longer pin isort 5.6.4 /
    pyupgrade 2.7.3; the isort profile lives in pyproject so both isort hooks use it; nbstripout already in.
    `pre-commit run --all-files` passes.
- [x] **6.4 Coverage:** 40 % overall. Plotting 11 %, warmup 0 %, quantum 0 %; numba code reads low because
  coverage cannot trace jitted functions. Add a CI job with `NUMBA_DISABLE_JIT=1` for a real number, if
  feasible.
  - 2026-11-07: 62 % (analysis package 94-100 %, plotting 15 %, Numba kernels read low). The
    `NUMBA_DISABLE_JIT=1` coverage job was not added (a full suite without JIT would take far longer).
- [x] **6.5 mypy to zero:** 13 errors fixed by 3.8, 8 by 5.1, 2 by 2.27; numba types used as annotations
  (`cmes/base.py:29`) need `TYPE_CHECKING` aliases; plus about 6 ndarray shape assignments.

## WP7: Documentation and release

  - Done 2026-11-07 (audit-fixes): mypy reports no issues in 35 files (numba types out of annotations, `EPS` a
    float, `cast` in `validate_cell`, array-typed reductions and annotations).
- [x] **7.1 Toctree** (`docs/source/tutorials.rst`/`index.rst`) has only notebooks 01-04. 05/06 and `quantum` were moved
  out on 2026-11-06 (thesis material). New docs notebooks for segmented devices and the edge basis can be written
  later from them, without thesis dependencies.
  - Done 2026-11-07: tutorials 01-06 in `tutorials.rst`; `physics.rst` added to the index.
- [x] **7.2 Notebooks** must use the post-refactor API, and must not depend on thesis files (`palette.py`,
  `prl_aps.mplstyle`, `edge_basis_sweep.py` → keep only what the docs need).
  - Done 2026-11-07: 03 and 04 ported to the new API (04's Floquet section moved to 05); new 05_other_twpas
    (Malnou's 4WM KI-TWPA, PR Applied 25, 054016 (2026), Appendix I values: rpm notch at 9.71 GHz from a user-defined
    cell with the Bloch hook, modulated capacitors opening the 26-30 GHz stopband; 17-19 dB at 9.1 GHz / 22 uA,
    third harmonic suppressed by 23 dB; Floquet TWPA from segments on the KI holotype: 13 dB fewer photons in
    conversion modes at equal gain) and 06_segmented_devices (identical segments vs uniform, runtime vs
    n_seg, random L/Lf variations, Lf taper). All execute without errors: 03 ~2 s, 04 ~23 s, 05 ~13 s,
    06 ~1 min 50 s. 01/02 used no analysis API.
- [x] **7.3 API reference:** `run_apidoc` in `conf.py` writes into the repo, so either ignore its output (1.5) or
  generate it at build time only. It currently lacks bloch*, rk45, warmup and quantum.
  - 2026-11-07: the `analysis/` package gives ~40 "duplicate object description" warnings (classes on
    both the package page, through the re-exports, and the module pages; dataclass fields both in
    `Attributes:` and as members). Document the package page only, or turn off member docs for dataclasses.
  - Done 2026-11-07: apidoc output stays git-ignored and regenerated at build; `ignore-module-all` and
    `napoleon_use_ivar` remove the duplicate-object warnings; the remaining RST errors (bloch_mode_tables,
    edge_basis_tables, logm, bloch_fb title and literal block, `|...|` in field descriptions) fixed: the build
    has no docstring warnings left. Notebook rendering needs pandoc (not installed on this machine; CI installs it).
- [x] **7.4 Execution policy:** `nbsphinx_execute` is unset (auto), so stripped notebooks run at every build,
  which is slow given numba compile times. Either commit outputs for heavy notebooks and set `never` for them, or
  cache them. Pandoc is needed for the build and is not installed locally.
  - Done 2026-11-07: `nbsphinx_execute = "always"`, timeout 900 s.
- [x] **7.5 README:** what twpasolver is, the models and when to use which, install, warmup, a minimal example,
  citation, license.
  - Done 2026-11-07: README with features, models, install, warmup, a tested example, citing, license.
- [x] **7.6 Physics notes for users:** document that the forward-only `general_no_reflections` (future `simple`)
  ignores `port`, `port_out` and `port_Z_ext`: its gain is output over input inside the line, with no connector
  losses or end reflections. Either say so or warn when connectors are set. the edge-basis derivation (`edge_basis_physics.tex`, cited by
  `cmes/bloch.py`) now lives in `notebooks/amplification/twpasolver_thesis/claude_outputs/`; adapt it for the docs.
  - the edge basis, and when to set `edge_threshold`;
  - `m_max` per supercell (2.4);
  - the convergence/residual meaning (4.7);
  - the nonlinear-element hook (5.2).
  - Done 2026-11-07: `docs/source/physics.rst` (models and when to use them, ports, edge basis and
    `edge_threshold`, mixing terms and `m_max` per supercell, convergence, mode sets, custom cells); `simple`
    now warns when connectors are set. The full edge-basis derivation (thesis tex) is not adapted.
- [x] **7.7 Changelog of breaking changes** (return keys, removed models and options).
  - Breaking so far (audit-fixes): `phase_matching(thin=)` removed; `bloch_fb_cmes_solve` removed;
    `bloch_fb_segmented_cmes_solve` defaults are now those of `BlochSolverConfig` (m_max 2, anderson_depth 10);
    the `twpasolver` logger no longer propagates to the root logger; `interpolate` defaults to `polar=False`;
    `data["stopband_freqs"]` replaced by `data["stopbands"]` (all bands, `(n, 2)`); `data["optimal_pump_freq"]`
    removed, `phase_matching(...)["optimal_pump_freq"]` instead; `gain` `pump` required; `phase_matching`
    `signal_arange`/`pump_arange` required (keyword-only).
  - `data["k"]` by winding (2.3): changes above resonator notches (was off by π per supercell), in
    left-handed bands (now k < 0) and on grids starting above a stopband (was off by 2π per supercell);
    `_bloch_params_for_cell` no longer returns `gamma_bloch`, `vg`, `N_tot` and drops `Z0_override`/`abcd_single`.
  - Done 2026-11-07: `CHANGELOG.md` for 0.1.0 (new, breaking changes, fixes).
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

## WP9: Assessment 2026-11-07 (after WP4-WP7)

State: `src/` 7.4k lines (14.8k at the first review), 178 tests, coverage ~62 %, mypy clean, pylint errors-only
clean, tutorials 01-06 execute (~2.5 min in total). Import 0.6 s; cold compile on first use: `simple` ~12 s,
`bloch` ~62 s (hence `twpasolver-warmup`). Largest remaining units are the Numba solver functions
(`_bloch_fb_solve_segmented` 378 lines, `bloch_fb_segmented_cmes_solve` 273, `rk45_solve` 253).

- [x] **9.1 Silent extrapolation of mode parameters.** Modes above the top of `freqs` (e.g. the fourth pump
  harmonic, 63 GHz, on a 40 GHz grid) got a cubic-spline extrapolated k (0.372 instead of 0.462 rad/cell on the
  holotype): gains off by up to 0.6 dB and NaN at 3 of 8 signals, with no warning. Done 2026-11-07:
  `build_problem` raises a `ValueError` naming the modes and the grid. Test `test_modes_outside_grid_raise`;
  tests and notebooks 03/04/06 unaffected.
- [x] **9.2 Plot smoke test** (`test_plots_render`, Agg backend) for every plot entry point; plotting was at 15 %.
- [x] **9.3 Numba cache in CI**, keyed on the sources and the lock file (saves the ~1-2 min compile per job).
- [x] **9.4 Linear response with connectors.** `an.linear.S21/S11` are those of the bare device; `ports` only enter
  the `bloch` gain. Cascading the connector ABCDs (or an `an.s_parameters()` with ports) would let users compare
  the linear response with measurements of a packaged device. Small.
  - Done 2026-11-07: `TWPAnalysis.response(connectors=True)` returns the `TwoPortCell` of the device with the
    connectors cascaded (output one reversed), referred to `Z_ext`. Test: with no pump, bloch S21/S11 equal it to
    1e-6 dB with a tight `convergence_threshold` (4e-5 dB at the default 1e-2: the end reflections come out of the
    forward-backward iteration and inherit its tolerance).
- [x] **9.5 `GainResult.history` pads converged points** with their last pass (`bloch_fb.py` kernel), so the pass
  axis is always `passes` long. Store the passes used per point in `diagnostics` (or mask the padding). Small.
  - Done 2026-11-07: the kernel returns the passes each point ran (`diagnostics["passes_used"]`); history entries
    after it are NaN (the last slot still holds the result). Test `test_history_stops_at_the_passes_used`.
- [x] **9.6 `thin`** is documented as a spacing in unit cells but sets `N_tot / thin` points; either document it as
  such or use `arange(0, N_tot + thin, thin)` (changes the x grid). Trivial.
  - Done 2026-11-08: documented (`gain` docstring, notebook 03); the grid is unchanged.
- [x] **9.7 Pump power in dBm.** `gain(Ip0=...)` takes amperes; `mathutils.dBm_to_I` exists but users must find it.
  A `pump_dbm=`/`signal_dbm=` convenience or a documented recipe in notebook 03. Small.
  - Done 2026-11-08: documented recipe (`Is0`/`Ip0` docstrings, notebook 03), no new arguments.
- [x] **9.8 The 4.2 finding** (segmented `simple` slightly depends on `thin`): measure once.
  - Done 2026-11-07: `simple` moves <= 1e-4 dB with `thin` (9e-4 dB at thin=400, 4 varied segments): negligible.
    `bloch` depends more (vs thin=2: uniform 2e-4 dB at thin=100, 0.011 dB at 400; 4 varied segments 2.2e-3 and
    0.035 dB), because the stored envelopes are interpolated (4-point Lagrange) on the internal grid, which is the
    output grid plus segment edges. Identical segments vs the uniform device: 2e-3 dB at the defaults (200
    segments), from the convergence tolerance (each run stops at a different point within 1e-2) plus the
    interpolation; tight threshold + thin=20 gives ~0. Tried refining the internal grid per segment (independent
    of `thin`, steps from kappa_cutoff): it removes the thin dependence but breaks convergence where the
    forward-backward iteration is fragile (Malnou P3 design in notebook 05: 17/17 -> 0/17; Floquet 14/15 -> 7/15;
    40-segment random chip seed 3). The old code fails the same way with a dense `thin` (Malnou: thin=9 0/17,
    thin=3 all NaN), so dense internal grids destabilise the iteration; reverted, see 9.9.
- [x] **9.9 Forward-backward iteration diverges on dense internal grids** with edge-basis modes (stopband
  modes: p3 of the Malnou design, p+s of random segmented chips). Reproducers: notebook 05 design with
  `thin=9`; `seed3.py`-like 40-segment holotype (sigma 2 %, rng seed 3) with `thin=20`. Probably the same
  ill-conditioning as 2.37 (edge-basis propagation over many short steps). Fixing it would allow a
  thin-independent internal grid (patch in the 9.8 note). Also consider an observable-based stopping test:
  the uniform holotype at 5 GHz stalls at residual 1e-4 while the gain is stable to 2e-5 dB.
  - **Done 2026-11-07** (cada7ee). Not 2.37: the edge-mode BVP took the rows linear in the edge mode itself
    (XPM from pump/signal/idler, pump-induced coupling of its two directions) as explicit sources lagged by one
    pass. Once the grid resolves the evanescent profile, that loop has gain > 1 (Malnou `p3` grew 1.3x per pass
    at thin=9; zeroing those 4 rows made it converge in 4 passes). Fix: those rows form an exact 2x2 Jacobian
    J(x), and the BVP steps with exp((K + J_mid) h). Malnou thin=9/3 converge 17/17 with the thin=100 gains;
    seed-3 thin=20 converges 20/20 within 0.02 dB of thin=100; the 2.37 device needs 10-27 passes instead of
    17-36. Cost: the Jacobian is 4 extra RHS evaluations of the few self-linear rows per edge-grid point; end to
    end +11 % on the 2.37 device, -11 % on seed 3. Notebook 06 now converges at every point (the uniform 5 GHz
    point too). At thin=100 the output-grid interpolation still costs up to ~0.02 dB near the upper band edge
    (12-12.5 GHz on the 2.37 device), in either direction: that is 9.10.
- [ ] **9.10 Thin-independent internal grid.** Dense grids now converge (9.9), so the per-segment refinement
  tried in 9.8 (steps from `kappa_cutoff`, independent of `thin`) can be re-applied. Measure its cost first:
  the pass time grows with the internal grid.
- [ ] **9.11 (after the release) Cheaper edge-mode Jacobian.** Partly done 2026-11-07: the Jacobian columns now reuse
  the field vector of the source evaluation (per-call overhead was most of the cost); uniform p2s edge point
  0.68 -> 0.49 s (0.43 s before the Jacobian), 40-signal uniform sweep 0.8 -> 0.6 s. What remains: The implicit self-terms (9.9) cost ~25 % on the
  uniform holotype (40 signals: 1.2 -> 1.5 s at fixed settings; ~50 % per pass, one pass saved). J depends almost
  only on the pump, which settles in the first passes: refresh it every few passes and add the defect correction
  ``(J_new - J_frozen) c_prev`` explicitly, so the fixed point is unchanged. Stability first was the choice for 0.1.0.

## WP8: General devices and user models (after the first release)

Goal: any lumped/distributed ladder (left-handed, CRLH, shunt-JJ, SNAIL, mixed JJ/kinetic) and user models that
are a plain `compose(...)` with no extra hooks. Today the nonlinearity is one device-wide `epsilon/xi/chi` on
`TWPA`, its location is an `isinstance` chain in `series_nl_elements` (series elements only), `simple` scales the
coupling by γ_n and `phase_matching` uses `2χk_p` (both right-handed continuum assumptions, wrong sign and
scaling on left-handed bands), and built-in cells are opaque ABCD functions.

- [ ] **8.1 Cells as primitive chains.** One `flatten(cell)` → ordered series Z / shunt Y / TL-section primitives
  (`Inductance`, `Capacitance`, `Stub`, `LosslessTL`, `twoport_parallel`). It gives the ABCD, exact winding steps
  (no `_t_split`; TL sections subdivided, closes the distributed-line winding item), nonlinear-element positions
  and both V and I profiles. Rewrite `LCLfBaseCell`/`StubBaseCell` as compositions. Black-box `single_abcd`
  cells stay valid for linear parts only (clear error if marked nonlinear). Useful alone.
- [ ] **8.2 Per-element nonlinearity.** Optional nonlinear spec on `Inductance` (L0, I*, I*2, Idc or ε/ξ);
  `TWPA` keeps the pump, its current fields become defaults. Removes the `isinstance` chain and the
  `bloch_nl_elements` hook (supersedes 5.2).
- [ ] **8.3 One coupling builder, series and shunt.** Series nonlinear L = voltage source projected with the I
  profile (as now); shunt nonlinear L = current source projected with the V profile; distributed kinetic
  nonlinearity by quadrature inside TL sections. Kernels unchanged (they take coefficient tables). Derivation of
  the shunt term to be checked on a small lumped left-handed ladder.
- [ ] **8.4 `simple` from the same tables, forward modes only.** Drop the γ-based coupling in `cmes/base.py`.
  **Withdrawn 2026-11-07:** `simple` stays the textbook CMEs.
  First measure the gap to the current form on the holotype (should be discreteness-level).
- [ ] **8.5 `phase_matching` from the same tables** (pump-only SPM/XPM), removing `TWPA.chi`.
- [ ] **8.6 Reference devices as tests** (right-handed ladder vs analytic, left-handed ladder, CRLH, KI stub, SNAIL 3WM;
  ideally a harmonic-balance cross-check on short devices) and a "writing a model" docs page (left-handed example).

Breaking: `TWPA` nonlinearity fields and the built-in cell classes.

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
| 8 | WP8 (next release) | large | medium | first release |

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
- 2026-11-06, dead code and near-copies on `audit-fixes` (3.9, 5.5, 5.7, 2.24, 2.26): one `_sweep` for both
  directions, `bloch.py` traversal and edge walk shared, rk45 trimmed, descriptor arrays; 15 files, +229 -717 (net -488 lines). Gains and
  currents bit-identical on the regression set, warm bloch timing unchanged (0.8 s / 4.7 s), mypy 20 -> 19.
- 2026-11-06, group B on `audit-fixes`: 2.2, 2.11, 2.13, 2.18, 3.7, 5.4, polar `interpolate` (2.35), and
  `bloch_fb_cmes_solve` removed. Gains bit-identical on the regression set (explicit pump); default pump,
  stopband and phase-matching results change as intended. mypy 19 -> 13.
- 2026-11-06, automatic pump estimate removed on `audit-fixes` (Pietro): `data["stopbands"]` lists all bands,
  `gain(pump)` and the `phase_matching` ranges are required, `optimal_pump_freq` moved to the `phase_matching`
  result. Gains and k_star bit-identical on the regression set.
- 2026-11-06, winding k (2.3) and base-data simplification on `audit-fixes`: gains of the uniform regression
  cases within 1.3e-11 dB, k within 1.1e-16 rad/cell on three devices; segmented edge case moves by up to 1e-3 dB,
  which a 1e-15 relative perturbation of k already causes on the old code (ill-conditioned there). mypy 13 -> 10.

- 2026-11-06, faster base data on `audit-fixes`: the winding loop is one parallel numba kernel
  (`bloch._wind_backward`, one atan2 per element step) and `supercell_matrix` uses `matmul_2x2`/`matpow_2x2`.
  `update_base_data` (warm, 39,900 points): holotype 172 -> 42 ms, 3-cell 110 -> 22 ms, segmented 361 -> 130 ms.
  Base data bit-identical; `supercell_matrix` moves by ~2e-13 (product order), so uniform `bloch` gains move
  <= 3.5e-11 dB, `simple` is identical, and the segmented edge-basis case moves 3.3e-3 dB (item 2.37).

## Where to pick up (updated 2026-11-08, `audit-fixes`)

WP0 to WP4 and 5.3 are done (2.37 done 2026-11-07; 2.4 documented, automatic fix deferred); the API is the one of the WP4 design (`TWPAnalysis(twpa,
freqs, unit, ports)`, `ModeSet`, `GainResult`/`PhaseMatchingResult`). Nothing is pushed. Next, roughly in order:
1. **WP5 is done** (2026-11-07). The open finding of 4.2 (segmented `simple` depends slightly on `thin`) is
   worth one measurement.
2. **WP6 is done** (2026-11-07): metadata, CI, publish workflow, pre-commit, mypy clean.
3. **WP7 is done except 7.8/7.9** (Pietro): merge the branches, configure the PyPI trusted publisher for
   `publish.yml`, check the author/maintainer metadata in `pyproject.toml`, build the docs once with pandoc,
   then tag `v0.1.0` (the tag publishes to PyPI and deploys the docs).
4. **Regression reference: done 2026-11-08.** `tests/test_regression.py` compares S21/S11 of the holotype,
   a 4-segment holotype and two small devices (`simple` and `bloch`) plus two phase-matching maps with
   `tests/data/regression_gains.npz` (1e-3 dB). Re-record with `python tests/test_regression.py` after an
   intended change.
   `tests/test_harmonic_balance.py` checks `bloch` against a converged JosephsonCircuits.jl HB gain of the holotype
   (`tests/data/hb_holotype_gain.csv`, 111 signals 2-13 GHz, pump 15.8 GHz; HB truncation 6/4 vs 12/8 differs
   by < 3e-6 dB). The test uses `kappa_cutoff=0.2`: median 0.02 dB, <= 0.10 dB in band, flanks within ~30 MHz,
   one known miss (12.9 GHz, a ripple near the stopband shifted ~50 MHz, 0.6 dB). With the default 0.1 the signal/idler
   pair 5.4/10.4 GHz (edge mode p+i active) is 2.0/2.4 dB off; `convergence_threshold=1e-3` and 40 passes change
   nothing. Cost of 0.2: 2.2 s vs 1.4 s for 111 signals (9 modes). Open question: default kappa_cutoff back to 0.2?
   Regenerate with `notebooks/jc_comparison` (`twpa_to_jc.py --fp 15.8 --signal 2 13 0.1`, `run_jc_gain.jl`).
