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

- [ ] **D1. `general` forward–backward model** (`cmes_fb.py` plus its `analysis.py` branch, about 1,850 lines).
  - **Against keeping it:** it is the least accurate model, has no convergence flags, no NaN detection and no
    tests, uses clamped perturbative boundary updates, and has a Manley–Rowe-breaking cutoff (WP2).
  - **What only it provides:** segmentations whose segments are not whole supercells.
  - **Options:** remove it, or deprecate it for one release, then remove it.
- [ ] **D2. `bloch_no_reflections`.**
  - **Against keeping it:** it covers uniform devices only, depends on the heuristic guard stack, and exists only
    as a fast Bloch-coupling estimate.
  - **Options:** remove it, or merge it into one forward-only kernel shared with `general_no_reflections`
    (Bloch coefficient tables as input).
- [x] **D3. `quantum/`.** It has about 1.8k lines, 0 % coverage and no API docs, and only the untracked
  `quantum.ipynb` uses it. Options: leave it out of the first release, or ship it as
  `twpasolver.experimental.quantum` with a warning and a smoke test. If it ships, it still needs
  `edge_guard_mode_weights` (see WP3.4).
  **Decided 2026-11-06:** keep `quantum/` in the repo for a future update. It is not part of this clean-up. When
  refactoring the Bloch code, keep `quantum/background.py` importable (adapter or minimal edits), but don't invest in
  it; how it is packaged for the first release is still open (WP6).
- [ ] **D4. `simulate_stopband_modes=False`** (the guard path superseded by the edge basis; it only buys about
  2.5× speed). Keep it as a fast option, or remove it.
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
- [ ] **1.2 Frozen fixtures.** Copy the holotype JSON into `tests/data/` and point `HOLOTYPE` there
  (`tests/test_bloch.py:510`; about 15 tests use it, and today they `skipif` silently if the docs file moves).
  - The current test file needs **both** JSON versions: the 6 older edge-basis tests pass only with the HEAD
    version, the new fold-point and contra tests only with the edited one.
  - Either keep two fixtures (`holotype_v3_head.json`, `holotype_v3_newsim.json`), or re-derive the hard-coded
    edge frequencies of one set for a single fixture. Re-deriving is preferred.
- [ ] **1.3 Stale tests.**
  - Delete `test_segmented_not_supported`: segmented devices are supported now.
  - Rewrite or delete `test_edge_guard_removes_spurious_conversion_gain`: the guard was replaced by the edge basis.
- [ ] **1.4 `test_mismatch_filter_keeps_partner_terms_together[general]`** is a real bug; fix it in WP2.1, or
  mark it xfail until then.
- [ ] **1.5 `.gitignore`:** add `Claude outputs/`, `docs/source/api-reference/` (regenerated by `conf.py`),
  `*.pkl`, `*.npz`. Move `Claude outputs/` out of the repo after asking Pietro.
- [x] **1.6 Move thesis and scratch files out of `docs/source/notebooks/`** (per D6). Done 2026-11-06, see Log.
  - Delete `04_generalized_cmes copy.ipynb`; delete or move `bloch_test.ipynb`, `test_ports.ipynb` and
    `models_benchmark.ipynb`.
  - Delete `src/**/.ipynb_checkpoints` (ignored by git, but they clutter the source tree).
- [ ] **1.7 Notebook outputs:** strip outputs from tracked notebooks 04 (1.07 MB) and 06 (1.4 MB), and add an
  nbstripout pre-commit hook (or decide on stored outputs, see WP7.4).

**Acceptance:** pytest is all green, and `git status` shows no untracked junk.

## WP2: Bug fixes, no restructuring (low risk)

Each fix gets a test.

### Physics and solvers

- [ ] **2.1 Asymmetric cutoff in `cmes_fb._filter_wm_terms`** (`cmes_fb.py:~111`). It normalises the mismatch
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
- [ ] **2.5 Silent NaN.** The rk45 step cap (`rk45.py:~182`) produces NaN; the no-reflections and `general`
  models return NaN gain without a warning. Add a warning, or a `converged`/`finite` flag on every model.
- [ ] **2.6 Global solver constants are frozen into cached numba kernels.** Changing `SOLVER_MAX_STEPS_PER_CELL`
  and similar has no effect until `__pycache__` is wiped. Remove `configure_solver`/`get_solver_config`
  (`solver_config.py:81-115`), which are never called and misleading, and pass tolerances as arguments.

### API bugs (confirmed by running them)

- [ ] **2.7 2-D `initial_amplitudes_*` crash `general` and `bloch`.** `y0_fwd[signal_idx]` at
  `analysis.py:~2030` and `~2458` picks a row instead of a column; use `y0_*_broadcast[:, signal_idx]`.
- [ ] **2.8 HDF5 save fails for every forward–backward result:** `gains_all_passes` is a list of dicts. Store it
  as arrays.
- [ ] **2.9 `phase_matching` on segmented devices raises `IndexError`** (k is 2-D there). It also fills
  `np.empty` with garbage when `gamma` is None (`~1663`).
- [ ] **2.10 `k_star` fit guard:** `len(freqs_mask) > 2` (`~715`) should be `freqs_mask.sum() > 2`.
- [ ] **2.11 Stopband detection from argmin/argmax of `diff(S21_db)`** (`~665-670`) can return start > end; it
  gave `[19.13, 14.07]` on the device in `tests/test_bloch.py`. That reversed pair feeds the default ranges and
  the pump estimate. Use the supercell half-trace (`passband_mask`/`in_stopband` from the edge-basis fix).
- [ ] **2.12 Base data is never computed automatically.** `gain()` on a fresh `TWPAnalysis` raises
  `KeyError 'optimal_pump_freq'`, and edits to `twpa` leave stale data without warning. Call `update_base_data()`
  from the decorator, as it did before 6097a93. Exclude `Ip0` from the state check.
- [ ] **2.13 `gain(Ip0=...)` permanently sets `self.twpa.Ip0`** (`~1816`). Use a local value instead.
- [ ] **2.14 Shared mutable default** `fb_config=FBSolverConfig()` (`~1738`): use `None`.
- [ ] **2.15 `parameter_sweep` `log.warning` format** has an argument but no `%s` (`~414`).
- [ ] **2.16 The decorator's overwrite message** checks `function_name` instead of `save_name` (`~368`); line
  `~375` is dead.
- [ ] **2.17 `bandwidth()` with no stored gain** raises a `TypeError`; with kwargs it overwrites `data["gain"]`.
- [ ] **2.18 `thin`** is a point count in `gain` (zero points if `N_tot < thin`) but a stride in
  `phase_matching`. Make both a stride, or rename one of them.
- [ ] **2.19 `TWPAnalysis(freqs_arange=...)`** (a typo) is silently ignored and gives an empty grid. Set
  `extra="forbid"` on the pydantic models. The unused fixture at `tests/test_analysis.py:30` has exactly this typo.
- [ ] **2.20 Likely, not run:** in segmented `general` forward–backward runs, the backward carrier reverses
  `gammas` but not the segment midpoints (`~1968-1988`). This affects only the phase of `I_tuples_bwd`. Moot if
  D1 removes the model.

### Models and modes

- [ ] **2.21 `TWPA` slicing drops parameters** (`modelarray.py:53`): `TWPA(Istar=1e-3, N=7)[0:2]` returns
  default `Istar` and `N=1`.
- [ ] **2.22 `validate_nested_cells` mutates the caller's list** (`modelarray.py:39`).
- [ ] **2.23 Small fixes:** `matrices_arrays.py:99` (`and` should be `or`), `bonus_types.py:41` (missing `f`
  prefix), and `Frequencies.f` silently prefers `f_arange` when `f_list` is also set (raise instead).
- [ ] **2.24 `modes_rwa._validate_modes` checks single characters** (`~553`), so multi-character mode names on
  the right-hand side fail ("Mode 2 not found" for `p2-s`).
- [ ] **2.25 Cyclic relations** are accepted silently (empty expressions); `_compute_substitutions` (`~315`) can
  loop forever. Detect cycles and raise.
- [ ] **2.26 `Mode.__eq__`/`__hash__`:** eq ignores the label but hash uses it (`~204/216`).
- [ ] **2.27 `twpa_cells.py:94,103`:** complex `C` is passed where the annotation says float (mypy).

**Acceptance:** each item has a test; the full suite is green; mypy errors go down.

## WP3: Solver consolidation (largest saving, about 1,100–1,500 lines)

Order: 3.1 → 3.2 → 3.3, then 3.4/3.5 per the decisions.

- [ ] **3.1 Drop the uniform Bloch forward–backward kernel; route uniform devices through the segmented solver
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
- [ ] **3.2 `analysis.py` uniform/segmented merge.**
  - `_prepare_bloch_coefficients` (`~1071`) → `_prepare_bloch_segmented` (`~1505`) with n_seg = 1.
  - `_apply_edge_basis` (`~1273`) → `_apply_edge_basis_segmented` (`~1376`).
  - Stop computing the unused `coeffs_3wm`/`coeffs_4wm` on the forward–backward path (rebuilt at `~1356-1360`
    and never used by the kernel).
- [ ] **3.3 One boundary-table helper, one forward–backward post-processing helper** (per-pass split, port gains,
  prompt reflections, `mode_info`), shared by every forward–backward driver.
- [ ] **3.4 Guard systems** (per D2/D3/D4).
  - **What exists:**
    - `_stopband_guard_weights` and `_process_weights_nd` serve `simulate_stopband_modes=False`;
    - `_bloch_mode_weights`, `_bloch_edge_guard`, `edge_proximity_mode_weights`, `cell_phase_mask` and the
      `BLOCH_EDGE_GUARD*` constants serve `bloch_no_reflections`;
    - `edge_guard_mode_weights` serves the quantum model;
    - `process_weights` (bloch.py:730) duplicates `_process_weights_nd` (`analysis.py:149`).
  - **Target:** at most one guard-weight mechanism, or none. Remove the `pw3`/`pw4` kernel plumbing if nothing
    needs it.
  - The 45° reactive-angle threshold is repeated at 4 places; make it one constant.
- [ ] **3.5 Forward-only kernels:** `bloch_no_reflections_cmes` (bloch.py:775) is `no_reflections_cmes`
  (base.py:63) with the (α+jκ) prefactor folded into the coefficients. Keep one kernel with per-frequency
  coefficient tables (per D2).
- [ ] **3.6 `general` model** (per D1). If removed, delete `cmes_fb.py`, `_compute_forward_backward_gain`, the
  `general` branches and options (`update_rate`, `BOUNDARY_CORR_MAX`) and `_filter_wm_terms`; also
  `_convergence_step`/`_store_pass` once `bloch_fb` no longer imports them.
- [ ] **3.7 Unify defaults:** `m_max` (config 2 vs wrapper 5), `anderson_depth` (10 vs 5), and the `general`
  defaults (`passes` 3 vs 10, `kappa_cutoff` 0.1 vs 0.2, ...). Use `FBSolverConfig` as the single source.
- [ ] **3.8 `warmup.py`:** also compile the Bloch kernels (today it compiles only the `general` models). Add the
  `twpasolver-warmup` script entry to `pyproject` (the README advertises it, but it exists only in old install
  metadata). Fix its 13 mypy errors (`**dict[str, object]` passed into `TWPA`, lines 100-101).

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
- [ ] **7.6 Physics notes for users:**
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
