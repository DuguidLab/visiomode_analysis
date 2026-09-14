# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Mesoscopy H5 input for regressors** — `--regressor-timestamps` (on both
  `regressors` and `session --with-regressors`) now accepts an `.h5` file written
  by `mesoscopy align`. Its `/timestamps_aligned` dataset is used directly as the
  regressor timestamps; a missing dataset or a `session_start_time` attribute that
  disagrees with the behaviour JSON raises. CSV/TXT input is unchanged. `h5py` is
  a new runtime dependency.
- **Regressor NPZ keys** — the `.npz` now also stores `session_start_time` (the
  JSON `timestamp`) and `behaviour_session` (the JSON filename stem) for every
  input type, so `mesoscopy process regression` can verify provenance.
- **`session-start-time` command** — prints the behaviour session's `timestamp`
  and nothing else.

### Changed

- **Tooling** — the project is now managed entirely with [uv](https://docs.astral.sh/uv/);
  Hatch is no longer used as a test/environment runner. Tests run with
  `uv run pytest --cov`, type checks with `uv run mypy`, and builds with
  `uv build`. Development dependencies moved into a `dev` dependency group in
  `pyproject.toml` and are now captured in `uv.lock`. Hatchling remains the
  build backend, so the built distributions are unchanged.
- **Makefile** — common tasks are wrapped in `make` targets (`test`, `test-cov`,
  `types`, `check`, `build`, `install`, `lock`, `clean`), which CI now calls
  directly so local and CI invocations cannot drift apart.
- **Versioning** — the version is now declared statically in `pyproject.toml`
  and bumped with `uv version --bump <major|minor|patch>`;
  `visiomode_analysis.__about__.__version__` reads it back from the installed
  distribution metadata.

### Fixed

- **Sessions with no responses no longer crash report generation.** The reaction
  time plots called `np.percentile` on an empty array (IndexError) when no trial
  had a response; they now render NaN markers instead. When the stimulus duration
  is unknown (legacy sessions without a `spec` report it as -1) the RT y-axis is
  left to auto-range rather than being fixed to `[0, -0.001]`.
- **Legacy `singletarget` sessions are treated as `targetonly`.** Older Visiomode
  versions named the target-only protocol `singletarget`; those sessions fell
  through to the Go/NoGo and 2AFC code paths, so their trials got no stimulus or
  SDT classification and the report tried to plot hit/false-alarm RTs that cannot
  exist. `session.is_targetonly()` / `session.TARGETONLY_PROTOCOLS` centralise the
  alias.
- **Legacy outcome labels are normalised before SDT inference.** `hit` /
  `false_alarm` / `miss` outcomes written by older Visiomode versions were only
  remapped to `correct` / `incorrect` / `no_response` after flattening, so the
  per-trial stimulus reconstruction and `sdt_type` inference never matched them
  and left every such trial unclassified. The remap now happens per trial inside
  `_flatten_trials()`.
- The type-check command no longer fails to start. The Hatch `types` environment
  inherited `path = ".venv"` from the default environment and tried to
  `pip install` into the uv-managed venv, which has no `pip`. Type stubs for
  pandas and scipy are now pinned as dev dependencies.

## [0.1.0] - 2026-08-11

First public release.

### Added

- **Session processing** (`visiomode_analysis.session`) — reads a raw Visiomode
  session JSON and flattens its nested trial list into a per-trial
  `pandas.DataFrame`, reconciling several historical Visiomode JSON schema
  versions along the way.
- **Session metadata** — `get_metadata()` derives animal ID, experiment, date,
  protocol and stimulus spec, preferring BIDS-like tokens encoded in the
  filename (`sub-<id>_exp-<name>_ses-<YYYYMMDD>_behaviour-<protocol>.json`) over
  the JSON body.
- **Session summaries** — `summary()` aggregates trials into signal-detection-theory
  metrics (hit and false alarm rates, d', criterion), trial counts and reaction
  time statistics, with `_wc` ("with corrections") variants computed alongside
  the corrections-excluded defaults.
- **Signal detection theory metrics** (`session.metrics`) — `d_prime`,
  `criterion` and `perseveration`, including a 2AFC correction for d'.
- **HTML session reports** — `generate_report()` renders a standalone,
  self-contained report from a Jinja2 template with embedded Plotly figures.
- **Plotly figure builders** (`session.plots`) — reusable figures for session
  reports, each able to return an embeddable HTML `<div>` via `as_html=True`.
- **GLM regressors** (`session.regressor`) — `generate_regressors()` builds
  stimulus, response and reward event regressors aligned to an external
  timestamp series such as imaging frame times or an electrophysiology
  acquisition clock. Go/No-Go is implemented; other protocols raise
  `NotImplementedError`.
- **Subject-level collation** (`visiomode_analysis.subject`) —
  `collate_sessions()` globs the `*trials.csv` files produced by the `session`
  command for one subject, aggregates them into a single per-subject summary CSV
  and assigns per-protocol `task_session` ranks.
- **Command line interface** — a `visiomode-analysis` entry point with `session`,
  `regressors`, `subject` and `group` subcommands. Output files follow the same
  BIDS-like naming convention as the inputs so downstream steps can find them
  automatically.
- **Group-level analysis** — `group` subcommand scaffolded; not yet implemented.
- Test suite covering trial flattening, metrics, plots, regressors, session
  summaries, subject collation and the CLI, run against Python 3.11–3.13 in CI.

[Unreleased]: https://github.com/DuguidLab/visiomode_analysis/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DuguidLab/visiomode_analysis/releases/tag/v0.1.0
