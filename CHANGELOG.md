# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
