# visiomode-analysis

Analysis library and CLI for behavioural session data recorded with [Visiomode](https://github.com/DuguidLab/visiomode), a visuomotor behaviour platform for rodents.

## Features

- **Session summaries** — quickly summarise session stats, including signal detection theory metrics.
- **HTML reports** — standalone, self-contained session reports with embedded Plotly figures.
- **GLM regressors** — event regressors (stimulus/response/reward windows) aligned to an external timestamp series, such as imaging frame timestamps or electrophysiology acquisition rates.
- **Subject-level and cohort-level analysis** — combine per-session trial summaries into a single per-subject summary CSV, as well as group-level analysis across subjects.

## Installation

Requires Python 3.11+.

```bash
pip install git+https://github.com/DuguidLab/visiomode_analysis.git
```

For local development, this project uses [uv](https://docs.astral.sh/uv/) to manage the virtual environment:

```bash
git clone https://github.com/DuguidLab/visiomode_analysis.git
cd visiomode_analysis
uv sync
```

This creates a `.venv` with the package and its dependencies installed in editable mode.

## Usage

### CLI

The package installs a `visiomode-analysis` command with four subcommands: `session`, `regressors`, `subject`, and `group`.

**Process a single session** — generates an HTML report and a trials CSV:

```bash
visiomode-analysis session path/to/sub-01_exp-myexperiment_ses-20260101_behaviour-gonogo.json -o output/
```

Skip the HTML report, or generate GLM regressors alongside it, with:

```bash
visiomode-analysis session path/to/session.json -o output/ --no-report
visiomode-analysis session path/to/session.json -o output/ --with-regressors --regressor-timestamps frame_times.csv
```

**Generate regressors** for an already-processed session, aligned to an external timestamp series:

```bash
visiomode-analysis regressors path/to/session.json -o output/ --regressor-timestamps frame_times.csv
```

**Collate a subject's sessions** — combines every `*trials.csv` file in a directory (as produced by `session`) into one subject-level summary CSV:

```bash
visiomode-analysis subject path/to/subject_dir/ -o output/
```

Run `visiomode-analysis --help` or `visiomode-analysis <command> --help` for full option details.

### Python API

The CLI is a thin wrapper around the `visiomode_analysis.session` module, which can also be used directly:

```python
from visiomode_analysis import session

trials = session.get_trials("path/to/session.json")
metadata = session.get_metadata("path/to/session.json")
summary = session.summary(trials)

session.generate_report(trials, metadata, output_dir="output/")
```

### Input files and naming convention

Session JSON filenames are expected to follow a BIDS-like pattern:

```
sub-<animal_id>_exp-<experiment>_ses-<YYYYMMDD>_behaviour-<protocol>.json
```

Metadata encoded in the filename takes precedence over the same fields in the JSON body. Output files (trials CSV, report HTML, regressors `.npz`, subject summary CSV) are named following the same convention, so downstream steps — e.g. `subject` globbing for `*trials.csv` — can find their inputs automatically.

## Project structure

```sh
src/visiomode_analysis/
├── __init__.py          # top-level Click CLI group, wires up subcommands
├── session/              # Session-level statistics
│   ├── __init__.py       # JSON → trials DataFrame, metadata, summaries, report/regressor generation
│   ├── metrics.py         # signal-detection-theory statistics 
│   ├── plots.py           # Plotly figure builders
│   └── regressor.py       # per-protocol GLM regressor construction
├── subject/               # collates per-session trials.csv files into a subject summary
│   └── __init__.py
├── group/                 # cohort-level aggregation across subjects (stub, unimplemented)
│   └── __init__.py
└── reports/                # Jinja2 templates for HTML session reports
    ├── __init__.py
    └── templates/
        ├── base.html
        └── session.html
```

## Development

```bash
# Run the full test suite with coverage (matches CI)
hatch test --cover

# Run tests directly with pytest (faster iteration)
.venv/bin/pytest

# Run a single test file / test
.venv/bin/pytest tests/test_metrics.py::test_d_prime_afc_correction -v

# Type checking
hatch run types:check
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
