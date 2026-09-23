#  Copyright (c) 2026 Constantinos Eleftheriou <Constantinos.Eleftheriou@ed.ac.uk>.
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this
#   software and associated documentation files (the "Software"), to deal in the
#   Software without restriction, including without limitation the rights to use, copy,
#   modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#   and to permit persons to whom the Software is furnished to do so, subject to the
#   following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies
#  or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#  HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
#  IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
#  IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import os
import click
import glob
import warnings
import pandas as pd

import visiomode_analysis.session as session

# Suffixes appended after `.csv` (e.g. `..._trials.csv.ignore`) that mark a session as excluded.
EXCLUDE_SUFFIXES = (".ignore", ".ignored", ".exclude", ".excluded", ".skip")

METADATA_FIELDS = ("animal_id", "session_date", "protocol", "environment", "experiment")


@click.command("subject")
@click.argument(
    "directory",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(dir_okay=True),
    default=".",
    help="Output directory for report and trials files.",
)
@click.option(
    "--no-ignore",
    is_flag=True,
    help=f"Drop sessions marked as excluded ({', '.join(EXCLUDE_SUFFIXES)} after .csv) altogether, "
    "instead of listing them without metrics.",
)
def subject_cmd(directory, output_dir, no_ignore):
    with warnings.catch_warnings(action="ignore"):
        out_dir = preprocess_subject(directory=directory, output_dir=output_dir, ignore=not no_ignore)
    click.echo(f"Files saved under {out_dir}")


def preprocess_subject(directory, output_dir: str = ".", ignore: bool = True) -> str:
    collate_sessions(directory=directory, output_dir=output_dir, ignore=ignore)
    return output_dir


def is_excluded(path: str) -> bool:
    """Whether a trials file is marked as excluded by a suffix after `.csv` (e.g. `..._trials.csv.ignore`)."""
    base, suffix = os.path.splitext(path)
    return base.endswith("trials.csv") and suffix.lower() in EXCLUDE_SUFFIXES


def collate_sessions(
    directory, output_dir: str | None, ignore: bool = True
) -> pd.DataFrame | tuple[pd.DataFrame, str]:
    """Collate preprocessed trials.csv files for one subject into a per-session summary.

    Sessions marked as excluded (see `EXCLUDE_SUFFIXES`) are listed with their metadata but no metrics,
    and still count towards `session_id` and `task_session`. With `ignore=False` they are dropped altogether.
    """
    session_files = [
        f for f in glob.glob(f"{directory}{os.sep}*trials.csv*") if f.endswith("trials.csv") or is_excluded(f)
    ]

    if not session_files:
        raise FileNotFoundError(f"No trials.csv files found in {directory}, did you forget to preprocess?")

    if not ignore:
        session_files = [f for f in session_files if not is_excluded(f)]
        if not session_files:
            raise FileNotFoundError(f"No trials.csv files found in {directory} that are not marked as excluded.")

    subject_sessions = []
    for session_file in session_files:
        if is_excluded(session_file):
            first_row = pd.read_csv(session_file, nrows=1).iloc[0]
            subject_sessions.append({**{k: first_row.get(k) for k in METADATA_FIELDS}, "excluded": True})
        else:
            summary = session.summary(session_file)
            subject_sessions.append({**{k: summary[k] for k in METADATA_FIELDS}, "excluded": False, **summary})

    subject_df = pd.DataFrame(subject_sessions)

    subject_df = subject_df.sort_values("session_date").reset_index(drop=True)
    subject_df["session_id"] = subject_df.index + 1
    subject_df["task_session"] = subject_df.groupby(["animal_id", "protocol"], sort=False)["session_id"].rank(
        ascending=True
    )

    metadata = {
        "animal_id": subject_df["animal_id"][0],
        "experiment": subject_df["experiment"][0],
    }

    if output_dir:
        out_path = f"{output_dir}{os.sep}sub-{metadata.get('animal_id')}_exp-{metadata.get('experiment')}_behaviour-summary.csv"
        subject_df.to_csv(out_path)
    return subject_df
