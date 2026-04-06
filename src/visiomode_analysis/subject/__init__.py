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
import pandas as pd

import visiomode_analysis.session as session


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
def subject_cmd(**kwargs):
    out_dir = preprocess_subject(**kwargs)
    click.echo(f"Files saved under {out_dir}")


def preprocess_subject(directory, output_dir: str = ".") -> str:
    collate_sessions(directory=directory, output_dir=output_dir)
    return output_dir


def collate_sessions(directory, output_dir: str | None) -> pd.DataFrame | tuple[pd.DataFrame, str]:
    session_files = glob.glob(f"{directory}{os.sep}*trials.csv")

    if not session_files:
        raise FileNotFoundError(f"No trials.csv files found in {directory}, did you forget to preprocess?")

    subject_sessions = []
    for session_file in session_files:
        subject_sessions.append(session.summary(session_file))

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
