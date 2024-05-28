import csv
import logging
import os
import sys

log = logging.getLogger()

from _util import sync_results_repo


def main():
    print("Getting latest eval results", file=sys.stderr)
    repo_dir = sync_results_repo()
    with _output_file() as f:
        writer = csv.writer(f)
        writer.writerow(["model", "eval"])
        for model, last_eval in _iter_model_evals(repo_dir):
            writer.writerow([model, last_eval])


def _output_file():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        print(f"Writing index to {filename}", file=sys.stderr)
        return open(filename, "w")
    return sys.stdout


def _iter_model_evals(dir):
    for root, dirs, files in os.walk(dir):
        model = _model_name_for_dir(root, dir)
        for eval in _model_evals_for_dir(files, root):
            yield model, eval


def _model_evals_for_dir(files, root):
    return sorted(
        [
            name[8:-5]
            for name in files
            if name.startswith("results_") and name.endswith(".json")
        ]
    )


def _model_name_for_dir(dir, repo_dir_root):
    return os.path.relpath(dir, repo_dir_root)


if __name__ == "__main__":
    main()
