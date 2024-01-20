import argparse
from curses import meta
import datetime
import logging
import subprocess

import gage

limit = None
preview = False

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
)
log = logging.getLogger()


def main():
    args = _parse_args()
    models = _evaluated_models()
    runs = _latest_model_runs(models)
    for model in _iter_stale_models(models, runs, args):
        _import_results(model, args)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--limit",
        metavar="N",
        help="Limit batch to N models",
        type=int,
        default=limit,
    )
    p.add_argument(
        "--preview",
        help="show models without importing them",
        action="store_true",
        default=preview,
    )
    return p.parse_args()


def _evaluated_models():
    p = subprocess.Popen(
        "python import_results.py --list-models",
        stdout=subprocess.PIPE,
        shell=True,
        text=True,
    )
    return [
        _parse_model_line(line)  # \
        for line in p.stdout.readlines()  # type: ignore
    ]


def _parse_model_line(line):
    parts = line.rstrip().split(" ", 1)
    assert len(parts) == 2, parts
    return parts[0], _parse_date(parts[1], parts[0])


def _parse_date(s: str, context: str):
    try:
        return datetime.datetime.fromisoformat(s)
    except ValueError:
        assert False, (s, context)


def _latest_model_runs(models):
    runs = {}
    for run in _open_llm_import_runs():
        model = run["attributes"]["model"]
        if not model:
            continue
        try:
            latest = runs[model]
        except KeyError:
            runs[model] = run
        else:
            if run["started"] > latest["started"]:
                runs[model] = run
    return runs


def _open_llm_import_runs():
    return gage.runs(
        filter=lambda run: (
            run["operation"] == "import" and run["status"] == "completed"
        )
    )


def _iter_stale_models(models, runs, args):
    count = 0
    for model, eval_date in sorted(models):
        if args.limit is not None and args.limit == count:
            log.info("Reached limit of %s, stopping", args.limit)
            break
        run = runs.get(model)
        if not run or _run_started_utc(run) < eval_date:
            yield model
            count += 1


def _run_started_utc(run):
    started_utc = run["started"]
    return datetime.datetime.utcfromtimestamp(started_utc.timestamp())


def _import_results(model, args):
    if args.preview:
        log.info("Will import results for %s", model)
        return
    log.info("Importing results for %s", model)
    try:
        subprocess.check_output(
            f"gage run import model='{model}' -y",
            shell=True,
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        log.error(e.output)


if __name__ == "__main__":
    main()
