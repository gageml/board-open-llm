import argparse
import glob
import json
import logging
import os
import re
import subprocess
import sys
import tempfile

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
)
log = logging.getLogger()


RESULT_FILENAME_P = re.compile(
    r"results_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+).json$"
)

OPEN_LLM_RESULTS_REPO = "https://huggingface.co/datasets/open-llm-leaderboard/results"
default_repo_dir = os.path.join(tempfile.gettempdir(), "open-llm-leaderboard")

model = None


def main():
    args = _parse_args()
    _sync_repo(args)
    if args.list_models:
        _show_models_and_exit(args)
    summary = _model_summary(args)
    _write_summary(summary, args)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-m",
        "--model",
        help="Model to import",
        default=model,
    )
    p.add_argument(
        "--list-models",
        help="Show available models and exit",
        action="store_true",
    )
    p.add_argument(
        "--repo-dir",
        help="Local directory for Open LLM Git repository",
        default=default_repo_dir,
    )
    p.add_argument(
        "--summary",
        help="Summary file to write",
        default="summary.json",
    )
    args = p.parse_args()
    if not args.model and not args.list_models:
        raise SystemExit("Specify either --model or --list-models")
    return args


def _sync_repo(args):
    if os.path.exists(os.path.join(args.repo_dir, ".git")):
        log.info("Syncing results")
        # Use fetch + merge -n to keep log noise down
        _run("git fetch", cwd=args.repo_dir)
        _run("git merge -n", cwd=args.repo_dir)
    else:
        log.info("Getting results")
        if not os.path.exists(args.repo_dir):
            os.makedirs(args.repo_dir)
        _run(
            f"git clone {OPEN_LLM_RESULTS_REPO} '{args.repo_dir}'",
            env={"GIT_LFS_SKIP_SMUDGE": "1"},
        )


def _run(cmd, cwd=None, env=None):
    env = {**(env or {}), **os.environ}
    p = subprocess.Popen(
        cmd,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    while True:
        buf = p.stdout.read(1024)  # type: ignore
        if not buf:
            break
        sys.stderr.buffer.write(buf)
    p.wait()
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def _show_models_and_exit(args):
    for model, result in sorted(_iter_models(args)):
        print(model, result)
    raise SystemExit(0)


def _iter_models(args):
    for root, dirs, files in os.walk(args.repo_dir):
        latest_eval = _latest_eval_for_dir(files, root)
        if latest_eval:
            yield _model_name_for_dir(root, args.repo_dir), latest_eval


def _latest_eval_for_dir(files, root):
    eval_results = sorted([name for name in files if name.endswith(".json")])
    if not eval_results:
        return None
    latest = eval_results[-1]
    m = RESULT_FILENAME_P.match(latest)
    if not m:
        if os.getenv("DEBUG"):
            log.warning(
                'Unexpected eval result file "%s" in "%s", skipping',
                latest,
                root,
            )
        return None
    return m.group(1)


def _model_name_for_dir(dir, repo_dir_root):
    return os.path.relpath(dir, repo_dir_root)


def _model_summary(args):
    assert args.model
    log.info("Importing results")
    summary = {}
    for results in _iter_results(args):
        results_summary = _results_summary(results, args.model)
        _apply_dict(results_summary, summary)
    return summary


def _iter_results(args):
    model_paths = args.model.split("/")
    paths = glob.glob(os.path.join(args.repo_dir, *model_paths, "*"))
    if not paths:
        log.error("No results for model '%s'", args.model)
        raise SystemExit(
            "Try 'python import_results.py --list-models' for available models."
        )
    for path in sorted(paths):
        with open(path) as f:
            yield json.load(f)


def _results_summary(results, model):
    config = _results_config(results)
    precision = _model_precision(config)
    return {
        "run": {
            "label": f"{model} {precision}",
        },
        "attributes": _results_attributes(model, config, precision),
        "metrics": _results_metrics(results),
    }


def _results_config(results):
    try:
        return results["config_general"]
    except KeyError:
        return results["config"]


def _model_precision(config):
    dtype = config["model_dtype"]
    map = {
        "torch.float16": "float16",
        "torch.bfloat16": "bfloat16",
        "None": "GPTQ",
    }
    return map.get(dtype, dtype)


def _results_attributes(model, config, precision):
    return {
        "model": {
            "value": model,
            "type": "hf:model",
        },
        "precision": precision,
        "params": _model_params_b(config),
    }


def _model_params_b(config):
    model_size = config.get("model_size")
    if not model_size:
        return None
    if model_size.endswith(" GB"):
        return float(model_size[:-3])
    elif model_size.endswith(" MB"):
        return float(model_size[:-3]) / 1000
    assert False, model_size


def _results_metrics(results):
    task_results = results["results"]
    metrics = {}
    tasks = [
        _apply_arc,
        _apply_hellaswag,
        _apply_mmlu,
        _apply_truthfulqa,
        _apply_winogrande,
        _apply_gsm8k,
    ]
    for task_f in tasks:
        task_f(task_results, metrics)
    return metrics


def _apply_arc(results, metrics):
    val = _results_val(results, "harness|arc:challenge", "acc_norm")
    if val:
        metrics["arc"] = val


def _results_val(results, startswith, val_key):
    vals = [
        val
        for val in [
            results[key].get(val_key) for key in results if key.startswith(startswith)
        ]
        if val is not None
    ]
    return 100 * sum(vals) / len(vals) if vals else None


def _apply_hellaswag(results, metrics):
    val = _results_val(results, "harness|hellaswag", "acc_norm")
    if val is not None:
        metrics["hellaswag"] = val


def _apply_mmlu(results, metrics):
    val = _results_val(results, "harness|hendrycksTest-", "acc")
    if val:
        metrics["mmlu"] = val


def _apply_truthfulqa(results, metrics):
    val = _results_val(results, "harness|truthfulqa:mc", "mc2")
    if val:
        metrics["truthfulqa"] = val


def _apply_winogrande(results, metrics):
    val = _results_val(results, "harness|winogrande", "acc")
    if val:
        metrics["winogrande"] = val


def _apply_gsm8k(results, metrics):
    val = _results_val(results, "harness|gsm8k", "acc")
    if val:
        metrics["gsm8k"] = val


def _apply_dict(src, dest):
    for name, src_val in src.items():
        try:
            dest_val = dest[name]
        except KeyError:
            dest[name] = src_val
        else:
            if isinstance(dest_val, dict) and isinstance(src_val, dict):
                _apply_dict(src_val, dest_val)
            else:
                dest[name] = src_val


def _write_summary(summary, args):
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
