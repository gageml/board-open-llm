import argparse
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile

import urllib.request
import urllib.error

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
)
log = logging.getLogger()


RESULT_FILENAME_P = re.compile(
    r"results_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+).json$"
)

OPEN_LLM_RESULTS_REPO = "https://huggingface.co/datasets/open-llm-leaderboard/results"
HF_MODELS_API_BASE_URL = "https://huggingface.co/api/models/"

DEFAULT_SUMMARY_FILE = "summary.json"
DEFAULT_DATA_PATH = None  # None -> data saved to a new tmp dir


default_repo_dir = os.path.join(tempfile.gettempdir(), "open-llm-leaderboard")

model = None


def main():
    args = _parse_args()
    _sync_repo(args)
    if args.list_models:
        _show_models_and_exit(args)
    assert args.model
    log.info("Importing results for %s", args.model)
    model_filename = _get_model_info(args)
    eval_filenames = _get_model_evals(args)
    summary = _model_summary(args.model, model_filename, eval_filenames)
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
        default=DEFAULT_SUMMARY_FILE,
    )
    p.add_argument(
        "--data-path",
        help="Location to save model data files (defaults to a tmp dir)",
        default=DEFAULT_DATA_PATH,
    )
    args = p.parse_args()
    if not args.model and not args.list_models:
        raise SystemExit("Specify either --model or --list-models")
    if not args.data_path:
        args.data_path = tempfile.mkdtemp(prefix="gage-open-llm-")
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
    return _eval_filename_date_to_isoformat(m.group(1))


def _eval_filename_date_to_isoformat(s):
    parts = s.split("T")
    assert len(parts) == 2, s
    return "T".join([parts[0], parts[1].replace("-", ":")])


def _model_name_for_dir(dir, repo_dir_root):
    return os.path.relpath(dir, repo_dir_root)


def _get_model_info(args):
    assert args.data_path
    url = f"{HF_MODELS_API_BASE_URL}{args.model}"
    try:
        resp = urllib.request.urlopen(url)
    except urllib.error.HTTPError as e:
        if e.code in (404, 401):
            return None
        log.error("Error getting model info from %s: %s", url, e)
        raise SystemExit(1)
    else:
        data = json.load(resp)
        filename = os.path.join(args.data_path, "model.json")
        with open(filename, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        return filename


def _get_model_evals(args):
    """Returns an iteration of results ordered by eval time.

    The most recent eval results are returned last.
    """
    assert args.data_path
    model_parts = args.model.split("/")
    src_filenames = glob.glob(os.path.join(args.repo_dir, *model_parts, "*.json"))
    if not src_filenames:
        log.error("No results for model '%s'", args.model)
        raise SystemExit(
            "Try 'python import_results.py --list-models' for available models."
        )
    dest_filenames = [
        os.path.join(args.data_path, os.path.basename(path))  # \
        for path in src_filenames
    ]
    for src, name in zip(src_filenames, dest_filenames):
        shutil.copyfile(src, name)
    return dest_filenames


def _model_summary(model_name, model_filename, eval_filenames):
    summary = {}
    _apply_eval_results(model_name, eval_filenames, summary)
    _apply_model_info(model_name, model_filename, summary)
    return summary


def _apply_model_info(model_name, model_filename, summary):
    if model_filename:
        model_info = json.load(open(model_filename))
        _apply_dict(_model_info_summary(model_info, model_name), summary)
    else:
        _apply_dict(_model_missing_summary(model_name), summary)


def _model_missing_summary(model_name):
    return {
        "attributes": {
            "model": model_name,
            "is-deleted": True,
        }
    }


def _model_info_summary(data, model_name):
    card = data.get("cardData", {})
    config = data.get("config", {})
    return {
        "attributes": {
            "model": model_name,
            "architectures": config.get("architectures"),
            "model-type": config.get("model_type"),
            "is-merged": "merge" in card.get("tags", []),
            "is-moe": "moe" in card.get("tags", []),
            "license": card.get("license"),
            "params": _model_params(data),
            "modified": _last_modified_to_iso(data.get("lastModified")),
            "languages": card.get("language"),
            "downloads": data.get("downloads"),
            "likes": data.get("likes"),
        }
    }


def _last_modified_to_iso(s):
    return s[:-1] if s and s.endswith("Z") else s


def _model_params(data):
    val = data.get("safetensors", {}).get("total")
    if val is None:
        return None
    assert isinstance(val, int)
    return val / 1_000_000_000


def _apply_eval_results(model_name, eval_filenames, summary):
    assert eval_filenames
    sorted_evals = sorted(eval_filenames)
    for eval_src in sorted_evals:
        results = json.load(open(eval_src))
        _apply_dict(_results_summary(results, eval_src, model_name), summary)
    _apply_dict(
        {"attributes": {"last-eval": _date_for_eval_src(sorted_evals[-1])}}, summary
    )


def _date_for_eval_src(filename):
    name = os.path.basename(filename)
    m = RESULT_FILENAME_P.match(name)
    assert m, filename
    return _eval_filename_date_to_isoformat(m.group(1))


def _results_summary(results, results_src, model):
    config = _results_config(results)
    precision = _model_precision(config)
    return {
        "run": {
            "label": f"{model} {precision}",
        },
        "attributes": {"precision": precision},
        "metrics": _results_metrics(results, results_src),
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


def _results_metrics(results, src):
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
        task_f(task_results, metrics, {"src": src})
    return metrics


def _results_val(results, startswith, val_key):
    vals = [
        val
        for val in [
            results[key].get(val_key) for key in results if key.startswith(startswith)
        ]
        if val is not None
    ]
    return 100 * sum(vals) / len(vals) if vals else None


def _apply_results(results, key_prefix, attr_map, meta, results_key, target):
    result_vals = _acc_result_vals(key_prefix, attr_map, results)
    avg_vals = _avg_result_vals(result_vals)
    if avg_vals:
        target[results_key] = {**avg_vals, **meta}


def _acc_result_vals(key_prefix, attr_map, results):
    acc_vals = []
    for key in results:
        if not key.startswith(key_prefix):
            continue
        acc_val = {}
        acc_vals.append(acc_val)
        results_val = results[key]
        for attr in attr_map:
            try:
                val = results_val[attr]
            except KeyError:
                pass
            else:
                acc_val[attr_map[attr]] = val * 100
    return acc_vals


def _avg_result_vals(result_vals):
    state = {}  # keyed tuples of total, count
    for result_val in result_vals:
        for key, val in result_val.items():
            total, count = state.setdefault(key, (0, 0))
            state[key] = total + val, count + 1
    return {key: total / count for key, (total, count) in state.items()}


def _apply_arc(results, metrics, meta):
    _apply_results(
        results,
        "harness|arc:challenge",
        {"acc_norm": "value", "acc_norm_stderr": "stderr"},
        meta,
        "arc",
        metrics,
    )


def _apply_hellaswag(results, metrics, meta):
    _apply_results(
        results,
        "harness|hellaswag",
        {"acc_norm": "value", "acc_norm_stderr": "stderr"},
        meta,
        "hellaswag",
        metrics,
    )


def _apply_mmlu(results, metrics, meta):
    _apply_results(
        results,
        "harness|hendrycksTest-",
        {"acc": "value", "acc_stderr": "stderr"},
        meta,
        "mmlu",
        metrics,
    )


def _apply_truthfulqa(results, metrics, meta):
    _apply_results(
        results,
        "harness|truthfulqa:mc",
        {"mc2": "value", "mc2_stderr": "stderr"},
        meta,
        "truthfulqa",
        metrics,
    )


def _apply_winogrande(results, metrics, meta):
    _apply_results(
        results,
        "harness|winogrande",
        {"acc": "value", "acc_stderr": "stderr"},
        meta,
        "winogrande",
        metrics,
    )


def _apply_gsm8k(results, metrics, meta):
    _apply_results(
        results,
        "harness|gsm8k",
        {"acc": "value", "acc_stderr": "stderr"},
        meta,
        "gsm8k",
        metrics,
    )


def _apply_dict(src, dest):
    """Merges vals from src to dest.

    Matching keys in `dest` are replaced with values from `src` down the
    hierarchy.
    """
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
