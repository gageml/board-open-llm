import datetime
import json
import os
import sys

from _util import sync_results_repo

model = "Eang/Marcoro14-7B-slerp"
eval = "2024-04-11T06-28-30.403172"


def main():
    print("Getting latest eval results", file=sys.stderr)
    repo_dir = sync_results_repo()
    eval_data = _read_eval_data(repo_dir, model, eval)
    print("Summarizing results", file=sys.stderr)
    summary = _summarize(eval_data)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _read_eval_data(repo_dir, model, eval):
    eval_filename = os.path.join(repo_dir, *model.split("/"), f"results_{eval}.json")
    with open(eval_filename) as f:
        return json.load(f)


def _summarize(eval_data):
    summary = {}
    _apply_eval_date(summary)
    _apply_arc(eval_data, summary)
    _apply_hellaswag(eval_data, summary)
    _apply_mmlu(eval_data, summary)
    _apply_truthfulqa(eval_data, summary)
    _apply_winogrande(eval_data, summary)
    _apply_gsm8k(eval_data, summary)
    return summary


def _apply_eval_date(summary):
    attrs = summary.setdefault("attributes", {})
    parsed = datetime.datetime.strptime(eval, "%Y-%m-%dT%H-%M-%S.%f")
    attrs["eval_date"] = parsed.isoformat()


def _parse_datetime(eval):
    return


def _apply_arc(eval_data, summary):
    _apply_metric(
        eval_data,
        "harness|arc:challenge",
        {"acc_norm": "value", "acc_norm_stderr": "stderr"},
        "arc",
        summary,
    )


def _apply_hellaswag(eval_data, summary):
    _apply_metric(
        eval_data,
        "harness|hellaswag",
        {"acc_norm": "value", "acc_norm_stderr": "stderr"},
        "hellaswag",
        summary,
    )


def _apply_mmlu(eval_data, summary):
    _apply_metric(
        eval_data,
        "harness|hendrycksTest-",
        {"acc": "value", "acc_stderr": "stderr"},
        "mmlu",
        summary,
    )


def _apply_truthfulqa(eval_data, summary):
    _apply_metric(
        eval_data,
        "harness|truthfulqa:mc",
        {"mc2": "value", "mc2_stderr": "stderr"},
        "truthfulqa",
        summary,
    )


def _apply_winogrande(eval_data, summary):
    _apply_metric(
        eval_data,
        "harness|winogrande",
        {"acc": "value", "acc_stderr": "stderr"},
        "winogrande",
        summary,
    )


def _apply_gsm8k(eval_data, summary):
    _apply_metric(
        eval_data,
        "harness|gsm8k",
        {"acc": "value", "acc_stderr": "stderr"},
        "gsm8k",
        summary,
    )


def _apply_metric(eval_data, key_prefix, attr_map, results_key, summary):
    results = eval_data["results"]
    metrics = summary.setdefault("metrics", {})
    result_vals = _acc_result_vals(key_prefix, attr_map, results)
    avg_vals = _avg_result_vals(result_vals)
    if avg_vals:
        metrics[results_key] = avg_vals


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


if __name__ == "__main__":
    main()
