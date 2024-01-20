# Tests

    >>> from import_results import _acc_result_vals

    >>> _acc_result_vals("foo", {"x": "x2"}, {
    ...     "foo": {"x": 0.1},
    ...     "foo2": {"x": 0.2}
    ... })
    [{'x2': 10.0}, {'x2': 20.0}]

    >>> from import_results import _avg_result_vals

    >>> _avg_result_vals([{"x2": 1}, {"x2": 2}, {"y": 10}])
    {'x2': 1.5, 'y': 10.0}

    >>> from import_results import (
    ...     _apply_arc,
    ...     _apply_hellaswag,
    ...     _apply_mmlu,
    ...     _apply_truthfulqa,
    ...     _apply_winogrande,
    ...     _apply_gsm8k,
    ... )

    >>> results = {
    ...   "harness|arc:challenge|25": {
    ...     "acc": 0.53,
    ...     "acc_stderr": 0.015,
    ...     "acc_norm": 0.5,
    ...     "acc_norm_stderr": 0.05
    ...   },
    ...   "harness|hellaswag|10": {
    ...     "acc": 0.593,
    ...     "acc_stderr": 0.0049,
    ...     "acc_norm": 0.83,
    ...     "acc_norm_stderr": 0.0043
    ...   },
    ...   "harness|hendrycksTest-abstract_algebra|5": {
    ...     "acc": 0.4,
    ...     "acc_stderr": 0.04,
    ...     "acc_norm": 0.32,
    ...     "acc_norm_stderr": 0.046
    ...   },
    ...   "harness|hendrycksTest-anatomy|5": {
    ...     "acc": 0.6,
    ...     "acc_stderr": 0.06,
    ...     "acc_norm": 0.503,
    ...     "acc_norm_stderr": 0.043
    ...   },
    ...   "harness|truthfulqa:mc|0": {
    ...     "mc1": 0.2607,
    ...     "mc1_stderr": 0.01536,
    ...     "mc2": 0.4,
    ...     "mc2_stderr": 0.01
    ...   },
    ...   "harness|gsm8k|5": {
    ...     "acc": 0.04,
    ...     "acc_stderr": 0.005
    ...   },
    ...   "harness|winogrande|5": {
    ...     "acc": 0.76,
    ...     "acc_stderr": 0.012
    ...   }
    ... }

    >>> metrics = {}

    >>> _apply_arc(results, metrics, {"src": "test"})
    >>> _apply_hellaswag(results, metrics, {"src": "test"})
    >>> _apply_mmlu(results, metrics, {"src": "test-2"})
    >>> _apply_truthfulqa(results, metrics, {"src": "test-2"})
    >>> _apply_winogrande(results, metrics, {"src": "test-2"})
    >>> _apply_gsm8k(results, metrics, {"src": "test-3"})

    >>> metrics  # +json +diff
    {
      "arc": {
        "src": "test",
        "stderr": 5.0,
        "value": 50.0
      },
      "gsm8k": {
        "src": "test-3",
        "stderr": 0.5,
        "value": 4.0
      },
      "hellaswag": {
        "src": "test",
        "stderr": 0.43,
        "value": 83.0
      },
      "mmlu": {
        "src": "test-2",
        "stderr": 5.0,
        "value": 50.0
      },
      "truthfulqa": {
        "src": "test-2",
        "stderr": 1.0,
        "value": 40.0
      },
      "winogrande": {
        "src": "test-2",
        "stderr": 1.2,
        "value": 76.0
      }
    }
