# Open LLM Leaderboard - Gage Mirror

This is a Gage board mirror of the esteemed [Open LLM
Leaderboard](http://tinyurl.com/2l5uspcp) hosted on Hugging Face.

Model eval results are imported using the Gage `open-llm-import`
operation. Install [Gage](https://github.com/gageml/gage) if you haven't
already.

Import a model:

```shell
gage run open-llm-import model=<model>
```

To list available models, visit the
[leaderboard](http://tinyurl.com/2l5uspcp) or use the local Python
script [`import_results.py`](import_results.py).

```shell
python import_results.py --list-models
```

View the latest import results using `gage show`:

```shell
gage show
```

View the latest import for a particular model using a `--where` filter:

```shell
gage show --where <model>
```

Generate a Gage board definition:

```shell
gage board --json --config board.json
```
