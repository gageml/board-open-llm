# Open LLM Leaderboard - Gage Mirror

This is a Gage board mirror of the esteemed [Open LLM
Leaderboard](http://tinyurl.com/2l5uspcp) hosted on Hugging Face.

Model eval results are imported using the Gage `open-llm-import`
operation. Install [Gage](https://github.com/gageml/gage) if you haven't
already.

### Import a Single Model

To import a single model (e.g. for testing), run the `open-llm-import`
Gage operation.

```shell
gage run open-llm-import-batch
```

To list available models, run `import_results.py` directly with the
`--list-models` option.

```shell
python import_results.py --list-models
```

### Import All Models

To import all models, run `open-llm-import-batch`.

```shell
gage run open-llm-import model=<model>
```

Subsequent runs only import new results.

### Publish Open LLM Board

Requirements:

- [Rclone](https://rclone.org/install/)
- Credentials to publish the board

Once RClone is installed, create a remote:

```shell
rclone config create gage s3 provider=CloudFlare endpoint=https://d5f5a59ff84ba96f4eba7a056261fd17.r2.cloudflarestorage.com acl=private
```

Verify that the `gage` remote is created:

```shell
rclone config show gage
```

Obtain the credentials to publish the board to Gage and load them into
the environment.

In bash (or compatible):

```shell
source <(gpg -d <credentials.gpg)
```

In Nushell:

```nu
load-env (gpg -d credentials.nuon.gpg | from nuon)
```

Run the `publish` Gage operation. It's a good idea to test credentials
first.

With credential test only (does not publish):

```shell
gage run publish test_credentials=true
```

If the credentials are okay, publish the board:

```shell
gage run publish test_credentials=true
```

### View Board

To view the published board, visit:

<https://gage-live.garrett-d5f.workers.dev/open-llm>
