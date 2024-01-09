import subprocess

models = (
    subprocess.check_output(
        "python import_results.py --list-models",
        text=True,
        shell=True,
    )
    .strip()
    .split()
)

for model in models:
    print(f"Importing {model}")
    subprocess.check_output(
        f"gage run open-llm-import model='{model}' -y",
        shell=True,
    )
