import logging
import os
import sys

from subprocess import run

log = logging.getLogger()

OPEN_LLM_RESULTS_REPO = "https://huggingface.co/datasets/open-llm-leaderboard/results"

LOCAL_REPOS_DIR = ".repos"


def sync_results_repo():
    repo = OPEN_LLM_RESULTS_REPO

    repo_dir = _repo_dir_for_repo(repo)
    if os.path.exists(repo_dir):
        run(["git", "fetch"], cwd=repo_dir, stdout=sys.stderr)
        run(["git", "merge", "-n"], cwd=repo_dir, stdout=sys.stderr)
    else:
        run(
            ["git", "clone", repo, repo_dir],
            env={"GIT_LFS_SKIP_SMUDGE": "1"},
            stdout=sys.stderr,
        )
    return repo_dir


def _repo_dir_for_repo(repo):
    parent_dir = os.getenv("PARENT_PWD") or "."
    return os.path.join(parent_dir, LOCAL_REPOS_DIR, repo.split("/")[-1])
