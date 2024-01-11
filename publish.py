import logging
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
)
log = logging.getLogger()

DATA_JSON = "data.json"


def main():
    _check_rclone()
    board = _generate_board()
    _save_board(board)
    _publish_board()


def _check_rclone():
    try:
        out = subprocess.check_output(
            "rclone --version",
            shell=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        raise SystemExit(
            "Cannot find rclone\n"
            "Confirm that it's installed and available on the system path.\n"
            "Refer to https://rclone.org/install/ for help."
        )
    else:
        log.info("Found %s", out.split("\n")[0])


def _generate_board():
    log.info("Generating board using board.json")
    return subprocess.check_output(
        "gage board --config board.json --json",
        shell=True,
        text=True,
    )


def _save_board(board):
    log.info("Saving board to %s", DATA_JSON)
    with open(DATA_JSON, "w") as f:
        f.write(board)


def _publish_board():
    pass


if __name__ == "__main__":
    main()
