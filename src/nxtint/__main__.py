"""Main entry point for nxtint."""

from pathlib import Path

from nxtint.utils.logging import setup_logger


def main():
    """Run the main function."""
    # Setup root logger with file output
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger = setup_logger("nxtint", log_file=log_dir / "nxtint.log")

    logger.info("Starting nxtint")


if __name__ == "__main__":
    main()
