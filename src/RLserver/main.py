import argparse
import logging
import sys

from . import auth, database
from .config import Config
from .retrain import run_retraining

logger = logging.getLogger("RLserver")


def _setup_logging():
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def run_server():
    from waitress import serve

    from .api import create_app

    auth.ensure_directories()
    database.init_db()
    app = create_app()
    logger.info(
        "PrintWatchAI server listening on http://%s:%s "
        "(token configured: %s, auto-retrain: %s)",
        Config.HOST,
        Config.PORT,
        auth.token_is_configured(),
        Config.AUTO_RETRAIN_ON_SYNC,
    )
    serve(app, host=Config.HOST, port=Config.PORT, threads=8)


def run_retrain_cli():
    auth.ensure_directories()
    database.init_db()
    logger.info("Running retrain socket manually")
    run_retraining()


def main():
    parser = argparse.ArgumentParser(description="PrintWatchAI server (RLserver)")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="run the retrain socket once and exit",
    )
    args = parser.parse_args()

    _setup_logging()
    if args.retrain:
        run_retrain_cli()
    else:
        run_server()


if __name__ == "__main__":
    sys.exit(main())