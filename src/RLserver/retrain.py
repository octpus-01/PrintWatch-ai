"""Retrain socket.

The server calls ``run_retraining`` automatically after a sync_data batch is
fully stored (see Config.AUTO_RETRAIN_ON_SYNC), or manually via
``python -m src.RLserver.main --retrain``. Implement ``run_retraining`` with
your own training pipeline and leave the rest of the file untouched.
"""

import logging
import threading


from . import database, models_store

logger = logging.getLogger(__name__)

_running = False
_lock = threading.Lock()


def run_retraining() -> None:
    """YOUR retrain logic goes here.a

    1) Pull records not yet used for training:
         records = database.get_unused_records()
       each record is a dict:
         {id, device_id, timestamp, label, confidence, image_path, model_path}

    2) Train YOUR way. Load decoded JPEGs from record["image_path"], labels
       from record["label"]. Nothing here is imposed; reuse
       src/RLserver/rl_agent.py or src/pretrain/pretrain.py as you like.

    3) Export the trained model to ONNX and write it into the models dir:
         models/server_vN.onnx   (N = version number)

    4) Tell the server this model is the new latest:
         version = models_store.next_version()
         models_store.publish_model(version, "models/server_vN.onnx")
       The next GET /api/v1/get_latest_model will serve it with
       X-Model-Version: vN.

    5) Only after a successful train, free the records for the next round:
         database.mark_records_used([r["id"] for r in records])
    """

    


    raise NotImplementedError(
        "Implement run_retraining() in src/RLserver/retrain.py to enable retraining"
    )


def trigger_retrain() -> bool:
    """Run run_retraining in a background thread, at most one at a time.

    Returns True if a run was dispatched, False if one is already running.
    """
    global _running
    with _lock:
        if _running:
            logger.warning("Retraining already in progress, skip this trigger")
            return False
        _running = True

    def _worker():
        global _running
        try:
            logger.info("Retraining started")
            run_retraining()
            logger.info("Retraining finished")
        except NotImplementedError:
            logger.warning(
                "run_retraining() not implemented yet - edit src/RLserver/retrain.py"
            )
        except Exception:
            logger.exception("Retraining failed")
        finally:
            with _lock:
                _running = False

    threading.Thread(target=_worker, name="rlserver-retrain", daemon=True).start()
    return True