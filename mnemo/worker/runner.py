"""Worker process. Sprint 0: sleeps in a loop. Sprint 1: drives the pipeline."""
import logging
import time

from mnemo.config import get_settings
from mnemo.db import init_for_settings

log = logging.getLogger(__name__)


def run_worker() -> None:
    settings = get_settings()
    conn = init_for_settings(settings)
    log.info("worker %s starting (poll=%ss)", settings.worker_id,
             settings.worker_poll_interval_seconds)
    try:
        while True:
            # TODO Sprint 1: claim task from processing_queue and execute pipeline
            time.sleep(settings.worker_poll_interval_seconds)
    except KeyboardInterrupt:
        log.info("worker shutting down")
    finally:
        conn.close()
