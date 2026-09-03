from __future__ import annotations

from nicegui import ui


def start(port: int = 8080, show: bool = True) -> None:
    from ..ingest.pipeline import startup_housekeeping
    from . import pages  # noqa: F401

    # Complete or fail runs the previous app instance left behind and delete
    # retired corpus outputs that nothing holds open.
    try:
        startup_housekeeping()
    except Exception:  # noqa: BLE001 - housekeeping must never block startup
        pass
    ui.run(
        host="127.0.0.1",
        port=port,
        title="yyLab",
        dark=True,
        reload=False,
        show=show,
        storage_secret="yylab-local-storage",
        favicon="🧪",
    )
