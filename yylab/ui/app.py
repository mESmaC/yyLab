from __future__ import annotations

from nicegui import ui


def start(port: int = 8080, show: bool = True) -> None:
    from . import pages  # noqa: F401

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
