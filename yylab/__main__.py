from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="yylab")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    from yylab.ui.app import start

    start(port=args.port, show=not args.no_browser)


if __name__ in {"__main__", "__mp_main__"}:
    main()
