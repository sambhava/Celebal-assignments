"""Enables ``python -m copilot ...`` by delegating to the CLI."""

from copilot.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
