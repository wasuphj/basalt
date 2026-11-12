"""`python -m serve.mantle` == `python -m serve.mantle.runner`."""
from .runner import main

if __name__ == "__main__":
    raise SystemExit(main())
