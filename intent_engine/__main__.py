"""Package entry point so ``python -m intent_engine ...`` works.

Module summary
--------------
Delegates to :func:`intent_engine.cli.main`. Keeping this file tiny (just
the delegation) is the conventional Python pattern: all real CLI logic
lives in ``cli.py`` where it can be imported and tested without spawning a
process.

Author
------
Project maintainers.
"""

from __future__ import annotations

from .cli import main

# ``python -m intent_engine`` imports this module as ``__main__``; forward the
# process exit code from the CLI so shell scripts can react to failures.
if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
