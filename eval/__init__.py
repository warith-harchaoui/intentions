"""Evaluation layer for the intent engines (coding standard rule 14).

Module summary
--------------
Groups the AI-evaluation assets: a committed labelled dataset
(``dataset.jsonl``), versioned pass/fail thresholds (``thresholds.py``), a
dependency-free accuracy harness (``harness.py``), and the DeepEval /
Giskard integrations. Kept in its own package so the fast unit-test suite
(``tests/``) stays separate from the heavier, framework-backed evaluation.

Author
------
Project maintainers.
"""

from __future__ import annotations
