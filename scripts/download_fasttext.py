"""Download the pretrained French fastText model (cc.fr.300.bin).

Module summary
--------------
The ``fasttext_pretrained`` engine (Approach 3) needs the official French
fastText vectors, ``cc.fr.300.bin`` (~4.5 GB compressed, ~7 GB on disk),
trained on Common Crawl + Wikipedia. This helper downloads it once into the
repository's ``models/`` folder, where the engine looks for it by default.
The download is large and one-off, so it lives in a script rather than being
pulled implicitly at import time.

Until the model is present, the pretrained engine is simply hidden from the
comparator (graceful degradation) — the other four engines run regardless.

Usage
-----
    python scripts/download_fasttext.py

Author
------
Project maintainers.
"""

from __future__ import annotations

import os
from pathlib import Path

# Target folder: the repo's ``models/`` (gitignored — the binary is huge).
_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> int:
    """Download ``cc.fr.300.bin`` into ``models/`` if not already present.

    Returns
    -------
    int
        Process exit code (0 on success).

    Notes
    -----
    ``fasttext.util.download_model`` writes to the current working directory,
    so we ``chdir`` into ``models/`` first. It also downloads a ``.gz`` next
    to the ``.bin``; we remove it afterwards to reclaim ~4.5 GB.
    """
    # Import lazily so ``--help`` and import-time checks do not require fastText.
    import fasttext.util

    # Ensure the destination exists, then download there.
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = _MODELS_DIR / "cc.fr.300.bin"
    if target.is_file():
        # Idempotent: nothing to do if the model is already downloaded.
        print(f"Déjà présent : {target}")
        return 0

    # ``download_model`` writes into the CWD, so switch into ``models/``.
    print("Téléchargement de cc.fr.300.bin (~4,5 Go)… cela peut être long.")
    os.chdir(_MODELS_DIR)
    fasttext.util.download_model("fr", if_exists="ignore")

    # The helper leaves the compressed ``.gz`` behind; drop it to save space.
    gz = _MODELS_DIR / "cc.fr.300.bin.gz"
    if gz.is_file():
        gz.unlink()
    print(f"Terminé : {target}")
    return 0


if __name__ == "__main__":  # pragma: no cover - one-off script
    raise SystemExit(main())
