#!/usr/bin/env bash
# start.sh — one-command launcher for the intent-engine web app.
#
# Creates/activates a virtualenv if needed, installs the runtime deps, and
# starts the FastAPI server. Meant for a quick local demo; production would
# use a process manager and a real ASGI deployment.
set -euo pipefail

# Resolve the repo root so the script works from any CWD.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Port is overridable: PORT=9000 ./start.sh
PORT="${PORT:-8000}"

# Create the virtualenv on first run so newcomers need only one command.
if [ ! -d ".venv" ]; then
  echo "→ Création de l'environnement virtuel (.venv)…"
  python3 -m venv .venv
fi

# Activate it and install the runtime dependencies (idempotent).
# shellcheck disable=SC1091  # path is created just above, not resolvable at lint time
source .venv/bin/activate
echo "→ Installation des dépendances…"
pip install -q -r requirements.txt

# Friendly reminder: the LLM engine needs Ollama; the app degrades gracefully
# without it (the LLM column is simply hidden).
if ! curl -s "http://localhost:11434/api/tags" >/dev/null 2>&1; then
  echo "⚠️  Ollama ne répond pas — le moteur LLM sera masqué."
  echo "    Démarrez-le avec 'ollama serve' puis 'ollama pull gemma4:e4b'."
fi

# Launch the server. --reload for a comfortable dev loop.
echo "→ Démarrage sur http://localhost:${PORT}"
exec uvicorn intent_engine.api:app --reload --port "${PORT}"
