#!/usr/bin/env bash
# ./run.sh serve [--port 8765]   web page for sending photos from a phone (reads .env)
# ./run.sh test                  run the tests
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -x .venv/bin/python ]] || [[ pyproject.toml -nt .venv/.installed ]]; then
  echo "Setting up .venv…"
  if command -v uv >/dev/null; then
    uv venv -q --allow-existing .venv
    uv pip install -q -p .venv/bin/python -e '.[test]'
  else
    python3 -m venv .venv
    .venv/bin/pip install -q -e '.[test]'
  fi
  touch .venv/.installed
fi

command=${1:-serve}
[[ $# -gt 0 ]] && shift
case "$command" in
  serve) exec .venv/bin/python -m mapper.web "$@" ;;
  test) exec .venv/bin/python -m pytest -q "$@" ;;
  *) echo "usage: ./run.sh serve [--port 8765] | ./run.sh test" >&2; exit 2 ;;
esac
