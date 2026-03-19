#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/data/xyh/code/HoloRAG"

cd "$REPO_DIR"

echo "[INFO] Repo: $REPO_DIR"

if [ ! -d .git ]; then
  echo "[ERROR] This is not a git repository: $REPO_DIR"
  exit 1
fi

MSG="${1:-update: $(date '+%Y-%m-%d %H:%M:%S')}"

echo "[INFO] Checking status..."
git status --short

echo "[INFO] Adding tracked/untracked files (respecting .gitignore)..."
git add .

if git diff --cached --quiet; then
  echo "[INFO] No staged changes to commit."
  exit 0
fi

echo "[INFO] Committing with message: $MSG"
git commit -m "$MSG"

echo "[INFO] Latest commit:"
git log -1 --oneline
