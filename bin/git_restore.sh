#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/data/xyh/code/HoloRAG"

cd "$REPO_DIR"

echo "[INFO] Repo: $REPO_DIR"

if [ ! -d .git ]; then
  echo "[ERROR] This is not a git repository: $REPO_DIR"
  exit 1
fi

echo "[WARN] This will discard all uncommitted changes in tracked files."
echo "[WARN] It will also remove untracked files not ignored by .gitignore."
read -r -p "Type YES to continue: " CONFIRM

if [ "$CONFIRM" != "YES" ]; then
  echo "[INFO] Aborted."
  exit 0
fi

echo "[INFO] Restoring tracked files to HEAD..."
git restore .

echo "[INFO] Removing untracked files..."
git clean -fd

echo "[INFO] Current status:"
git status --short

echo "[INFO] Restored to latest committed version:"
git log -1 --oneline
