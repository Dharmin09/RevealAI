#!/usr/bin/env bash
# Convenience script to stage, commit, and push all changes to origin
# Usage: ./tools/git_push_all.sh "commit message" [branch]

set -euo pipefail

MSG=${1:-}
BRANCH=${2:-main}

if ! command -v git >/dev/null 2>&1; then
  echo "git is not installed or not in PATH"
  exit 2
fi

ROOT=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [ -z "$ROOT" ]; then
  echo "Not inside a git repository"
  exit 3
fi
cd "$ROOT"

if [ -z "$MSG" ]; then
  MSG="chore: sync workspace $(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi

echo "Repository root: $ROOT"
echo "Branch: $BRANCH"

echo "Staging changes..."
git add -A

if [ -n "$(git status --porcelain)" ]; then
  echo "Committing: $MSG"
  git commit -m "$MSG" || true
else
  echo "No changes to commit"
fi

echo "Fetching and rebasing from origin/$BRANCH..."
git fetch origin "$BRANCH" || true
git pull --rebase origin "$BRANCH" || true

ORIGIN_URL=$(git remote get-url origin || true)

if [ -n "${GITHUB_TOKEN:-}" ] && [[ "$ORIGIN_URL" == https://* ]]; then
  PUSH_URL="${ORIGIN_URL/https:\/\//https://$GITHUB_TOKEN@}"
  echo "Temporarily setting origin to token-authenticated url"
  git remote set-url origin "$PUSH_URL"
  git push origin "$BRANCH"
  git remote set-url origin "$ORIGIN_URL"
else
  git push origin "$BRANCH"
fi

echo "Push complete"
