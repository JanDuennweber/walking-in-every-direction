#!/bin/bash
# Publish local changes to the public GitHub repository: stage, commit, push.
# Usage: ./commitRepo.sh ["commit message"]
# Needs GitHub credentials in git (SSH key registered with GitHub, or a
# personal access token for HTTPS).
set -e
cd "$(dirname "$0")"

git add -A
if git diff --cached --quiet; then
  echo "Nothing new to commit, pushing existing commits."
else
  git commit -m "${1:-Update from local $(date '+%Y-%m-%d %H:%M')}"
fi
git push -u origin main
