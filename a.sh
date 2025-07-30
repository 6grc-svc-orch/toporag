#!/bin/bash

# Usage:
#   ./delete_empty_dirs.sh /path/to/target/folder           → deletes empty dirs
#   ./delete_empty_dirs.sh /path/to/target/folder --dry-run → shows what would be deleted

TARGET_DIR="$1"
DRY_RUN="$2"

if [ -z "$TARGET_DIR" ]; then
  echo "Usage: $0 /path/to/target/folder [--dry-run]"
  exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
  echo "Error: '$TARGET_DIR' is not a directory."
  exit 1
fi

# Function to check if a directory is empty or only contains "[]"-like files
is_effectively_empty() {
  local dir="$1"
  shopt -s nullglob
  local files=("$dir"/*)
  shopt -u nullglob

  # If no files, it's empty
  [ ${#files[@]} -eq 0 ] && return 0

  # Check each file
  for file in "${files[@]}"; do
    if [ -d "$file" ]; then
      # It's a subdirectory, so not empty
      return 1
    fi

    # Check if file content is only "[]", possibly with whitespace
    if ! grep -qP '^\s*\[\s*\]\s*$' "$file"; then
      return 1
    fi
  done

  return 0
}

# Collect candidate directories
mapfile -t CANDIDATES < <(find "$TARGET_DIR" -type d)

# Filter and optionally delete
for dir in "${CANDIDATES[@]}"; do
  if is_effectively_empty "$dir"; then
    if [ "$DRY_RUN" = "--dry-run" ]; then
      echo "Would delete: $dir"
    else
      rm -r "$dir"
      echo "Deleted: $dir"
    fi
  fi
done
