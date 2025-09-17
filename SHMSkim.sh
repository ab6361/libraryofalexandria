#!/bin/bash

# Usage:
#   ./SHMSkim.sh /path/to/runlist.txt
# (Make sure SHMSkim.C is in the same directory or adjust path below)

RUNLIST="$1"

if [[ -z "$RUNLIST" ]]; then
  echo "Usage: $0 runlist.txt"
  exit 1
fi

if [[ ! -f "$RUNLIST" ]]; then
  echo "Error: Runlist file '$RUNLIST' not found!"
  exit 1
fi

# Loop through each line in runlist.txt
while read -r run; do
  # Skip blank lines and comments starting with '#'
  [[ -z "$run" || "$run" =~ ^# ]] && continue

  echo ">>> Skimming run $run ..."
  root -l -b -q "SHMSkim.C+($run)"
done < "$RUNLIST"

echo ">>> All runs processed."
