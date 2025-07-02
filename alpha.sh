#!/usr/bin/env bash

# Loop alpha from -30 to +30 in steps of 1
for alpha in $(seq 8 0.1 10); do
  # For each alpha, loop over the specified dell values
  for dell in -4 -2 0 2 4; do
    # Execute the notebook with papermill, passing parameters and naming the output
    papermill alpha_tester.ipynb \
      notebook/output_alpha${alpha}_dell${dell}.ipynb \
      -p alpha "${alpha}" \
      -p dell  "${dell}"
  done
done