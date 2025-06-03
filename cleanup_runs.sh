#!/bin/bash

# Script to delete all runs directories except the most recent one in each project

# Kaggle project
cd /Users/markconway/Projects/alphapy-pro/projects/kaggle/runs
latest_run_kaggle=$(ls -d run_* | sort -r | head -n 1)
echo "Keeping most recent run in kaggle project: $latest_run_kaggle"
for dir in run_*; do
  if [ "$dir" != "$latest_run_kaggle" ]; then
    echo "Deleting $dir in kaggle project"
    rm -rf "$dir"
  fi
done

# Time-series project
cd /Users/markconway/Projects/alphapy-pro/projects/time-series/runs
latest_run_ts=$(ls -d run_* | sort -r | head -n 1)
echo "Keeping most recent run in time-series project: $latest_run_ts"
for dir in run_*; do
  if [ "$dir" != "$latest_run_ts" ]; then
    echo "Deleting $dir in time-series project"
    rm -rf "$dir"
  fi
done

echo "Cleanup complete!"
