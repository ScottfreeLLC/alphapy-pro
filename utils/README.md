# AlphaPy Pro Utilities

This directory contains utility scripts for managing AlphaPy Pro projects.

## Scripts

### cleanup_runs.sh

Cleans up old run directories in all projects, keeping only the most recent run for each project.

**Usage:**
```bash
# From the alphapy-pro root directory
./utils/cleanup_runs.sh
```

**Features:**
- Automatically detects all projects in the `projects/` directory
- Keeps only the most recent run directory for each project
- Shows how many directories were deleted
- Displays remaining disk usage for runs
- Works with any user's installation (no hardcoded paths)

**Example Output:**
```
AlphaPy Pro - Cleanup Runs Utility
Project root: /path/to/alphapy-pro

Processing kaggle project:
  Keeping most recent run: run_2024_06_28_1420
  Found 3 old run(s) to delete
  Deleting run_2024_06_27_1130
  Deleting run_2024_06_26_0945
  Deleting run_2024_06_25_1615

Processing time-series project:
  Keeping most recent run: run_2024_06_28_1015
  No old runs to delete

Cleanup complete!

Disk usage for remaining runs:
4.2M    /path/to/alphapy-pro/projects/kaggle/runs
2.8M    /path/to/alphapy-pro/projects/time-series/runs
```

## Adding New Utilities

When adding new utility scripts:

1. Place them in this `utils/` directory
2. Make them executable: `chmod +x utils/your_script.sh`
3. Use relative paths or detect the project root dynamically
4. Add documentation to this README
5. Follow the existing scripts' patterns for output and error handling