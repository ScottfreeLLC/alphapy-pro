#!/bin/bash

# Script to delete all runs directories except the most recent one in each project
# This script should be run from the alphapy-pro root directory

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the script's directory (should be utils/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root directory (parent of utils/)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo -e "${GREEN}AlphaPy Pro - Cleanup Runs Utility${NC}"
echo "Project root: $PROJECT_ROOT"
echo ""

# Function to clean runs in a project directory
cleanup_project_runs() {
    local project_name=$1
    local project_path=$2
    
    if [ ! -d "$project_path/runs" ]; then
        echo -e "${YELLOW}No runs directory found in $project_name project${NC}"
        return
    fi
    
    cd "$project_path/runs" || return
    
    # Check if there are any run directories
    run_count=$(find . -maxdepth 1 -name "run_*" -type d | wc -l)
    
    if [ "$run_count" -eq 0 ]; then
        echo -e "${YELLOW}No run directories found in $project_name project${NC}"
        return
    fi
    
    # Get the most recent run directory
    latest_run=$(ls -d run_* 2>/dev/null | sort -r | head -n 1)
    
    if [ -z "$latest_run" ]; then
        echo -e "${YELLOW}No runs to clean in $project_name project${NC}"
        return
    fi
    
    echo -e "${GREEN}Processing $project_name project:${NC}"
    echo "  Keeping most recent run: $latest_run"
    
    # Count directories to be deleted
    delete_count=0
    for dir in run_*; do
        if [ "$dir" != "$latest_run" ]; then
            ((delete_count++))
        fi
    done
    
    if [ "$delete_count" -eq 0 ]; then
        echo "  No old runs to delete"
    else
        echo "  Found $delete_count old run(s) to delete"
        
        # Delete old runs
        for dir in run_*; do
            if [ "$dir" != "$latest_run" ]; then
                echo -e "  ${RED}Deleting${NC} $dir"
                rm -rf "$dir"
            fi
        done
    fi
    
    echo ""
}

# Check if projects directory exists
if [ ! -d "$PROJECT_ROOT/projects" ]; then
    echo -e "${RED}Error: projects directory not found at $PROJECT_ROOT/projects${NC}"
    echo "Please run this script from the alphapy-pro root directory"
    exit 1
fi

# Process all projects
cd "$PROJECT_ROOT/projects" || exit 1

# Find all subdirectories in projects/ and process them
for project_dir in */; do
    if [ -d "$project_dir" ]; then
        project_name=${project_dir%/}
        cleanup_project_runs "$project_name" "$PROJECT_ROOT/projects/$project_name"
    fi
done

echo -e "${GREEN}Cleanup complete!${NC}"

# Show disk space saved (optional)
if command -v du &> /dev/null; then
    echo ""
    echo "Disk usage for remaining runs:"
    du -sh "$PROJECT_ROOT"/projects/*/runs 2>/dev/null | grep -v "No such file"
fi