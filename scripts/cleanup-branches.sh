#!/bin/bash

# Script to clean up merged branches in AlphaPy repository
# Usage: ./scripts/cleanup-branches.sh [--dry-run]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    printf "${1}${2}${NC}\n"
}

DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            print_color $BLUE "Usage: $0 [--dry-run]"
            echo ""
            echo "Clean up merged branches in AlphaPy repository"
            echo ""
            echo "Options:"
            echo "  --dry-run    Show what would be deleted without actually deleting"
            echo "  -h, --help   Show this help message"
            exit 0
            ;;
        *)
            print_color $RED "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ "$DRY_RUN" = true ]; then
    print_color $YELLOW "🔍 DRY RUN MODE - No branches will be deleted"
else
    print_color $BLUE "🧹 Cleaning up merged branches..."
fi

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_color $RED "Error: Not in a git repository"
    exit 1
fi

# Fetch latest changes
print_color $YELLOW "Fetching latest changes..."
git fetch origin --prune

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
print_color $BLUE "Current branch: $CURRENT_BRANCH"

# Protected branches that should never be deleted
PROTECTED_BRANCHES=("main" "develop")

# Function to check if branch is protected
is_protected() {
    local branch=$1
    for protected in "${PROTECTED_BRANCHES[@]}"; do
        if [[ "$branch" == "$protected" ]]; then
            return 0
        fi
    done
    return 1
}

# Function to check if branch is merged
is_merged() {
    local branch=$1
    local target=$2
    git merge-base --is-ancestor $branch $target 2>/dev/null
}

# Clean up local branches merged into main
print_color $YELLOW "\n📋 Checking local branches merged into main..."
merged_count=0
for branch in $(git branch --format='%(refname:short)'); do
    # Skip current branch and protected branches
    if [[ "$branch" == "$CURRENT_BRANCH" ]] || is_protected "$branch"; then
        continue
    fi
    
    # Check if merged into main
    if is_merged "$branch" "origin/main"; then
        if [ "$DRY_RUN" = true ]; then
            print_color $YELLOW "Would delete: $branch (merged into main)"
        else
            print_color $GREEN "Deleting: $branch (merged into main)"
            git branch -d "$branch"
        fi
        ((merged_count++))
    fi
done

# Clean up local branches merged into develop (but not main)
print_color $YELLOW "\n📋 Checking local branches merged into develop..."
develop_count=0
for branch in $(git branch --format='%(refname:short)'); do
    # Skip current branch and protected branches
    if [[ "$branch" == "$CURRENT_BRANCH" ]] || is_protected "$branch"; then
        continue
    fi
    
    # Check if merged into develop but not main
    if is_merged "$branch" "origin/develop" && ! is_merged "$branch" "origin/main"; then
        if [ "$DRY_RUN" = true ]; then
            print_color $YELLOW "Would delete: $branch (merged into develop)"
        else
            print_color $GREEN "Deleting: $branch (merged into develop)"
            git branch -d "$branch"
        fi
        ((develop_count++))
    fi
done

# Clean up remote tracking branches that no longer exist
print_color $YELLOW "\n📋 Checking for stale remote tracking branches..."
stale_count=0
for branch in $(git branch -r --format='%(refname:short)' | grep '^origin/' | grep -v 'origin/HEAD'); do
    local_branch=${branch#origin/}
    
    # Skip protected branches
    if is_protected "$local_branch"; then
        continue
    fi
    
    # Check if the remote branch still exists
    if ! git ls-remote --exit-code --heads origin "$local_branch" >/dev/null 2>&1; then
        if [ "$DRY_RUN" = true ]; then
            print_color $YELLOW "Would clean stale tracking branch: $branch"
        else
            print_color $GREEN "Cleaning stale tracking branch: $branch"
        fi
        ((stale_count++))
    fi
done

# Actually prune remote tracking branches
if [ "$DRY_RUN" = false ] && [ $stale_count -gt 0 ]; then
    git remote prune origin
fi

# Show summary
print_color $BLUE "\n📊 Summary:"
if [ "$DRY_RUN" = true ]; then
    echo "Branches that would be deleted:"
    echo "  - Merged into main: $merged_count"
    echo "  - Merged into develop: $develop_count"
    echo "  - Stale remote tracking: $stale_count"
    echo ""
    print_color $YELLOW "Run without --dry-run to actually delete these branches"
else
    echo "Branches deleted:"
    echo "  - Merged into main: $merged_count"
    echo "  - Merged into develop: $develop_count"
    echo "  - Stale remote tracking: $stale_count"
    echo ""
    total=$((merged_count + develop_count + stale_count))
    if [ $total -gt 0 ]; then
        print_color $GREEN "✅ Cleaned up $total branches"
    else
        print_color $BLUE "ℹ️  No branches to clean up"
    fi
fi

# Show remaining branches
print_color $YELLOW "\n📋 Remaining local branches:"
git branch --format='%(refname:short)' | while read branch; do
    if [[ "$branch" == "$CURRENT_BRANCH" ]]; then
        print_color $GREEN "* $branch (current)"
    elif is_protected "$branch"; then
        print_color $BLUE "  $branch (protected)"
    else
        echo "  $branch"
    fi
done