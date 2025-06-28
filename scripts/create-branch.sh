#!/bin/bash

# Script to create a new branch following AlphaPy naming conventions
# Usage: ./scripts/create-branch.sh <type> <description> [issue-number]

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

# Function to show usage
show_usage() {
    print_color $BLUE "Usage: $0 <type> <description> [issue-number]"
    echo ""
    print_color $YELLOW "Types:"
    echo "  feature  - New features and enhancements"
    echo "  bugfix   - Bug fixes for current development"
    echo "  hotfix   - Critical fixes for immediate release"
    echo "  release  - Preparing new releases"
    echo ""
    print_color $YELLOW "Examples:"
    echo "  $0 feature \"Add LSTM model\" 123"
    echo "  $0 bugfix \"Fix memory leak\" 456"
    echo "  $0 hotfix \"Security patch\""
    echo "  $0 release \"v3.1.0\""
    exit 1
}

# Check arguments
if [ $# -lt 2 ]; then
    print_color $RED "Error: Missing required arguments"
    show_usage
fi

TYPE=$1
DESCRIPTION=$2
ISSUE_NUMBER=$3

# Validate type
case $TYPE in
    feature|bugfix|hotfix|release)
        ;;
    *)
        print_color $RED "Error: Invalid type '$TYPE'"
        show_usage
        ;;
esac

# Clean up description (lowercase, replace spaces with hyphens)
CLEAN_DESC=$(echo "$DESCRIPTION" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/--*/-/g' | sed 's/^-\|-$//g')

# Build branch name
if [ -n "$ISSUE_NUMBER" ] && [ "$TYPE" != "release" ]; then
    BRANCH_NAME="${TYPE}/issue-${ISSUE_NUMBER}-${CLEAN_DESC}"
else
    BRANCH_NAME="${TYPE}/${CLEAN_DESC}"
fi

# Determine base branch
case $TYPE in
    feature|bugfix)
        BASE_BRANCH="develop"
        ;;
    hotfix)
        BASE_BRANCH="main"
        ;;
    release)
        BASE_BRANCH="develop"
        ;;
esac

print_color $BLUE "Creating branch '$BRANCH_NAME' from '$BASE_BRANCH'..."

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_color $RED "Error: Not in a git repository"
    exit 1
fi

# Check if branch already exists
if git show-ref --verify --quiet refs/heads/$BRANCH_NAME; then
    print_color $RED "Error: Branch '$BRANCH_NAME' already exists locally"
    exit 1
fi

if git show-ref --verify --quiet refs/remotes/origin/$BRANCH_NAME; then
    print_color $RED "Error: Branch '$BRANCH_NAME' already exists on remote"
    exit 1
fi

# Fetch latest changes
print_color $YELLOW "Fetching latest changes..."
git fetch origin

# Check if base branch exists
if ! git show-ref --verify --quiet refs/remotes/origin/$BASE_BRANCH; then
    print_color $RED "Error: Base branch 'origin/$BASE_BRANCH' does not exist"
    exit 1
fi

# Switch to base branch and pull latest
print_color $YELLOW "Switching to $BASE_BRANCH and pulling latest changes..."
git checkout $BASE_BRANCH
git pull origin $BASE_BRANCH

# Create and checkout new branch
print_color $YELLOW "Creating new branch..."
git checkout -b $BRANCH_NAME

# Push to remote and set upstream
print_color $YELLOW "Pushing to remote and setting upstream..."
git push -u origin $BRANCH_NAME

print_color $GREEN "✅ Successfully created branch '$BRANCH_NAME'"
echo ""
print_color $BLUE "Next steps:"
echo "1. Make your changes and commit them"
echo "2. Push your changes: git push origin $BRANCH_NAME"
echo "3. Create a Pull Request to merge back to $BASE_BRANCH"

if [ -n "$ISSUE_NUMBER" ]; then
    echo "4. Link your PR to issue #$ISSUE_NUMBER"
fi

echo ""
print_color $YELLOW "Remember to follow the coding guidelines and add tests!"