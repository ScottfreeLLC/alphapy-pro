# Git Workflow Guide for AlphaPy Pro

This document outlines the Git workflow process for contributing to AlphaPy Pro, including branching strategy, pull request process, and release management.

## Branch Strategy

AlphaPy Pro uses a **GitFlow-inspired** branching model with the following structure:

### Main Branches

- **`main`** - Production-ready code
  - Always stable and deployable
  - Protected branch with strict rules
  - Only accepts PRs from `develop` or `hotfix/*` branches
  - Every commit should be tagged with a version

- **`develop`** - Integration branch for active development
  - All feature development merges here first
  - Always in a "ready to release" state
  - Protected branch with PR requirements
  - Regularly merged to `main` for releases

### Supporting Branches

- **`feature/*`** - New features and enhancements
  - Branch from: `develop`
  - Merge back to: `develop`
  - Naming: `feature/issue-123-short-description`

- **`bugfix/*`** - Bug fixes for the current development cycle
  - Branch from: `develop`
  - Merge back to: `develop`
  - Naming: `bugfix/issue-456-short-description`

- **`hotfix/*`** - Critical fixes that need immediate release
  - Branch from: `main`
  - Merge back to: `main` AND `develop`
  - Naming: `hotfix/critical-issue-description`

- **`release/*`** - Preparing new releases
  - Branch from: `develop`
  - Merge back to: `main` and `develop`
  - Naming: `release/v3.1.0`

## Workflow Processes

### 1. Feature Development

#### Step 1: Create Issue
```bash
# Create a GitHub issue describing the feature
# Get the issue number (e.g., #123)
```

#### Step 2: Create Feature Branch
```bash
# Start from develop
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/issue-123-add-lstm-model

# Push to origin
git push -u origin feature/issue-123-add-lstm-model
```

#### Step 3: Develop Feature
```bash
# Make your changes
# Commit regularly with descriptive messages
git add .
git commit -m "Add: LSTM model implementation"

# Push changes
git push origin feature/issue-123-add-lstm-model
```

#### Step 4: Create Pull Request
1. Go to GitHub and create a PR from your feature branch to `develop`
2. Fill out the PR template completely
3. Link to the related issue
4. Request reviews from maintainers

#### Step 5: Code Review & Merge
- Address review feedback
- Ensure all CI checks pass
- Squash and merge when approved

### 2. Bug Fix Process

#### Step 1: Create Issue
```bash
# Report the bug with reproduction steps
# Get issue number (e.g., #456)
```

#### Step 2: Create Bugfix Branch
```bash
git checkout develop
git pull origin develop
git checkout -b bugfix/issue-456-fix-memory-leak
git push -u origin bugfix/issue-456-fix-memory-leak
```

#### Step 3: Fix and Test
```bash
# Fix the bug
# Add tests to prevent regression
git add .
git commit -m "Fix: memory leak in data processing"
git push origin bugfix/issue-456-fix-memory-leak
```

#### Step 4: Create Pull Request
- Same process as features
- Include test that reproduces the bug
- Explain the root cause and solution

### 3. Hotfix Process (Critical Issues)

#### Step 1: Create Hotfix Branch
```bash
# Start from main for critical fixes
git checkout main
git pull origin main
git checkout -b hotfix/security-vulnerability
```

#### Step 2: Fix Issue
```bash
# Implement critical fix
git add .
git commit -m "Fix: critical security vulnerability"
git push -u origin hotfix/security-vulnerability
```

#### Step 3: Fast-Track Review
- Create PR to `main`
- Get expedited review
- Merge immediately after approval

#### Step 4: Merge Back to Develop
```bash
# After merging to main, also merge to develop
git checkout develop
git pull origin develop
git merge main
git push origin develop
```

### 4. Release Process

#### Step 1: Create Release Branch
```bash
git checkout develop
git pull origin develop
git checkout -b release/v3.1.0
```

#### Step 2: Prepare Release
```bash
# Update version
python scripts/bump_version.py minor

# Update CHANGELOG.md with release notes
# Test thoroughly
# Fix any last-minute issues

git add .
git commit -m "Prepare release v3.1.0"
git push -u origin release/v3.1.0
```

#### Step 3: Merge to Main
```bash
# Create PR to main
# After approval and merge, tag the release
git checkout main
git pull origin main
git tag -a v3.1.0 -m "Release version 3.1.0"
git push origin v3.1.0
```

#### Step 4: Merge Back to Develop
```bash
git checkout develop
git merge main
git push origin develop
```

## Branch Naming Conventions

### Format
```
<type>/<issue-number>-<short-description>
```

### Examples
- `feature/issue-123-add-lstm-model`
- `feature/issue-124-improve-data-loading`
- `bugfix/issue-456-fix-memory-leak`
- `bugfix/issue-457-correct-calculation`
- `hotfix/security-patch`
- `hotfix/critical-performance-fix`
- `release/v3.1.0`

### Rules
- Use lowercase and hyphens
- Keep descriptions short but descriptive
- Always include issue number when applicable
- Use present tense verbs

## Pull Request Guidelines

### Requirements
1. **All tests must pass** (enforced by CI)
2. **Code quality checks pass** (black, flake8, isort)
3. **At least one reviewer approval**
4. **Branch is up to date** with target branch
5. **Descriptive title and description**

### PR Title Format
```
<Type>: <Brief description>
```

Examples:
- `Add: LSTM model for time series prediction`
- `Fix: memory leak in data processing pipeline`
- `Update: improve documentation for feature engineering`
- `Remove: deprecated utility functions`

### PR Description Checklist
- [ ] Clear description of changes
- [ ] Type of change identified
- [ ] Related issues linked
- [ ] Testing approach described
- [ ] CHANGELOG.md updated (if applicable)
- [ ] Documentation updated (if applicable)

## Code Review Process

### For Reviewers
1. **Check functionality** - Does it work as intended?
2. **Verify tests** - Are there adequate tests?
3. **Review code quality** - Is it readable and maintainable?
4. **Check documentation** - Is it properly documented?
5. **Verify conventions** - Does it follow project standards?

### Review Timeline
- **Features**: 2-3 business days
- **Bug fixes**: 1-2 business days  
- **Hotfixes**: Same day (expedited)
- **Documentation**: 1-2 business days

### Approval Requirements
- **Features**: 1+ approvals from core maintainers
- **Bug fixes**: 1+ approvals
- **Hotfixes**: 1 approval (can be expedited)
- **Documentation**: 1 approval

## Commit Message Guidelines

### Format
```
<Type>: <Subject>

<Body (optional)>

<Footer (optional)>
```

### Types
- **Add**: New features or functionality
- **Fix**: Bug fixes
- **Update**: Improvements to existing features
- **Remove**: Deletion of features or files
- **Docs**: Documentation changes
- **Test**: Test-related changes
- **Refactor**: Code refactoring
- **Style**: Code style changes (formatting, etc.)

### Examples
```
Add: LSTM model for time series prediction

Implements a new LSTM-based model for improved time series
forecasting with configurable layers and dropout.

Closes #123
```

```
Fix: memory leak in data processing pipeline

Resolves issue where large datasets caused memory to grow
unbounded during feature engineering.

Fixes #456
```

## Release Management

### Version Numbering
AlphaPy follows [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0): Breaking changes
- **MINOR** (x.Y.0): New features, backwards compatible
- **PATCH** (x.y.Z): Bug fixes, backwards compatible

### Release Types

#### Major Release (X.0.0)
- Breaking API changes
- Significant architectural changes
- Migration guide required

#### Minor Release (x.Y.0)
- New features
- Enhancements to existing features
- Deprecation notices

#### Patch Release (x.y.Z)
- Bug fixes
- Security patches
- Performance improvements

### Release Schedule
- **Major releases**: Every 6-12 months
- **Minor releases**: Every 1-3 months
- **Patch releases**: As needed for critical fixes

## Automation and Tools

### GitHub Actions
- **Tests**: Run on all PRs and pushes
- **Code Quality**: Enforce style and linting
- **Release**: Automatic PyPI publishing on tags
- **Branch Validation**: Check naming conventions

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Useful Scripts
```bash
# Bump version
python scripts/bump_version.py [major|minor|patch]

# Create feature branch
./scripts/create-branch.sh feature "Add new ML algorithm"

# Check workflow status
gh pr status
```

## Best Practices

### General
1. **Keep branches focused** - One feature/fix per branch
2. **Write descriptive commits** - Explain the "why" not just "what"
3. **Test thoroughly** - Add tests for new functionality
4. **Update documentation** - Keep docs current with changes
5. **Review your own PR** - Check it once before requesting review

### Performance
1. **Squash related commits** - Clean up commit history
2. **Rebase vs merge** - Use rebase for cleaner history
3. **Delete merged branches** - Keep repository clean

### Communication
1. **Link issues in PRs** - Use "Fixes #123" or "Closes #456"
2. **Tag reviewers** - Request specific reviewers when needed
3. **Respond to feedback** - Address review comments promptly
4. **Update progress** - Comment on issues with status updates

## Troubleshooting

### Common Issues

#### Branch Behind Target
```bash
git checkout feature/my-branch
git rebase develop
git push --force-with-lease
```

#### Merge Conflicts
```bash
git checkout develop
git pull origin develop
git checkout feature/my-branch
git rebase develop
# Resolve conflicts
git add .
git rebase --continue
git push --force-with-lease
```

#### Failed CI Checks
```bash
# Run checks locally
pytest tests/
black --check .
flake8 .
isort --check-only .

# Fix issues and commit
```

#### Wrong Branch Name
```bash
# Rename local branch
git branch -m old-name new-name

# Delete old remote branch and push new one
git push origin --delete old-name
git push -u origin new-name
```

### Getting Help
1. **Check documentation** - Read relevant docs first
2. **Search issues** - Look for similar problems
3. **Ask in discussions** - Use GitHub Discussions for questions
4. **Contact maintainers** - Tag @maintainers for urgent issues

## Quick Reference

### Common Commands
```bash
# Setup
git clone https://github.com/ScottFreeLLC/AlphaPy.git
cd AlphaPy
pip install -e .[dev]
pre-commit install

# Start new feature
git checkout develop
git pull origin develop
git checkout -b feature/issue-123-description

# Update branch
git checkout develop
git pull origin develop
git checkout feature/my-branch
git rebase develop

# Clean up
git branch -d feature/merged-branch
git remote prune origin
```

### Workflow Checklist
- [ ] Issue created and assigned
- [ ] Branch follows naming convention
- [ ] Code follows style guidelines
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if needed)
- [ ] PR created with complete description
- [ ] CI checks passing
- [ ] Code reviewed and approved
- [ ] Branch merged and deleted

This workflow ensures high code quality, proper review processes, and smooth releases while maintaining development velocity.