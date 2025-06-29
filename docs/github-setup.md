# GitHub Repository Setup Guide

This guide will help you configure GitHub settings for the AlphaPy Pro project with proper branch protection and workflow management.

## Branch Protection Rules

### For `main` branch:

1. Go to **Settings → Branches** in your GitHub repository
2. Click **Add rule** or edit existing rule for `main`
3. Configure the following settings:

#### Required Settings:
- ✅ **Require a pull request before merging**
  - ✅ Require approvals: `1`
  - ✅ Dismiss stale PR approvals when new commits are pushed
  - ✅ Require review from code owners (if you add a CODEOWNERS file)

- ✅ **Require status checks to pass before merging**
  - ✅ Require branches to be up to date before merging
  - Select these required checks:
    - `test (3.9)` - Python 3.9 tests
    - `test (3.10)` - Python 3.10 tests  
    - `test (3.11)` - Python 3.11 tests
    - `pre-commit` - Code quality checks

- ✅ **Require conversation resolution before merging**
- ✅ **Restrict pushes that create files over 100MB**

#### Administrative Settings:
- ✅ **Include administrators** (enforces rules for everyone)
- ✅ **Allow force pushes** → **Everyone** (UNCHECK this for security)
- ✅ **Allow deletions** (UNCHECK this for security)

### For `develop` branch:

1. Create another rule for `develop` branch
2. Configure these settings:

#### Required Settings:
- ✅ **Require a pull request before merging**
  - ✅ Require approvals: `1`
  - ✅ Dismiss stale PR approvals when new commits are pushed

- ✅ **Require status checks to pass before merging**
  - ✅ Require branches to be up to date before merging
  - Select the same required checks as main branch

- ✅ **Require conversation resolution before merging**

## Issue Labels

Create these labels in **Issues → Labels**:

### Type Labels:
- `type: bug` - 🐛 Something isn't working (color: #d73a4a)
- `type: enhancement` - ✨ New feature request (color: #a2eeef)
- `type: documentation` - 📚 Documentation improvements (color: #0075ca)
- `type: question` - ❓ Further information is requested (color: #d876e3)
- `type: refactor` - 🔨 Code refactoring (color: #f9d71c)

### Priority Labels:
- `priority: critical` - 🚨 Critical priority (color: #b60205)
- `priority: high` - 🔥 High priority (color: #d93f0b)
- `priority: medium` - ⚡ Medium priority (color: #fbca04)
- `priority: low` - 🐌 Low priority (color: #0e8a16)

### Status Labels:
- `status: needs-triage` - 🕵️ Needs review (color: #ededed)
- `status: in-progress` - 🚧 Being worked on (color: #f9d71c)
- `status: blocked` - 🚫 Blocked by something (color: #e99695)
- `status: ready-for-review` - 👀 Ready for review (color: #c2e0c6)

### Special Labels:
- `good first issue` - 👋 Good for newcomers (color: #7057ff)
- `help wanted` - 🙏 Extra attention is needed (color: #008672)
- `breaking change` - ⚠️ Breaking change (color: #b60205)

## Repository Settings

### General Settings:
1. Go to **Settings → General**
2. **Features section**:
   - ✅ Wikis (if you want community documentation)
   - ✅ Issues
   - ✅ Sponsorships (if you want GitHub Sponsors)
   - ✅ Preserve this repository (for backup)
   - ✅ Discussions (for community Q&A)

3. **Pull Requests section**:
   - ✅ Allow merge commits
   - ✅ Allow squash merging (recommended)
   - ✅ Allow rebase merging
   - ✅ Always suggest updating pull request branches
   - ✅ Allow auto-merge
   - ✅ Automatically delete head branches

### Security Settings:
1. Go to **Settings → Security & analysis**
2. Enable:
   - ✅ Dependency graph
   - ✅ Dependabot alerts
   - ✅ Dependabot security updates
   - ✅ Secret scanning (if available)

## Collaborators and Teams

### Repository Roles:
- **Admin**: Full access (repository owners)
- **Maintain**: Manage repository without access to sensitive actions
- **Write**: Push access to repository
- **Triage**: Manage issues and pull requests
- **Read**: View and clone repository

### Recommended Setup:
1. Core maintainers: **Admin** role
2. Regular contributors: **Write** role
3. Community helpers: **Triage** role

## Environment Setup (for PyPI Publishing)

1. Go to **Settings → Environments**
2. Create new environment: `release`
3. Configure protection rules:
   - ✅ Required reviewers: Add core maintainers
   - ✅ Wait timer: 5 minutes (optional safety delay)

## Automated Actions Setup

The GitHub Actions workflows are already configured to:
- Run tests on Python 3.9, 3.10, 3.11
- Check code quality with pre-commit hooks
- Publish to PyPI automatically on version tags
- Run security scans

No additional configuration needed for Actions unless you want to modify the workflows.

## Quick Setup Checklist

- [ ] Set up branch protection for `main`
- [ ] Set up branch protection for `develop`  
- [ ] Create issue labels
- [ ] Configure repository settings
- [ ] Set up `release` environment
- [ ] Add collaborators with appropriate roles
- [ ] Enable security features
- [ ] Test the workflow by creating a test PR

## After Setup

Once configured, the workflow will be:
1. Developers create feature branches from `develop`
2. Submit PRs to merge back to `develop`
3. Regular releases merge `develop` → `main` with version tags
4. Hotfixes can go directly to `main` with immediate release

This ensures code quality, proper review, and controlled releases while maintaining development velocity.