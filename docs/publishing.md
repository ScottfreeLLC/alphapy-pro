# Publishing to PyPI

This document explains how AlphaPy is published to PyPI using modern packaging and trusted publishing.

## Overview

AlphaPy uses:
- Modern Python packaging with `pyproject.toml`
- GitHub Actions for automated CI/CD
- PyPA's trusted publishing for secure PyPI releases
- Semantic versioning with git tags

## Publishing Process

### 1. Prepare a Release

Update the version and changelog:

```bash
# Bump version (major, minor, or patch)
python scripts/bump_version.py minor

# Review and edit CHANGELOG.md if needed
# Commit changes
git commit -am "Bump version to X.Y.Z"
```

### 2. Create and Push a Tag

```bash
# Create an annotated tag
git tag -a vX.Y.Z -m "Release version X.Y.Z"

# Push the tag to GitHub
git push origin main --tags
```

### 3. Automated Publishing

When you push a tag starting with `v`, GitHub Actions will:

1. **Run tests** across Python 3.9, 3.10, and 3.11
2. **Build the package** using `python -m build`
3. **Validate the package** with `twine check`
4. **Publish to PyPI** using trusted publishing

## Trusted Publishing Setup

### Initial Setup (One-time)

1. **Create PyPI Project**: Go to [PyPI](https://pypi.org) and create the project
2. **Configure Trusted Publishing**:
   - Go to project settings on PyPI
   - Add GitHub as a trusted publisher
   - Repository: `ScottFreeLLC/AlphaPy`
   - Workflow: `release.yml`
   - Environment: `release`

### GitHub Environment Setup

Create a `release` environment in GitHub repository settings:
1. Go to Settings → Environments
2. Create new environment named `release`
3. Add protection rules (require reviewers, etc.)

## Manual Publishing (Fallback)

If needed, you can publish manually:

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# Check the build
twine check dist/*

# Upload to PyPI (requires API token)
twine upload dist/*
```

## Package Structure

The built package includes:
- Source distribution (`.tar.gz`)
- Wheel distribution (`.whl`)
- All necessary metadata from `pyproject.toml`

## Validation

Before publishing, the workflow validates:
- Package builds successfully
- All tests pass
- Package metadata is correct
- No syntax errors in the distribution

## Security

- **No API tokens** are stored in the repository
- **Trusted publishing** uses OpenID Connect for secure authentication
- **Environment protection** requires manual approval for releases
- **Sensitive data** is excluded via `.gitignore`

## Troubleshooting

### Build Failures
- Check Python version compatibility
- Verify all dependencies are properly specified
- Ensure tests pass locally

### Publishing Failures
- Verify trusted publishing is configured correctly
- Check GitHub environment settings
- Ensure version number hasn't been used before

### Version Conflicts
- PyPI doesn't allow re-uploading the same version
- Use `python scripts/bump_version.py patch` to increment
- Delete and recreate the git tag if needed:
  ```bash
  git tag -d vX.Y.Z
  git push origin :refs/tags/vX.Y.Z
  git tag -a vX.Y.Z -m "Release version X.Y.Z"
  git push origin vX.Y.Z
  ```

## Best Practices

1. **Test thoroughly** before tagging
2. **Use semantic versioning** consistently
3. **Update CHANGELOG.md** for each release
4. **Run tests locally** before pushing
5. **Monitor the GitHub Actions** workflow for issues
6. **Verify the package** on PyPI after publishing

## References

- [PyPA Trusted Publishing](https://docs.pypa.io/en/latest/trusted-publishers/)
- [Python Packaging Guide](https://packaging.python.org/)
- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)