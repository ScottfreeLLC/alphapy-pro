#!/usr/bin/env python3
"""
Version bumping script for AlphaPy.

Usage:
    python scripts/bump_version.py [major|minor|patch]

This script:
1. Updates the version in alphapy/__init__.py
2. Creates a git tag for the new version
3. Updates the CHANGELOG.md with the new version
"""

import re
import sys
import os
from pathlib import Path
from datetime import datetime


def get_current_version():
    """Get the current version from the package."""
    init_path = Path("alphapy") / "__init__.py"
    with open(init_path, 'r') as f:
        content = f.read()
    
    match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find version string.")


def bump_version(current_version, bump_type):
    """Bump the version based on the bump type."""
    major, minor, patch = map(int, current_version.split('.'))
    
    if bump_type == 'major':
        major += 1
        minor = 0
        patch = 0
    elif bump_type == 'minor':
        minor += 1
        patch = 0
    elif bump_type == 'patch':
        patch += 1
    else:
        raise ValueError("bump_type must be 'major', 'minor', or 'patch'")
    
    return f"{major}.{minor}.{patch}"


def update_version_in_file(new_version):
    """Update the version in alphapy/__init__.py."""
    init_path = Path("alphapy") / "__init__.py"
    with open(init_path, 'r') as f:
        content = f.read()
    
    new_content = re.sub(
        r"^__version__ = ['\"][^'\"]*['\"]",
        f'__version__ = "{new_version}"',
        content,
        flags=re.M
    )
    
    with open(init_path, 'w') as f:
        f.write(new_content)


def update_changelog(new_version):
    """Update the CHANGELOG.md with the new version."""
    changelog_path = Path("CHANGELOG.md")
    if not changelog_path.exists():
        print("CHANGELOG.md not found, skipping changelog update")
        return
    
    with open(changelog_path, 'r') as f:
        content = f.read()
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Replace [Unreleased] with the new version
    new_content = content.replace(
        "## [Unreleased]",
        f"## [Unreleased]\n\n## [{new_version}] - {today}"
    )
    
    # Update the links at the bottom
    new_content = re.sub(
        r"\[Unreleased\]: https://github\.com/ScottFreeLLC/AlphaPy/compare/v[\d.]+\.\.\.HEAD",
        f"[Unreleased]: https://github.com/ScottFreeLLC/AlphaPy/compare/v{new_version}...HEAD",
        new_content
    )
    
    # Add the new version link
    new_content = new_content.replace(
        f"[{new_version}]: https://github.com/ScottFreeLLC/AlphaPy/releases/tag/v{new_version}",
        f"[{new_version}]: https://github.com/ScottFreeLLC/AlphaPy/releases/tag/v{new_version}"
    )
    
    with open(changelog_path, 'w') as f:
        f.write(new_content)


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/bump_version.py [major|minor|patch]")
        sys.exit(1)
    
    bump_type = sys.argv[1]
    if bump_type not in ['major', 'minor', 'patch']:
        print("bump_type must be 'major', 'minor', or 'patch'")
        sys.exit(1)
    
    # Get current version
    current_version = get_current_version()
    print(f"Current version: {current_version}")
    
    # Calculate new version
    new_version = bump_version(current_version, bump_type)
    print(f"New version: {new_version}")
    
    # Update files
    update_version_in_file(new_version)
    update_changelog(new_version)
    
    print(f"Version bumped from {current_version} to {new_version}")
    print(f"Don't forget to:")
    print(f"1. Review the changes")
    print(f"2. Commit the changes: git commit -am 'Bump version to {new_version}'")
    print(f"3. Create a tag: git tag -a v{new_version} -m 'Release version {new_version}'")
    print(f"4. Push changes and tags: git push origin main --tags")


if __name__ == "__main__":
    main()