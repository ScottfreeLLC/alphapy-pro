#!/usr/bin/env python3
"""
Workflow validation script for AlphaPy development.

This script checks various aspects of the development workflow:
- Branch naming conventions
- Git status and cleanliness
- Required files and structure
- Code quality checks
"""

import subprocess
import sys
import re
import os
from pathlib import Path


class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def print_color(color, message):
    """Print colored message."""
    print(f"{color}{message}{Colors.NC}")


def run_command(cmd, capture_output=True):
    """Run shell command and return result."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=capture_output, text=True, check=True
        )
        return result.stdout.strip() if capture_output else True
    except subprocess.CalledProcessError as e:
        if capture_output:
            return None
        return False


def check_git_repository():
    """Check if we're in a git repository."""
    if not run_command("git rev-parse --git-dir"):
        print_color(Colors.RED, "❌ Not in a git repository")
        return False
    print_color(Colors.GREEN, "✅ In git repository")
    return True


def check_branch_name():
    """Check if current branch follows naming convention."""
    branch = run_command("git branch --show-current")
    if not branch:
        print_color(Colors.RED, "❌ Could not determine current branch")
        return False
    
    print_color(Colors.BLUE, f"Current branch: {branch}")
    
    # Skip check for main and develop branches
    if branch in ['main', 'develop']:
        print_color(Colors.GREEN, "✅ Main/develop branch")
        return True
    
    # Check naming convention
    pattern = r'^(feature|bugfix|hotfix|release)/(issue-\d+-)?[a-z0-9-]+$'
    if re.match(pattern, branch):
        print_color(Colors.GREEN, "✅ Branch name follows convention")
        return True
    else:
        print_color(Colors.RED, "❌ Branch name does not follow convention")
        print_color(Colors.YELLOW, "Expected: feature/issue-123-description or bugfix/issue-456-description")
        return False


def check_git_status():
    """Check git working directory status."""
    status = run_command("git status --porcelain")
    if not status:
        print_color(Colors.GREEN, "✅ Working directory clean")
        return True
    else:
        print_color(Colors.YELLOW, "⚠️  Working directory has changes:")
        run_command("git status --short", capture_output=False)
        return True  # Not a failure, just informational


def check_required_files():
    """Check if required files exist."""
    required_files = [
        'pyproject.toml',
        'README.md',
        'CHANGELOG.md',
        'CONTRIBUTING.md',
        'LICENSE',
        'alphapy/__init__.py',
        'tests/__init__.py',
        '.github/workflows/tests.yml',
        '.github/pull_request_template.md',
        '.pre-commit-config.yaml',
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print_color(Colors.RED, "❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    else:
        print_color(Colors.GREEN, "✅ All required files present")
        return True


def check_code_quality():
    """Run basic code quality checks."""
    checks = [
        ("black --check --diff .", "Black formatting"),
        ("isort --check-only --diff .", "Import sorting"),
        ("flake8 . --count --select=E9,F63,F7,F82", "Critical linting errors"),
    ]
    
    all_passed = True
    for cmd, description in checks:
        print_color(Colors.BLUE, f"Checking {description}...")
        if run_command(cmd):
            print_color(Colors.GREEN, f"✅ {description} passed")
        else:
            print_color(Colors.RED, f"❌ {description} failed")
            all_passed = False
    
    return all_passed


def check_tests():
    """Run basic test checks."""
    print_color(Colors.BLUE, "Running basic tests...")
    if run_command("python -m pytest tests/test_version.py -v"):
        print_color(Colors.GREEN, "✅ Basic tests passed")
        return True
    else:
        print_color(Colors.RED, "❌ Basic tests failed")
        return False


def check_version_consistency():
    """Check if version is consistent across files."""
    try:
        # Get version from package
        import alphapy
        package_version = alphapy.__version__
        
        # Check pyproject.toml has dynamic version
        with open('pyproject.toml', 'r') as f:
            content = f.read()
            if 'dynamic = ["version"]' in content:
                print_color(Colors.GREEN, f"✅ Version management configured (v{package_version})")
                return True
            else:
                print_color(Colors.YELLOW, "⚠️  Version not configured as dynamic in pyproject.toml")
                return False
    except Exception as e:
        print_color(Colors.RED, f"❌ Could not check version: {e}")
        return False


def main():
    """Main workflow check function."""
    print_color(Colors.BLUE, "🔍 AlphaPy Workflow Checker")
    print("=" * 50)
    
    checks = [
        ("Git Repository", check_git_repository),
        ("Branch Name", check_branch_name), 
        ("Git Status", check_git_status),
        ("Required Files", check_required_files),
        ("Version Consistency", check_version_consistency),
        ("Code Quality", check_code_quality),
        ("Basic Tests", check_tests),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n📋 {name}")
        print("-" * 30)
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print_color(Colors.RED, f"❌ Error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print_color(Colors.BLUE, "📊 Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:20} {status}")
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print_color(Colors.GREEN, "\n🎉 All checks passed! Ready for development.")
        return 0
    else:
        print_color(Colors.YELLOW, f"\n⚠️  {total - passed} checks failed. Please address issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())