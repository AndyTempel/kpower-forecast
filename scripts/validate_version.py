import re
import sys
from pathlib import Path

# Regex for YYYY.MM.Patch (e.g., 2026.2.1)
VERSION_PATTERN = r"^\d{4}\.(?:[1-9]|1[0-2])\.\d+$"


def validate_version(file_path: str, pattern: str):
    content = Path(file_path).read_text()
    # Find version in pyproject.toml or __init__.py
    match = re.search(pattern, content, re.MULTILINE)
    if not match:
        print(f"❌ Could not find version in {file_path}")
        return False

    version = match.group(1)
    if not re.match(VERSION_PATTERN, version):
        print(f"❌ Invalid version format in {file_path}: '{version}'")
        print("   Expected format: YYYY.MM.Patch (e.g., 2026.2.1)")
        return False

    print(f"✅ Version {version} in {file_path} is valid.")
    return True


if __name__ == "__main__":
    success = True
    # Check pyproject.toml
    if not validate_version("pyproject.toml", r'^version\s*=\s*"([^"]+)"'):
        success = False

    # Check __init__.py
    pattern = r'^__version__\s*=\s*"([^"]+)"'
    if not validate_version("src/kpower_forecast/__init__.py", pattern):
        success = False

    if not success:
        sys.exit(1)
