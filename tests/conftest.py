"""
tests/conftest.py
Override pytest's tmp_path base directory to avoid Windows permission errors
when the system TEMP folder belongs to a different user account (e.g., Harshitha).

This machine runs as 'vikas' but the system TEMP is under 'Harshitha', causing
OSError when pytest tries to create numbered dirs there.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

# Redirect all tmp_path usage to a project-local folder
_BASE_TMP = Path(__file__).resolve().parent.parent / ".pytest_tmp"
_BASE_TMP.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def tmp_path(request, tmp_path_factory):
    """
    Override pytest's built-in tmp_path to use a project-local base directory
    instead of the system temp, which may belong to a different Windows user.
    """
    # Create a unique per-test subdirectory under our local base
    test_name = request.node.name
    # Sanitize the name for filesystem use
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in test_name)[:50]
    test_dir = _BASE_TMP / safe_name
    # Remove if leftover from previous run
    if test_dir.exists():
        shutil.rmtree(test_dir, ignore_errors=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    yield test_dir
    # Cleanup after test
    shutil.rmtree(test_dir, ignore_errors=True)
