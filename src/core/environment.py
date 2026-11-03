"""Make pip-installed NVIDIA runtime libraries discoverable on Linux."""
import os
from pathlib import Path
import site
import sys


def setup():
    """Restart Python if installed NVIDIA library directories need adding to its search path."""
    if sys.platform != 'linux':
        return
    roots = [*site.getsitepackages(), site.getusersitepackages()]
    directories = []
    for root in roots:
        for package in ('cublas', 'cudnn'):
            directory = Path(root) / 'nvidia' / package / 'lib'
            if directory.is_dir() and str(directory) not in directories:
                directories.append(str(directory))
    existing = os.environ.get('LD_LIBRARY_PATH', '')
    existing_paths = existing.split(':') if existing else []
    missing = [directory for directory in directories if directory not in existing_paths]
    if not missing:
        return
    environment = os.environ.copy()
    environment['LD_LIBRARY_PATH'] = ':'.join([*missing, *existing_paths])
    # Preserve interpreter flags and module invocation (python -m src).
    os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], environment)
