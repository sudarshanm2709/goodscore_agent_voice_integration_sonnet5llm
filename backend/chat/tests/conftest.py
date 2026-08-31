import sys
from pathlib import Path

# Make backend/*.py importable when pytest is run from backend/ (`pytest`
# or `pytest tests/`) without installing the project as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
