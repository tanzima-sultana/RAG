import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import config  # noqa: F401
except ImportError:
    pytest.exit(
        "config.py not found at repo root. Copy config.py.template to config.py "
        "and fill in the required values before running the test suite.",
        returncode=1,
    )


@pytest.fixture()
def isolated_cwd(tmp_path, monkeypatch):
    """Run a test inside an empty temp directory so pipeline code that writes to
    relative paths (chunks/, embeddings/, index/, manifests/, ...) never touches
    the real repo directories."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture()
def sample_docs():
    return [
        {
            "doc_id": "d1",
            "title": "Solar System",
            "text": (
                "The Sun is the star at the center of the Solar System. "
                "Earth orbits the Sun once every year. "
                "Mars is the fourth planet from the Sun."
            ),
        },
        {
            "doc_id": "d2",
            "title": "Oceans",
            "text": (
                "The Pacific Ocean is the largest ocean on Earth. "
                "It covers more area than all the land on the planet combined. "
                "Many species live in the deep Pacific."
            ),
        },
    ]
