import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip tests marked `live` unless both provider keys are present."""
    have_keys = bool(os.environ.get("RIME_API_KEY") and os.environ.get("OPENROUTER_API_KEY"))
    if have_keys:
        return
    skip = pytest.mark.skip(reason="live test: set RIME_API_KEY and OPENROUTER_API_KEY")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
