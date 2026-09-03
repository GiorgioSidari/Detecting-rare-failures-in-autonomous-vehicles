"""
Import paths for the test suite.

The repository is not installed as a package when the tests are run from a
checkout, and `lanekeeping` is not inside one at all: it is imported with
`opensbt-core/Simulator/` on the path. pytest imports this file before
collecting any test, so the four roots below are on `sys.path` for every test
module.

`tests/test_lane_keeping.py` and `tests/test_pipeline_lane_keeping.py` are also
runnable as plain scripts and keep their own bootstrap for that case.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for _p in (
    _REPO_ROOT,
    os.path.join(_REPO_ROOT, "opensbt-core"),
    os.path.join(_REPO_ROOT, "opensbt-core", "Simulator"),
    os.path.join(_REPO_ROOT, "scripts"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)


import pytest


@pytest.fixture(autouse=True)
def _close_metadrive_engine():
    """
    Close MetaDrive's global engine after each test.

    MetaDrive keeps one engine per process and refuses to build a new
    environment while it is up ("Can not call this API after engine
    initialization!"), so a test that leaves an environment open makes every
    later MetaDrive test fail. The fixture is a no-op when metadrive is not
    installed or no engine was started.
    """
    yield
    try:
        from metadrive.engine.engine_utils import close_engine, engine_initialized
    except Exception:
        return
    if engine_initialized():
        close_engine()
