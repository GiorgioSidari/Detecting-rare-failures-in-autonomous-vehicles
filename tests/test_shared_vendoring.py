"""
Checks the vendored copies of the shared components.

`opensbt-core/Simulator/shared/` holds copies of the files in
`scenarios/common/`, because the containers' build context is `opensbt-core/`
and Docker cannot copy from outside it.

The tests compare the file HASHES of each pair, so any difference fails,
including a comment or a blank line. A failure means one of the two sides was
edited: copy the file from `scenarios/common/` to
`opensbt-core/Simulator/shared/` again.
"""
from __future__ import annotations

import hashlib
import os

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SOURCE = os.path.join(_REPO_ROOT, "scenarios", "common")
_VENDORED = os.path.join(_REPO_ROOT, "opensbt-core", "Simulator", "shared")

MODULI = ["driver.py", "episode_budget.py", "road_frame.py", "road_geometry.py"]

# Every module of `scenarios/common/` is vendored: no exception, no bridge. An
# exemption is one more rule to remember, and the first one forgotten is the one
# that breaks the comparison silently.


def _sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@pytest.mark.parametrize("name", MODULI)
def test_copy_is_identical(name):
    src = os.path.join(_SOURCE, name)
    dst = os.path.join(_VENDORED, name)

    assert os.path.isfile(src), f"the original {src} is missing"
    assert os.path.isfile(dst), (
        f"the copy {dst} is missing. Regenerate it with:\n"
        f"  cp scenarios/common/{name} opensbt-core/Simulator/shared/{name}"
    )
    assert _sha(src) == _sha(dst), (
        f"{name} diverges between scenarios/common/ and Simulator/shared/.\n"
        f"The container would use a different controller from the comparison.\n"
        f"Regenerate the copy:\n"
        f"  cp scenarios/common/{name} opensbt-core/Simulator/shared/{name}"
    )


def test_no_module_is_forgotten():
    """
    If a module is added to `scenarios/common/`, it must end up in the list
    above too -- otherwise the copy in the container falls behind silently.
    """
    present = {f for f in os.listdir(_SOURCE)
                if f.endswith(".py") and f != "__init__.py"}
    uncovered = present - set(MODULI)
    assert not uncovered, (
        f"modules in scenarios/common/ not covered by the test: {uncovered}.\n"
        f"Ogni modulo di scenarios/common/ va vendorizzato. Aggiungerli a\n"
        f"MODULI e copiarli:\n"
        f"  cp scenarios/common/<modulo> opensbt-core/Simulator/shared/"
    )


def test_copy_is_importable_on_its_own():
    """
    The copy must work without the rest of the repo: inside the container only
    the `Simulator/` tree exists, not `scenarios/`.
    """
    import importlib.util
    import sys

    for name in MODULI:
        mod_name = f"_vendored_probe_{name[:-3]}"
        spec = importlib.util.spec_from_file_location(
            mod_name, os.path.join(_VENDORED, name))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)          # solleva se ha import del repo
        finally:
            sys.modules.pop(mod_name, None)


def test_behaviour_is_identical():
    """
    Counter-check on the hash: given the same inputs the two copies must produce
    the same sequence of actions. Redundant while the hashes agree, useful if the
    criterion is ever loosened.
    """
    import importlib.util
    import sys

    import numpy as np

    from scenarios.common.driver import LateralFeedbackDriver as Originale

    spec = importlib.util.spec_from_file_location(
        "_vendored_driver_probe", os.path.join(_VENDORED, "driver.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_vendored_driver_probe"] = mod
    try:
        spec.loader.exec_module(mod)
        Copia = mod.LateralFeedbackDriver

        a, b = Originale(), Copia()
        a.reset(); b.reset()
        rng = np.random.default_rng(0)
        for _ in range(200):
            st = {
                "lateral_error": float(rng.uniform(-3, 3)),
                "heading_error": float(rng.uniform(-0.6, 0.6)),
                "speed": float(rng.uniform(0, 30)),
                "target_speed": 12.0,
            }
            assert a.act(dict(st)) == b.act(dict(st))
    finally:
        sys.modules.pop("_vendored_driver_probe", None)
