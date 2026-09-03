"""
Tests on the compose file that starts the state-based arm's containers, and on
`gen_c2_compose.py` which generates it.

The compose file sets `command:` to launch `Simulator.c2.server`. A `command:`
replaces the Dockerfile CMD entirely, and that CMD also starts Xvfb and exports
`DISPLAY=:99`, which Unity requires even headless: without it the container
comes up, uvicorn answers, and the Unity process waits indefinitely for a GL
context without raising or exiting.

The tests parse the compose file and the Dockerfile and check that the declared
command starts Xvfb, exports `DISPLAY` without letting the host interpolate it,
points at the state-based server, keeps the whole bootstrap of the original CMD,
and that re-running the generator reproduces the committed file. No container is
started.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

yaml = pytest.importorskip("yaml")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OPENSBT = os.path.join(_REPO_ROOT, "opensbt-core")
_DOCKERFILE = os.path.join(_OPENSBT, "Simulator", "Dockerfile")

_COMPOSE_C2 = [
    (os.path.join(_OPENSBT, "docker-compose.c2.yml"), "simulator"),
    (os.path.join(_OPENSBT, "docker-compose.c2.parallel.yml"), "simulator-0"),
]


def _command(path: str, service: str) -> str:
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    cmd = doc["services"][service]["command"]
    # forma lista (exec) o stringa (shell): normalizzo per i controlli testuali
    return " ".join(cmd) if isinstance(cmd, list) else str(cmd)


def _cmd_dockerfile() -> str:
    """Extracts the CMD from the Dockerfile, stitching line continuations."""
    with open(_DOCKERFILE, encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"^CMD (.*?)(?=\n(?![ \t])\S|\Z)", text, re.S | re.M)
    assert m, "CMD not found in the Dockerfile: the test needs updating"
    return re.sub(r"\s*\\\s*\n\s*", " ", m.group(1)).strip()


@pytest.mark.parametrize("path,service", _COMPOSE_C2,
                         ids=[os.path.basename(p) for p, _ in _COMPOSE_C2])
def test_command_starts_xvfb(path, service):
    """
    Without those two things the container hangs on "repeat to connect".

    This is the bug this file exists to prevent: replacing the CMD while keeping
    only the uvicorn line looks correct and is not.
    """
    cmd = _command(path, service)
    assert "Xvfb :99" in cmd, (
        f"{os.path.basename(path)} does not start Xvfb.\n"
        f"`command:` overrides the Dockerfile CMD, so the display bootstrap must "
        f"be replicated: without it Unity gets no GL context and the container "
        f"hangs on 'sleep...and repeat to connect'.")
    assert "DISPLAY=:99" in cmd, (
        f"{os.path.basename(path)}: Xvfb starts but DISPLAY is not exported, "
        f"so Unity does not know which display to attach to.")


@pytest.mark.parametrize("path,service", _COMPOSE_C2,
                         ids=[os.path.basename(p) for p, _ in _COMPOSE_C2])
def test_command_points_at_the_c2_server(path, service):
    """L'unica differenza voluta rispetto al CMD originale."""
    cmd = _command(path, service)
    assert "Simulator.c2.server:app" in cmd
    assert "SimulatorServer" not in cmd, (
        "this compose would launch the C1 arm (the DNN), not C2")


@pytest.mark.parametrize("path,service", _COMPOSE_C2,
                         ids=[os.path.basename(p) for p, _ in _COMPOSE_C2])
def test_xvfb_is_not_interpolated_by_the_host(path, service):
    """
    `$${VAR}` and not `${VAR}`.

    Docker Compose substitutes variables on the HOST, before handing the command
    to the container. With a single `$`, `${XVFB_RESOLUTION}` would be resolved
    against the host environment -- where it is not defined -- and the
    resolution would arrive empty or from the compose default, ignoring the
    Dockerfile ENV. `$$` escapes the substitution and lets the container's shell
    expand it.
    """
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    if "XVFB_RESOLUTION" not in raw:
        pytest.skip("resolution hardcoded: no interpolation to protect")
    assert "$${XVFB_RESOLUTION" in raw, (
        "use $${XVFB_RESOLUTION:-...}: with a single $ the variable is expanded "
        "by Docker on the host instead of by the shell in the container")


@pytest.mark.parametrize("fragment", [
    "rm -f /tmp/.X99-lock",   # riavvio: "Server is already active for display 99"
    "-ac +extension GLX",     # Unity forces graphical mode: GLX is required
    "sleep 3",                # Unity may start before the display is ready
])
def test_bootstrap_is_complete(fragment):
    """
    Every piece of the Dockerfile bootstrap is present in C2 too.

    They are all there for a reason documented in the Dockerfile; omitting one
    produces an intermittent failure, which is worse than a hard one.
    """
    for path, service in _COMPOSE_C2:
        cmd = _command(path, service)
        assert fragment in cmd, f"{os.path.basename(path)} does not contain {fragment!r}"


def test_stays_aligned_with_the_dockerfile():
    """
    The C2 command and the Dockerfile CMD must differ ONLY in the module.

    This test is the alarm for the future: if someone changes the bootstrap in
    the Dockerfile (pre-existing, and untouched by us), the C2 compose files stay
    indietro in silenzio e i container tornano a bloccarsi. Meglio un test
    a red test than a hang.
    """
    expected = _cmd_dockerfile().replace("Simulator.SimulatorServer:app",
                                       "Simulator.c2.server:app")
    # normalise differences of form, not of substance
    def _norm(s: str) -> str:
        s = s.replace("$${", "${")                      # escape del compose
        s = re.sub(r"\$\{XVFB_RESOLUTION(:-[^}]*)?\}", "${XVFB_RESOLUTION}", s)
        s = re.sub(r"^/bin/s?h -c [\"']|[\"']$", "", s.strip())
        return re.sub(r"\s+", " ", s).strip()

    for path, service in _COMPOSE_C2:
        assert _norm(_command(path, service)) == _norm(expected), (
            f"{os.path.basename(path)} is no longer aligned with the Dockerfile CMD.\n"
            f"  compose   : {_norm(_command(path, service))}\n"
            f"  dockerfile: {_norm(expected)}\n"
            f"If the Dockerfile changed, carry the change over into "
            f"gen_c2_compose.py e rigenerare.")


def test_generator_reproduces_the_committed_file(tmp_path):
    """
    `docker-compose.c2.parallel.yml` è generated: se qualcuno lo modifica a mano,
    the change disappears at the next regeneration. This test catches that.

    The check output goes to `tmp_path` and not next to the original: writing
    into the working copy would leave stray files if the test failed halfway.
    """
    path = os.path.join(_OPENSBT, "docker-compose.c2.parallel.yml")
    doc = yaml.safe_load(open(path, encoding="utf-8"))
    n = len(doc["services"])
    speed = next(v.split("=", 1)[1] for v in doc["services"]["simulator-0"]["environment"]
                 if v.startswith("LK_SPEED_SCALE="))

    generated = tmp_path / "check.yml"
    out = subprocess.run(
        [sys.executable, "gen_c2_compose.py", str(n), "--speed-scale", speed,
         "--out", str(generated)],
        cwd=_OPENSBT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr

    # the header quotes the output path: realign it to the canonical name
    expected = generated.read_text(encoding="utf-8").replace(
        str(generated), "docker-compose.c2.parallel.yml")
    assert open(path, encoding="utf-8").read() == expected, (
        "docker-compose.c2.parallel.yml does not match the generator output. "
        "Regenerate instead of editing it by hand:\n"
        f"  cd opensbt-core\n"
        f"  python gen_c2_compose.py {n} --speed-scale {speed}")
