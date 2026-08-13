"""
Copy of the shared components, reachable from inside the containers.

`driver.py`, `episode_budget.py`, `road_frame.py` and `road_geometry.py` are
**byte-identical copies** of `scenarios/common/`. They are not a fork:
`tests/test_shared_vendoring.py` compares the file hashes and fails if they
diverge by a single character.

Perché la copia esiste
----------------------
The containers' build context is `opensbt-core/`, so a `COPY ./scenarios` in
the Dockerfile is impossible: Docker cannot read outside the context. Moving the
context to the repo root would also ship `.venv`, making the
build inutilizzabile.

The alternative would be to let every container write its own controller. That
is exactly what had happened -- one backend's driver had a lateral controller with
`k_head=1.0` and `k_throttle=0.5` against the shared 0.8 and 0.3 -- and it would
have made the C2 arm's assumption ("the same controller on different
simulators") false with nothing to flag it. Better one copy watched by a test
than several free implementations.

Per modificare la legge di controllo: si tocca `scenarios/common/`, poi si
ricopia. Il test dice subito se ci si è dimenticati.
"""
