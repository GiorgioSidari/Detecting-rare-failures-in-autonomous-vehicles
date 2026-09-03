"""
Copy of the shared components, reachable from inside the containers.

`driver.py`, `episode_budget.py`, `road_frame.py` and `road_geometry.py` are
byte-identical copies of the files in `scenarios/common/`.
`tests/test_shared_vendoring.py` compares the file hashes and fails if any pair
differs by a single character.

The copy exists because the containers' build context is `opensbt-core/`, so a
Dockerfile cannot `COPY ./scenarios` from outside that context.

To change any of these files: edit the original under `scenarios/common/`, then
copy it here. The vendoring test reports the mismatch if the copy is skipped.
"""
