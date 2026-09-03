"""
State-based arm on Udacity: an additive package that leaves `lanekeeping/`
untouched.

The package reuses `lanekeeping` by composition:

  * `state_based_agent.StateBasedAgent` -- implements the same interface as
    `SupervisedAgent` (`predict(obs, state)`), so the two are interchangeable;
  * `udacity_simulation_c2.UdacitySimulatorC2` -- subclasses `UdacitySimulator`,
    inheriting `__init__` (which brings up the Unity environment) and overriding
    `simulate()` with a loop that also passes the agent `pos`, the run's
    centreline and the measured `dt`, none of which the inherited loop supplies;
  * `server.py` -- a FastAPI application exposing the same routes and payloads
    as `SimulatorServer.py`, so a client sees no difference between the arms.

Which of the two runs inside the container is selected by the compose
entrypoint alone:

    command: uvicorn Simulator.c2.server:app --host 0.0.0.0 --port 8000

There is no separate Dockerfile: the existing one copies the whole `./Simulator`
tree. `tests/test_c2_udacity.py` covers the rewritten loop.
"""
