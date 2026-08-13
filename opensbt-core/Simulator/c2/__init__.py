"""
C2 arm on Udacity -- an **additive** package, it does not touch `lanekeeping/`.

Project constraint: nothing that existed in `opensbt-core` before this branch is
modified. `lanekeeping/` is the working Udacity pipeline, used by others too: a
change there propagates into work that is not ours.

So C2 lives here and **reuses** `lanekeeping` by composition:

  * `state_based_agent.StateBasedAgent`  -- implements the same interface as
    `SupervisedAgent` (`predict(obs, state)`), so it is interchangeable;
  * `udacity_simulation_c2.UdacitySimulatorC2` -- a subclass of
    `UdacitySimulator`: it reuses `__init__` for the Unity environment and
    overrides `simulate()` with the loop the state-based controller needs;
  * `server.py` -- a FastAPI server with the same contract as
    `SimulatorServer.py`, so the client cannot tell the two arms apart.

The container is chosen at runtime by changing only the compose entrypoint:

    command: uvicorn Simulator.c2.server:app --host 0.0.0.0 --port 8000

No new Dockerfile: the existing one already copies all of `./Simulator`.

Why the loop is rewritten rather than reused
--------------------------------------------
`UdacitySimulator.simulate()` passes the agent only `speed` and
`simulator_name`. The state-based controller also needs `pos` (to project itself
onto the road), the run's centreline, and the real `dt` (Udacity's control rate
varies with load). Adding those to the original loop would mean modifying it;
rewriting it here leaves the original intact, at the price of a duplication that
`tests/test_c2_udacity.py` keeps under control.
"""
