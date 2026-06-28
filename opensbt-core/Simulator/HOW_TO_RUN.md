# How to run a simulation
In this package you can find a UdacitySimulatorConfig class containing all the configs of the simulation.

To run a simulation you can do:

```python 
from lanekeeping import UdacitySimulator
from UdacitySimulatorIO import UdacitySimulatorConfig

udacityConfig = UdacitySimulatorConfig(maxTime=2)

simulator = UdacitySimulator()

result = simulator.simulate(simulator_config=udacityConfig)

simulator.killSimulation()

print(result)
```

# How to run the server
```python 
uvicorn Simulator.SimulatorServer:app --host 0.0.0.0 --port 8000
```

then you can access documentation @ `http://127.0.0.1:8080/docs`