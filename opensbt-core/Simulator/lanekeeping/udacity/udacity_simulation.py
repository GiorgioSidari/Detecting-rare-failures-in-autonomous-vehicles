# imports related to OpenSBT
# try:
import time
from UdacitySimulatorIO import UdacitySimulationOutput, UdacitySimulatorConfig
from .os_utils import kill_udacity_simulator

# except ImportError as e:
#     print(f"OpenSBT imports failed: {e}")
#     raise ImportError(
#         f"OpenSBT is not installed or not properly configured. Please install OpenSBT to use this module. Got: {e}")

# all other imports
try:
    from typing import List, Tuple
    import numpy as np
    import gym
except ImportError as e:
    print(f"Import failed: {e}")
    raise ImportError(
        "Some required packages are not installed. Please install the necessary packages to use this module.")

# related to this simulator
from .env.udacity_gym_env import UdacityGymEnv_RoadGen
from ..road_generator.custom_road_generator import CustomRoadGenerator
from ..self_driving.road import Road
from ..config import UDACITY_SIM_NAME, DNN_MODEL_PATH, INPUT_SHAPE, UDACITY_EXE_PATH, MAX_XTE, CAP_XTE

from ..self_driving.supervised_agent import SupervisedAgent
from typing import Callable
class UdacitySimulator():
    def __init__(self) -> None:
        # Initialize the agent
        self.agent : SupervisedAgent = SupervisedAgent(
            env_name=UDACITY_SIM_NAME,
            model_path=DNN_MODEL_PATH,
            min_speed=0,
            max_speed=1,
            input_shape=INPUT_SHAPE,
            predict_throttle=False,
            fake_images=False
        )

        #Initialize the environment
        self.env : UdacityGymEnv_RoadGen = UdacityGymEnv_RoadGen(
                        seed=1,
                        exe_path=UDACITY_EXE_PATH
                        )



    def simulate(self, simulator_config: UdacitySimulatorConfig) -> UdacitySimulationOutput:
        """Runs a simulation with the given simulator configurations

        Args:
            simulator_config (simConfig): The input simulator configuration

        Raises:
            e: The exception received during the simulation

        Returns:
            SimulationOutput: The output of the simulation
        """
        # Set up the given speed limits for the agent
        self.agent.setSpeedLimits(
            minSpeed=simulator_config.minSpeed,
            maxSpeed=simulator_config.maxSpeed
            )
        
        # Initialize the road generator
        test_generator : CustomRoadGenerator = CustomRoadGenerator(map_size=simulator_config.map_size,
                                             num_control_nodes=len(simulator_config.angles),
                                             seg_length=simulator_config.segLength)

        try:
            #Instantiate the output accumulator
            simulationOutput = UdacitySimulationOutput()

            angles = simulator_config.angles

            # Generate a road
            road : Road = test_generator.generate(
                starting_pos=simulator_config.initial_position,
                angles=angles,
                simulator_name=UDACITY_SIM_NAME)

            #Add the road configuration to the output
            simulationOutput.road = road.get_concrete_representation(to_plot=True)
            
            # Convert it to the string representation
            waypoints : str = road.get_string_repr()

            # Reset the environment, this also sets the road configuration
            obs = self.env.reset(skip_generation=False, track_string=waypoints)

            # This variable will contains the current speed, it is needed by the agent to compute the next action
            speed:float = 0

            self.done = False
            self.loopStartTime = time.time()
            iterations = 0

            # Split dei tempi per step: inferenza (agent.predict) vs attesa del
            # frame Unity (env.step). Serve a capire DOVE va il tempo del loop di
            # controllo (~0.67 s/step): CPU/inferenza o I/O verso il simulatore.
            predictSeconds = 0.0
            stepSeconds = 0.0

            while not self.done:
                # Infer next actions (moved inline)
                _t_pred0 = time.perf_counter()
                actions = self.agent.predict(obs=obs, state=dict(speed=speed, simulator_name=UDACITY_SIM_NAME))
                predictSeconds += time.perf_counter() - _t_pred0

                # Clip actions inline
                if isinstance(self.env.action_space, gym.spaces.Box): # type: ignore
                    actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high) # type: ignore

                # Environment step
                _t_step0 = time.perf_counter()
                obs, done, info = self.env.step(actions)
                stepSeconds += time.perf_counter() - _t_step0

                # Update speed
                speed = info.get("speed", 0.0)
                
                # Compute XTE inline (avoid redundant calculations)
                xte = info['cte']
                if CAP_XTE and abs(xte) > MAX_XTE:
                    xte = MAX_XTE if xte > 0 else -MAX_XTE
                
                # Add stats
                simulationOutput.addStats(
                    position=info['pos'],
                    speed=speed,
                    xte=xte,
                    steering=actions[0][0],
                    throttle=actions[0][1]
                )
                
                # Check end conditions inline
                elapsed = time.time() - self.loopStartTime
                if elapsed > simulator_config.maxTime or abs(info["cte"]) > simulator_config.maxXTE or done:
                    self.done = True
                
                iterations += 1
            
            elapsedTime = time.time() - self.loopStartTime

            # Reset the environment
            self.env.reset(skip_generation=False, track_string=waypoints)

            #Add the timing stats to the output
            simulationOutput.elapsedTime = elapsedTime
            simulationOutput.iterations = iterations
            simulationOutput.predictSeconds = predictSeconds   # tempo tot. inferenza
            simulationOutput.stepSeconds = stepSeconds         # tempo tot. attesa Unity

        except Exception as e:
            raise e
    

        return simulationOutput
    
    def killSimulation(self):
        """
        This method permanently closes the simulator
        """
        self.env.close()
        kill_udacity_simulator()
