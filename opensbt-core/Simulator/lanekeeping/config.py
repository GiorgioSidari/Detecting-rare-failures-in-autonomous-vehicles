import os
from typing import Tuple

# DNN Model
# Selectable per-container via the DNN_MODEL_PATH environment variable so that
# two simulator containers can run different autopilots for the multi-model
# comparison (Step D). Falls back to the chauffeur model when unset.
DNN_MODEL_PATH = os.environ.get(
    "DNN_MODEL_PATH",
    r"./Simulator/SelfDrivingModels/mixed-chauffeur.h5",
)


# Simulators
UDACITY_EXE_PATH = "./Simulator/SimulatorExec/ubuntu_binaries/ubuntu.x86_64"

################

IN_WIDTH : int = 320
IN_HEIGHT : int = 160

ROAD_WIDTH = 8.0  # Used but useless as far as I have tested
NUM_SAMPLED_POINTS = 100  # Number of points used to build the road
SEG_LENGTH = 25  # Length of each segment in the road

MAX_EPISODE_STEPS = 2000

IMAGE_CHANNELS : int = 3
INPUT_SHAPE : Tuple[int]  = (IN_HEIGHT, IN_WIDTH, IMAGE_CHANNELS)
INPUT_DIM = INPUT_SHAPE
DISPLACEMENT = 2

MAX_SPEED = 30  # Maximum speed for the car
MIN_SPEED = 10  # Minimum speed for the car

MAP_SIZE = 250  # Size of the map in meters

# The portion of the image to crop in the Udacity simulator, it will keep the vertical strip of the image whose x coordinate is between [0] and [1]
CROP_UDACITY = [60, -25]
CROP_DONKEY = [60, 0]

# Wait time for the Udacity simulator to load
UDACITY_SLEEP = 2

# Scalar multiplied to the steering angle to correct it ( > 1 means more steering, < 1 means less steering)
STEERING_CORRECTION = 1
# Max # of seconds the simulation can run
TIME_LIMIT = 30
# If True, the XTE will be capped to MAX_XTE
CAP_XTE = True
# Max cross-track error (XTE) allowed
MAX_XTE = 3

# USI

BEAMNG_SIM_NAME = "beamng"
DONKEY_SIM_NAME = "donkey"
UDACITY_SIM_NAME = "udacity"
MOCK_SIM_NAME = "mock"
SIMULATOR_NAMES = [BEAMNG_SIM_NAME, DONKEY_SIM_NAME,
                   UDACITY_SIM_NAME, MOCK_SIM_NAME]
AGENT_TYPE_RANDOM = "random"
AGENT_TYPE_SUPERVISED = "supervised"
AGENT_TYPE_AUTOPILOT = "autopilot"
AGENT_TYPES = [AGENT_TYPE_RANDOM, AGENT_TYPE_SUPERVISED, AGENT_TYPE_AUTOPILOT]
TEST_GENERATORS = ["random", "sin"]

IMAGE_HEIGHT = IN_HEIGHT
IMAGE_WIDTH = IN_WIDTH
