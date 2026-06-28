from lanekeeping.plotter.scenario_plotter_roads import plot_gif
import opensbt
from lanekeeping.evaluation import critical
from lanekeeping.evaluation import fitness
from opensbt.utils.log_utils import disable_pymoo_warnings, setup_logging
from opensbt.problem.adas_problem import ADASProblem
from opensbt.experiment.search_configuration import DefaultSearchConfiguration
from opensbt.config import LOG_FILE
from lanekeeping.udacity.udacity_simulation import UdacitySimulator
import os
import logging as log
from opensbt.algorithm.nsga2_optimizer import NsgaIIOptimizer
import numpy as np
from opensbt.model_ga.problem import SimulationProblem
from opensbt.model_ga.result import SimulationResult
from opensbt.model_ga.population import PopulationExtended
import pymoo

from opensbt.model_ga.individual import IndividualSimulated
import opensbt.visualization
import opensbt.visualization.scenario_plotter
pymoo.core.individual.Individual = IndividualSimulated

pymoo.core.population.Population = PopulationExtended

pymoo.core.result.Result = SimulationResult

pymoo.core.problem.Problem = SimulationProblem


opensbt.visualization.scenario_plotter.plot_scenario_gif = plot_gif

os.chmod(os.getcwd(), 0o777)
logger = log.getLogger(__name__)
setup_logging(LOG_FILE)
disable_pymoo_warnings()

problem = ADASProblem(
    problem_name="Udacity_5A_0-85_XTE_AVG",
    scenario_path="",
    xl=[0, 0, 0, 0, 0],
    xu=[85, 85, 85, 85, 85],
    simulation_variables=[
        "angle1",
        "angle2",
        "angle3",
        "angle4",
        "angle5"],
    fitness_function=fitness.MaxAvgXTEFitness(),
    critical_function=critical.MaxXTECriticality(),
    simulate_function=UdacitySimulator.simulate,
    simulation_time=30,
    sampling_time=0.25,
)

# Set search configuration
config = DefaultSearchConfiguration()
config.n_generations = 2
config.population_size = 1

config.ideal = np.asarray([-3, -10])    # worst (=most critical) fitness values
config.nadir = np.asarray([0, 0])    # worst (=most critical) fitness values

optimizer = NsgaIIOptimizer(
    problem=problem,
    config=config)

res = optimizer.run()
res.write_results(params=optimizer.parameters)
