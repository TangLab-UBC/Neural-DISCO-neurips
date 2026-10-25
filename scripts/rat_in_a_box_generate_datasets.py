import os
import sys
import random

# Import this in scripts to access self-made packages (ie. src.utils)
current_dir = os.getcwd()
sys.path.append(current_dir)

# Import ratinabox
import ratinabox
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import HeadDirectionCells, Neurons, PlaceCells, SpeedCell
from tqdm import tqdm  # gives time bar
import pandas as pd
import torch
import numpy as np
from src.utils import load_config


DEFAULT_SEED = 42


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PureNoiseNeurons(Neurons):
    def __init__(self, Agent, params={}):
        super().__init__(Agent, params)
        self.mean_rate = params.get("mean_rate", 5)
        self.std_rate = params.get("std_rate", 1)

    def get_state(self, evaluate_at=None, **kwargs):
        n_neurons = self.n
        rates = np.random.normal(self.mean_rate, self.std_rate, size=(n_neurons,))
        return np.maximum(0, rates)


def generate_trajectories(minutes):
    Env1 = Environment(params={"aspect": 2, "scale": 1})
    Env1.add_wall([[1, 0], [1, 1]])

    Ag1 = Agent(Env1)
    Ag1.pos = np.array([0.5, 0.5])
    Ag1.speed_mean = 0.2

    Env2 = Environment(params={"aspect": 2, "scale": 1})

    Ag2 = Agent(Env2)
    Ag2.pos = np.array([0.5, 0.5])
    Ag2.speed_mean = 0.2

    dt = 50e-3
    T = minutes * 60

    for _ in tqdm(range(int(T / dt) + 100)):
        Ag1.update(dt=dt)
        Ag2.update(dt=dt)

    shifted_times = Ag1.history["t"]
    shifted_positions = np.array(Ag1.history["pos"])

    cfact_shifted_times = Ag1.history["t"]
    cfact_shifted_positions = np.array(Ag1.history["pos"])
    cfact_shifted_positions[:, 0] = cfact_shifted_positions[:, 0] + 1

    train_times = Ag2.history["t"]
    train_positions = Ag2.history["pos"]

    inverted_times = Ag2.history["t"]
    inverted_positions = np.flip(Ag2.history["pos"], axis=0)

    train = (train_times, train_positions)
    inverted = (inverted_times, inverted_positions)
    shifted = (shifted_times, shifted_positions)
    cfact_shifted = (cfact_shifted_times, cfact_shifted_positions)
    return [train, inverted, shifted, cfact_shifted]


def run_simulation(noise=0.1, num_cells=100, minutes=5, include_noise_cell=False, seed=DEFAULT_SEED):
    set_global_seed(seed)
    ratinabox.stylize_plots()

    trajectories = generate_trajectories(minutes)
    names = ["train", "inverted", "shifted", "cfact_shifted"]

    Env = Environment(params={"aspect": 2, "scale": 1})

    for trajectory, name in zip(trajectories, names):
        times, positions = trajectory
        Ag = Agent(Env)
        Ag.import_trajectory(
            times=times,
            positions=positions,
        )
        create_cells(
            Ag,
            name,
            noise=noise,
            num_cells=num_cells,
            minutes=minutes,
            include_noise_cell=include_noise_cell,
            seed=seed,
        )


def create_cells(Ag, name, noise=0.1, num_cells=100, minutes=5, include_noise_cell=False, seed=DEFAULT_SEED):
    # Reset the RNG so every dataset variant shares the same place-cell layout.
    np.random.seed(seed)

    PCs = PlaceCells(
        Ag,
        params={
            "n": num_cells,
            "description": "gaussian",
            "place_cell_centres": "uniform",
            "widths": 0.40,
            "wall_geometry": "euclidean",
            "color": "C1",
        },
    )

    HeadDirectionCells_cells = HeadDirectionCells(
        Ag, params={"n": num_cells, "noise_std": noise, "noise_coherence_time": 0.2}
    )
    SpeedCell_cells = SpeedCell(
        Ag, params={"n": num_cells, "noise_std": 0.2 * noise, "noise_coherence_time": 0.2}
    )
    neuron_populations = [PCs, HeadDirectionCells_cells, SpeedCell_cells]
    if include_noise_cell:
        neuron_populations.append(
            PureNoiseNeurons(Ag, params={"n": num_cells, "mean_rate": 10, "std_rate": 2})
        )

    dt = 50e-3
    T = minutes * 60

    for _ in tqdm(range(int(T / dt) + 100)):
        Ag.update(dt=dt)
        for population in neuron_populations:
            population.update()

    dataset = pd.DataFrame(Ag.history)
    all_neural_data = np.maximum(
        np.hstack(
            [np.stack(population.history["firingrate"]) for population in neuron_populations]
        ),
        0,
    ).astype(np.float32)

    dataset["spikes"] = list(torch.poisson(torch.from_numpy(all_neural_data)).numpy())
    dataset["rates"] = list(all_neural_data)

    dataset = {key: values[50:-50] for key, values in dataset.items()}
    dataset = pd.DataFrame(dataset)

    path = (
        f"./datasets/custom_rat_in_a_box_dataset_type:_({name})_"
        f"noise_cell:_{include_noise_cell}_noise:_{noise}_minutes:_{minutes}_num_cells:_{num_cells}.pkl"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataset.to_pickle(path)


if __name__ == "__main__":
    config_filepath = str(sys.argv[1])
    cfg = load_config(filepath=config_filepath)
    run_simulation(
        cfg.noise,
        cfg.num_cells,
        cfg.minutes,
        getattr(cfg, "include_noise_cell", False),
        getattr(cfg, "seed", DEFAULT_SEED),
    )
