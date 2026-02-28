import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np


DEFAULT_ENVS = envs = ["atari_battle_zone", "atari_double_dunk", "atari_phoenix", "atari_this_game", "atari_battle_zone", "box2d_lunar_lander", "box2d_continuous_lunar_lander", "box2d_bipedal_walker", "cc_acrobot", "cc_cartpole", "cc_mountain_car", "cc_continuous_mountain_car", "cc_pendulum", "minigrid_door_key", "minigrid_empty_random", "minigrid_four_rooms", "minigrid_unlock", "brax_ant", "brax_halfcheetah", "brax_hopper", "brax_humanoid"]
DEFAULT_ALGOS = ["ppo", "dqn", "sac"]


class Dataloader:
    """
        Loads and normalizes performance data for reinforcement learning experiments across multiple environments and algorithms.

        The Dataloader aggregates results from CSV files, each corresponding to a specific environment and algorithm, and provides several normalization schemes to facilitate fair comparison of performance metrics. 

        Parameters:
            path (str): Directory containing the CSV files. Each file should be named as '{env_name}_{algorithm}.csv'.
            envs (list[str], optional): List of environment names to load. Defaults to a predefined set.
            algos (list[str], optional): List of algorithm names to load. Defaults to a predefined set.

        Attributes:
            data (pd.DataFrame): The loaded and normalized data, with columns for environment, algorithm, configuration, seed, and various normalized performance metrics.

        Normalization schemes (see `normalize_data` for details):
            - Normed performance across algorithms and tasks.
            - Normed performance within algorithms.
            - Normed performance across algorithms averaged over seeds.
            - Normed performance within algorithms averaged over seeds.

        Example:
            >>> loader = Dataloader(path="results/")
            >>> df = loader.data
    """
    def __init__(self, 
                 path: str,
                 envs: list[str] = DEFAULT_ENVS,
                 algos: list[str] = DEFAULT_ALGOS):
        
        self.path = path
        self.envs = envs
        self.algos = algos

        self.data = self.load_data()
        self.normalize_data()

        # add new columns
        self.data["domain"] = self.data["env_name"].apply(lambda x: x.split("_")[0])

    def load_data(self) -> pd.DataFrame:
        
        data = []
        for env in self.envs:
            for algo in self.algos:
                try:
                    partial_data = pd.read_csv(f"{self.path}/{env}_{algo}.csv")
                    partial_data["env_name"] = env
                    partial_data["algorithm"] = algo
                    data.append(partial_data)
                except FileNotFoundError:
                    continue
        return pd.concat(data)
    
    def normalize_data(self) -> None:
        """
        Normalizes the performance metric in several ways to enable meaningful comparisons across algorithms, tasks, configurations, and seeds.

        Let $y_{a,m,j,s}$ denote the post-training return for algorithm $a$, task $m$, configuration $j$, and seed $s$. Normalization is always performed as:
            y_{a,m,j,s} → (y_{a,m,j,s} - y_min) / (y_max - y_min)

        The following normalizations are provided:

        **Normed Performance Across Algorithms (`normed_performance_c`):**
            - Normalizes performance per environment (task) across all algorithms, configurations, and seeds.
            - $y_\mathrm{max} = \max_{a,j,s} y_{a,m,j,s}$ and $y_\mathrm{min} = \min_{a,j,s} y_{a,m,j,s}$ for each task $m$.

        **Normed Performance Within Algorithms (`normed_performance_g`):**
            - Normalizes performance per environment and algorithm, across all configurations and seeds.
            - $y_\mathrm{max} = \max_{j,s} y_{a,m,j,s}$ and $y_\mathrm{min} = \min_{j,s} y_{a,m,j,s}$ for each algorithm $a$ and task $m$.

        **Normed Performance Across Algorithms Averaged Over Seeds (`normed_performance_cs`):**
            - Computes the average performance over seeds for each configuration, then normalizes these averages per environment across all algorithms and configurations.
            - $\bar{y}_{a,j,m} = \mathrm{mean}_{s} y_{a,j,m,s}$
            - $y_\mathrm{max} = \max_{a,j} \bar{y}_{a,j,m}$ and $y_\mathrm{min} = \min_{a,j} \bar{y}_{a,j,m}$ for each task $m$.

        **Normed Performance Within Algorithms Averaged Over Seeds (`normed_performance_gs`):**
            - Computes the average performance over seeds for each configuration, then normalizes these averages per environment and algorithm.
            - $y_\mathrm{max} = \max_{j} \bar{y}_{a,j,m}$ and $y_\mathrm{min} = \min_{j} \bar{y}_{a,j,m}$ for each algorithm $a$ and task $m$.

        Note:
            - The normalization axis (across or within algorithms) determines whether the best observed value is taken globally or per algorithm.
            - Unless stated otherwise, "across algorithms" normalization uses the best single run, and "within algorithms" normalization uses the best per algorithm.
        """
        # normalization cross-algorithm per environment
        self.data["normed_performance_c"] = self.data.groupby("env_name")["performance"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        )

        # on data_two we normalize per env and algorithm
        self.data["normed_performance_g"] = self.data.groupby(["env_name", "algorithm"])["performance"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        )

        # on data_two we normalize per env and algorithm
        self.data["_seed_avg"] = self.data.groupby(["env_name", "algorithm", "config_id"])["performance"].transform("mean")
        
        # self.data["normed_performance_gs"] = self.data.groupby(["env_name", "algorithm"], group_keys=False)["_seed_avg"].transform(
        #     lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        # )
        self.data["normed_performance_gs"] = (
            (self.data["performance"] - self.data.groupby(["env_name", "algorithm"])["_seed_avg"].transform("min")) /
            (self.data.groupby(["env_name", "algorithm"])["_seed_avg"].transform("max") -
            self.data.groupby(["env_name", "algorithm"])["_seed_avg"].transform("min") + 1e-8)
        )

        self.data["normed_performance_cs"] = self.data.groupby("env_name", group_keys=False)["_seed_avg"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        )

        

        # drop the intermediate column
        self.data.drop(columns=["_seed_avg"], inplace=True)