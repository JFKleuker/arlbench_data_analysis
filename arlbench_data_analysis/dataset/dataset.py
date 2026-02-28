import pandas as pd 
import numpy as np

from metrics import Metric

class DataSet:

    def __init__(self,
                 data: pd.DataFrame
                 ):
        
        self.data = data
    
    def filter_envs(self, 
                    envs: list[str] = ["cc_mountain_car", "minigrid_unlock"]
                    ):
        """
        Filters out environments from the dataset. This is useful for creating subsets of the data for specific analyses or visualizations.

        Default are two notoriously bad working environments on our dataset
        """
        self.data = self.data[~self.data["env_name"].isin(envs)] 
    

    def get_performance_profile_data(
        self,
        n_bootstrap: int = 1,
        confidence: float = 0.95,
        tau_max: float = 1.05,
        tau_min: float = -0.05,
        n_tau: int = 100,
        perf_column: str = "normed_performance_c"
        ):
        """
        Compute performance profile with bootstrapping:
        - Stratified over envs (always include all envs)
        - Bootstrapped over (config_id, seed) pairs within each env
        
        Returns DataFrame with mean, lower CI, and upper CI.
        """
        results = []
        alpha = (1 - confidence) / 2
        tau_values = np.linspace(tau_min, tau_max, n_tau)
        
        for algo in self.data["algorithm"].unique():
            algo_data = self.data[self.data["algorithm"] == algo]
            envs = algo_data["env_name"].unique()
            
            for tau in tau_values:
                bootstrap_scores = []
                
                for _ in range(n_bootstrap):
                    env_scores = []
                    
                    for env in envs:
                        env_data = algo_data[algo_data["env_name"] == env]
                        
                        # Bootstrap sample within this env (sample with replacement)
                        boot_sample = env_data.sample(n=len(env_data), replace=True)
                        
                        HS = len(boot_sample)
                        
                        indicator_sum = (
                            (boot_sample[perf_column]) > tau
                        ).sum()
                        env_score = indicator_sum / HS
                        env_scores.append(env_score)
                    
                    # Average over environments (stratified)
                    bootstrap_scores.append(np.mean(env_scores))
                
                # Compute statistics
                mean_score = np.mean(bootstrap_scores)
                ci_lower = np.percentile(bootstrap_scores, alpha * 100)
                ci_upper = np.percentile(bootstrap_scores, (1 - alpha) * 100)
                
                results.append({
                    "algorithm": algo,
                    "tau": tau,
                    "performance_profile": mean_score,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper
                })
        
        return pd.DataFrame(results)

    def get_tunability_profile_data(
        self,
        n_tau: int = 50,
        perf_column: str = "normed_performance_g"
        ):

        all_tunabilities_list = []
        for percentile in np.arange(0, n_tau+1)/n_tau:

            tunabilities = []
            algos = []
            envs = []
            for env in self.data["env_name"].unique():
                for algo in self.data["algorithm"].unique():
                    subset = self.data[(self.data["env_name"]==env) & (self.data["algorithm"]==algo)]
                    threshold = subset[perf_column].quantile(percentile)
                    tunability = subset[perf_column].max() - threshold
                    tunabilities.append(tunability)
                    algos.append(algo)
                    envs.append(env)
            tunabilities = pd.DataFrame({
                "env_name": envs,
                "algorithm": algos,
                "tunability": tunabilities,
                "domain": [env.split("_")[0] for env in envs]
            })

            tunabilities["percentile"] = percentile
            all_tunabilities_list.append(tunabilities)
        
        return pd.concat(all_tunabilities_list)

    def get_algorithm_performances_for_metric(
            self,
            algorithm: str,
            perf_col: str = "normed_performance_c"
    ):

        alg_subset = self.data[self.data["algorithm"]==algorithm]
        alg_performances = []

        for env in alg_subset["env_name"].unique():

            env_performances = []
            
            for config_id in alg_subset["config_id"].unique():

                env_config_subset = alg_subset[(alg_subset["env_name"]==env) & (alg_subset["config_id"]==config_id)]
                env_performances.append(env_config_subset[perf_col].values)
            
            alg_performances.append(env_performances)
        
        return alg_performances


    def get_metric_data(
            self,
            metric_list: list,#[Metric],
            perf_col: str = "normed_performance_c",
            n_bootstrap: int = 100,
            ci: float = 0.95):
        metrics = {}

        for alg in self.data["algorithm"].unique():
            algorithm_performance_array = self.get_algorithm_performances_for_metric(alg, perf_col=perf_col)

            metrics[alg] = {
                metric.name: metric.bootstrap(algorithm_performance_array, n_bootstrap=n_bootstrap, ci=ci, seed=42) 
                for metric in metric_list
            }

        plot_data = []
        for alg, alg_metrics in metrics.items():
            for metric_name, stats in alg_metrics.items():
                plot_data.append({
                    "algorithm": alg,
                    "metric": metric_name,
                    "mean": stats["mean"],
                    "lower_ci": stats["lower_ci"],
                    "upper_ci": stats["upper_ci"]
                })

        return pd.DataFrame(plot_data)
    
    @staticmethod
    def _bootstrap_ci(x: list, n_boot=100, ci=95):
        boot_means = []
        for _ in range(n_boot):
            sample = np.random.choice(x, size=len(x), replace=True)
            boot_means.append(np.mean(sample))
        mean = np.mean(boot_means)
        half_width = (np.percentile(boot_means, 97.5) - np.percentile(boot_means, 2.5)) / 2
        return mean, half_width

    @staticmethod
    def _compute_rankings_binary(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        # Step 1: Sort by upper bound
        r_idx = np.argsort(mu + sigma)
        mu_sorted = mu[r_idx]
        sigma_sorted = sigma[r_idx]
        n = len(mu)
        ranks = np.zeros(n)

        # Precompute bounds
        upper_bounds = mu_sorted + sigma_sorted
        lower_bounds = mu_sorted - sigma_sorted

        for j in range(n):
            # Step 5: uj = binary_search(lower_bounds, mu[j] + sigma[j])
            uj = np.searchsorted(lower_bounds, mu[j] + sigma[j], side='right')
            # Step 6: lj = binary_search(upper_bounds, mu[j] - sigma[j])
            lj = np.searchsorted(upper_bounds, mu[j] - sigma[j], side='left')
            # Step 8: Average rank
            ranks[j] = (lj + uj) / 2

        # Reorder ranks to original order
        ranks_final = np.zeros(n)
        ranks_final[r_idx] = ranks
        return ranks_final

    def get_rank_data(
            self,
            perf_column: str = "performance"
            ) -> dict[str, pd.DataFrame]:
        """
        
        """

        rank_dfs = {}
        envs = self.data["env_name"].unique()
        algos = self.data["algorithm"].unique()

        for algo in algos:
            rank_df_data = {}
            for env in envs:

                subset = self.data[(self.data["env_name"] == env) & (self.data["algorithm"] == algo)]

                mu, sigma = [], []

                for config_id in subset["config_id"].unique():
                    config_subset = subset[subset["config_id"] == config_id]
                    mean, ci_half_width = self._bootstrap_ci(config_subset[perf_column].values)
                    mu.append(mean)
                    sigma.append(ci_half_width)
                
                mu = np.array(mu)
                sigma = np.array(sigma)

                ranks_i = self._compute_rankings_binary(mu, sigma)

                if len(ranks_i) > 0:
                    rank_df_data[env] = ranks_i

            # print(rank_df_data)
            rank_dfs[algo] = pd.DataFrame(rank_df_data)

        # give index column the name "config_id"
        for algo in algos:
            rank_dfs[algo].index.name = "config_id"
        
        return rank_dfs