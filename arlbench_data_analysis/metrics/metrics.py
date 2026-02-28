import numpy as np


class Metric:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, x):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def bootstrap(self, x, n_bootstrap=1000, ci=0.95, seed=None):
        """
        Faster stratified bootstrap using pre-generated indices.
        """
        rng = np.random.default_rng(seed)
        
        # Pre-generate all bootstrap indices for each (env, config)
        boot_indices = [
            [rng.choice(len(config), size=(n_bootstrap, len(config)), replace=True) 
            for config in env]
            for env in x
        ]
        
        # Convert x to numpy arrays for faster indexing
        x_np = [[np.array(config) for config in env] for env in x]
        
        bootstrap_values = []
        for b in range(n_bootstrap):
            x_boot = [
                [x_np[e][c][boot_indices[e][c][b]].tolist() for c in range(len(x[e]))]
                for e in range(len(x))
            ]
            bootstrap_values.append(self(x_boot))
        
        bootstrap_values = np.array(bootstrap_values)
        
        alpha = 1 - ci
        return {
            "mean": np.mean(bootstrap_values),
            "lower_ci": np.percentile(bootstrap_values, 100 * alpha / 2),
            "upper_ci": np.percentile(bootstrap_values, 100 * (1 - alpha / 2))
        }


class TunedScorePerEnv(Metric):

    def __init__(self):
        super().__init__(name="Tuned Score Per Environment")

    @staticmethod
    def __call__(x):
        """
        x: list[list[list[float]]] - shape (envs, configs, seeds)
        
        Computes: avg(max(avg(x, axis=-1), axis=-1), axis=-1) - max(avg(avg(x, axis=-1), axis=0), axis=-1)
        
        First term: For each env, find the best config (by seed-averaged performance), then average across envs
        Second term: Average across envs first, then find the best config
        """
        # avg(x, axis=-1) -> average over seeds: list[list[float]] shape (envs, configs)
        seed_avg = [[np.mean(seeds) for seeds in env] for env in x]
        
        max_per_env = [max(env) for env in seed_avg]
        return np.mean(max_per_env)


class GeneralizationGap(Metric):

    def __init__(self):
        super().__init__("Generalization Gap")

    @staticmethod
    def __call__(x):
        """ (Adkins Phi)

        x: list[list[list[float]]] - shape (envs, configs, seeds)
        
        Computes: avg(max(avg(x, axis=-1), axis=-1), axis=-1) - max(avg(avg(x, axis=-1), axis=0), axis=-1)
        
        First term: For each env, find the best config (by seed-averaged performance), then average across envs
        Second term: Average across envs first, then find the best config
        """
        # avg(x, axis=-1) -> average over seeds: list[list[float]] shape (envs, configs)
        seed_avg = [[np.mean(seeds) for seeds in env] for env in x]
        
        # First term: avg(max(seed_avg, axis=-1), axis=-1)
        # max over configs per env, then average over envs
        max_per_env = [max(env) for env in seed_avg]
        first_term = np.mean(max_per_env)
        
        # Second term: max(avg(seed_avg, axis=0), axis=-1)
        # average over envs (axis=0), then max over configs
        n_configs = len(seed_avg[0])
        avg_over_envs = [np.mean([seed_avg[env][config] for env in range(len(seed_avg))]) for config in range(n_configs)]
        second_term = max(avg_over_envs)
        
        return first_term - second_term


class SpecificityGap(Metric):

    def __init__(self):
        super().__init__("Specificity Gap")

    @staticmethod
    def __call__(x):
        """
        x: list[list[list[float]]] - shape (envs, configs, seeds)
        
        Computes: avg(max(avg(x, axis=-1), axis=-1), axis=-1) - max(avg(avg(x, axis=-1), axis=0), axis=-1)
        
        First term: For each env, find the best config (by seed-averaged performance), then average across envs
        Second term: Average across envs first, then find the best config
        """
        # avg(x, axis=-1) -> average over seeds: list[list[float]] shape (envs, configs)
        seed_avg = [[np.mean(seeds) for seeds in env] for env in x]
        
        # First term: avg(max(seed_avg, axis=-1), axis=-1)
        # max over configs per env, then average over envs
        max_per_env = [max(env) for env in seed_avg]
        first_term = np.mean(max_per_env)
        
        # Second term: 1/M sum_m seed_avg[m][opt_j(m)]
        # where opt_j(m) = argmax_{m' != m, j} seed_avg[m'][j]
        M = len(seed_avg)
        second_term_values = []
        for m in range(M):
            # Find the best (m', j) pair excluding env m
            best_j = None
            best_val = -np.inf
            for m_prime in range(M):
                if m_prime == m:
                    continue
                for j in range(len(seed_avg[m_prime])):
                    if seed_avg[m_prime][j] > best_val:
                        best_val = seed_avg[m_prime][j]
                        best_j = j
            
            # Evaluate config best_j on env m
            second_term_values.append(seed_avg[m][best_j])
        
        second_term = np.mean(second_term_values)
        
        return first_term - second_term