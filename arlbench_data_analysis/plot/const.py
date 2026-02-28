import seaborn as sns

DEFAULT_ALGOS = ["ppo", "dqn", "sac"]
DEFAULT_PALETTE = sns.color_palette("Set2")
DEFAULT_ALGO_COLORS = {algo: DEFAULT_PALETTE[i] for i, algo in enumerate(DEFAULT_ALGOS)}