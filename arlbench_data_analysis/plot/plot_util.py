import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns   
import math

from ..metrics.const import metric_arrows

# Function to style axes
def style_axis_perf_profile_curves(
        ax, 
        h_lines: list = []
    ):
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make bottom and left spines thicker
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    # Add gap in the corner by adjusting spine positions
    ax.spines['left'].set_position(('outward', 5))
    ax.spines['bottom'].set_position(('outward', 5))
    
    # Add light grey grid
    ax.grid(True, color='lightgrey', linestyle='-', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)  # Grid behind data
    
    # Add horizontal lines
    for h in h_lines:
        ax.axhline(y=h, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)


def plot_performance_profile_curves(
        df, 
        algo_colors: dict, 
        algo_list: list,
        h_lines: list = [0.25, 0.5, 0.75],
        save_path: str = "_figures/perf_profiles_only.pdf"
    ):
    # --- Left plot: cross-algorithm normalization (performance profiles) ---
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    for algo in algo_list:
        algo_data = df[df["algorithm"] == algo]
        # Filter tau to [0, 1]
        algo_data = algo_data[(algo_data["tau"] >= -0.01) & (algo_data["tau"] <= 1)]
        if len(algo_data) > 0:
            ax1.plot(algo_data["tau"], algo_data["performance_profile"],
                    label=algo.upper(), color=algo_colors.get(algo, None), linewidth=2)
            if "ci_lower" in algo_data.columns and "ci_upper" in algo_data.columns:
                ax1.fill_between(algo_data["tau"], algo_data["ci_lower"], algo_data["ci_upper"],
                                alpha=0.2, color=algo_colors.get(algo, None))

    ax1.set_xlabel(r"Normalized Score ($\beta$)", fontsize=14)
    ax1.set_ylabel(r"Fraction with score > $\beta$", fontsize=14)
    ax1.set_ylim(0, 1.00)
    ax1.set_xlim(-0.05, 1.05)
    style_axis_perf_profile_curves(ax1, h_lines=h_lines)
    ax1.tick_params(axis='both', labelsize=14)  # Increase tick font size
    # ax1.legend(loc='upper right', fontsize=11, frameon=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    return ax1

def plot_tunability_curves(
    df,
    algo_colors: dict,
    algo_list: list,
    h_lines: list = [0.2],
    save_path: str = "_figures/tunability_only.pdf",
    y_max: float = 1.0,
    y_min: float = 0.0
    ):  

    fig, ax = plt.subplots(figsize=(6, 3))
    data_ = df.groupby(["algorithm", "percentile"])["tunability"].mean().reset_index()
    for algo in algo_list:
        algo_data = data_[data_["algorithm"] == algo].copy()
        algo_data = algo_data.sort_values("tunability")
        if len(algo_data) > 0:
            ax.plot(algo_data["tunability"], algo_data["percentile"],
                    label=algo.upper(), color=algo_colors.get(algo, None), linewidth=2)

    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_ylim(y_max, y_min)
    ax.set_xlim(1.05, -0.05)
    ax.set_ylabel(r"Fraction with score > $\delta$", fontsize=14)
    ax.set_xlabel(r"Performance loss ($\delta$)", fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    style_axis_perf_profile_curves(ax, h_lines=h_lines)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f')) 
    # ax.legend(loc='upper right', fontsize=11, frameon=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    return ax


def plot_metrics(
    plot_df,
    alg_colors: dict,
    title_: str = None,
    save_axes_prefix=None,
    plot_ylabels=True,
    xtick_fontsize=14,
    num_xticks=5
    ):

    metric_names = plot_df["metric"].unique()
    algorithms = plot_df["algorithm"].unique()

    fig, axes = plt.subplots(1, len(metric_names), figsize=(4 * len(metric_names), 4), sharey=True)

    bar_height = 0.6

    def get_nice_ticks(vmin, vmax, num):
        # Always use a step that is a multiple of 0.05
        raw_step = (vmax - vmin) / max(num - 1, 1)
        step = max(0.05, round(raw_step / 0.05) * 0.05)
        # Find the first tick <= vmin that is a multiple of step
        start = math.floor(vmin / step) * step
        # Find the last tick >= vmax that is a multiple of step
        end = math.ceil(vmax / step) * step
        # Generate ticks
        ticks = []
        val = start
        while val <= end + 1e-8:
            ticks.append(round(val, 6))
            val += step
        # Ensure vmin and vmax are included
        if vmin < ticks[0]:
            ticks = [round(vmin, 6)] + ticks
        if vmax > ticks[-1]:
            ticks.append(round(vmax, 6))
        return ticks

    for i, (ax, metric) in enumerate(zip(axes, metric_names)):
        metric_df = plot_df[plot_df["metric"] == metric]

        for j, alg in enumerate(algorithms):
            alg_data = metric_df[metric_df["algorithm"] == alg]
            if len(alg_data) == 0:
                continue

            mean = alg_data["mean"].values[0]
            lower = alg_data["lower_ci"].values[0]
            upper = alg_data["upper_ci"].values[0]

            ax.barh(j, upper - lower, left=lower, height=bar_height,
                    color=alg_colors[alg], alpha=0.8)
            ax.vlines(mean, j - bar_height / 2, j + bar_height / 2,
                      color='black', linewidth=1.5)

        ax.set_yticks(range(len(algorithms)))
        ax.set_yticklabels([alg.upper() for alg in algorithms])
        ax.set_title(metric.replace("_", " ").title() + " " + metric_arrows[metric])
        ax.invert_yaxis()

        ax.xaxis.grid(True, color='lightgrey', linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        max_upper_ci = metric_df["upper_ci"].max()
        min_lower_ci = metric_df["lower_ci"].min()
        ax.set_xlim(min_lower_ci - 0.05, max_upper_ci + 0.05)

        # Set number of x-ticks and font size, using nice intervals
        xticks = get_nice_ticks(min_lower_ci - 0.05, max_upper_ci + 0.05, num_xticks)
        ax.set_xticks(xticks)
        ax.tick_params(axis='x', labelsize=xtick_fontsize)
        ax.tick_params(axis='y', length=0)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Save each axis as a separate figure by re-plotting
        if save_axes_prefix is not None:
            fig_single, ax_single = plt.subplots(figsize=(4, 4))
            for j, alg in enumerate(algorithms):
                alg_data = metric_df[metric_df["algorithm"] == alg]
                if len(alg_data) == 0:
                    continue
                mean = alg_data["mean"].values[0]
                lower = alg_data["lower_ci"].values[0]
                upper = alg_data["upper_ci"].values[0]
                ax_single.barh(j, upper - lower, left=lower, height=bar_height,
                               color=alg_colors[alg], alpha=0.8)
                ax_single.vlines(mean, j - bar_height / 2, j + bar_height / 2,
                                 color='black', linewidth=1.5)
            ax_single.set_yticks(range(len(algorithms)))
            if i == 0 and plot_ylabels:
                ax_single.set_yticklabels([alg.upper() for alg in algorithms])
            ax_single.set_yticklabels([])
            ax_single.invert_yaxis()
            ax_single.xaxis.grid(True, color='lightgrey', linestyle='-', linewidth=0.5)
            ax_single.set_axisbelow(True)
            ax_single.set_xlim(min_lower_ci - 0.05, max_upper_ci + 0.05)
            ax_single.set_xticks(xticks)
            ax_single.tick_params(axis='x', labelsize=xtick_fontsize)
            ax_single.spines['top'].set_visible(False)
            ax_single.spines['right'].set_visible(False)
            ax_single.spines['left'].set_visible(False)
            ax_single.tick_params(axis='y', length=0)
            ax_single.tick_params(axis='x', length=3)
            fig_single.tight_layout()
            fig_single.savefig(f"_figures/{save_axes_prefix}_{metric}.pdf")
            plt.close(fig_single)

    if title_:
        fig.suptitle(title_, fontsize=16, y=1.02)
    plt.tight_layout()