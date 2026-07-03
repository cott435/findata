from os import path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import RangeSlider
import seaborn as sns
from findata.configs import DataSplits

def plot_tvt_hists(df, columns=None, bins=100, training_dates: DataSplits = DataSplits(), save_dir=None, prefix=None):
    columns = columns or df.select_dtypes(include=["number"]).columns.tolist()
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    # Define the periods based on input training data_splits
    periods = [
        ("Train", training_dates.train_start, training_dates.train_end),
        ("Validation", training_dates.validation_start, training_dates.validation_end),
        ("Test", training_dates.test_start, training_dates.test_end),
    ]

    print(f"Found data for periods: {', '.join([p[0] for p in periods])}")

    # Set a nice plot style
    sns.set_style("whitegrid")

    # --- Iterate through each indicator and create its plot ---
    for indicator in columns:
        if indicator not in df.columns:
            print(f"Warning: Column '{indicator}' not found in DataFrame. Skipping.")
            continue

        num_periods = len(periods)
        fig, axes = plt.subplots(
            num_periods, 1,
            figsize=(12, num_periods * 3),
            sharex=True  # All subplots will share the same x-axis
        )

        # If there's only one period, axes will not be an array, so we wrap it
        if num_periods == 1:
            axes = [axes]

        fig.suptitle(f'Distribution of "{indicator.upper()}" Over Time', fontsize=18, y=0.99)

        # --- Create a subplot for each period ---
        for i, (name, start, end) in enumerate(periods):
            ax = axes[i]

            # Filter the data_ for the specific period
            period_data = df[(df['date'] >= pd.to_datetime(start)) & (df['date'] <= pd.to_datetime(end))]

            if period_data.empty:
                ax.set_title(f"{name}: {start} to {end} (No data)")
                ax.set_xlabel("")  # Remove x-label for all but the last plot
                ax.set_ylabel("Frequency")
                continue

            # Plot the histogram with a Kernel Density Estimate (KDE) overlay
            sns.histplot(
                data=period_data,
                x=indicator,
                bins=bins,
                kde=True,
                ax=ax,
                alpha=0.7
            )

            # Calculate mean and std dev for context
            mean_val = period_data[indicator].mean()
            std_val = period_data[indicator].std()

            ax.set_title(f"{name}: {start} to {end} (μ: {mean_val:.2f}, σ: {std_val:.2f})")
            ax.set_xlabel("")  # Remove x-label for all but the last plot
            ax.set_ylabel("Frequency")
            ax.axvline(mean_val, color='r', linestyle='--', linewidth=1.5, label=f'Mean ({mean_val:.2f})')
            ax.legend()

        # Add a single x-label to the last subplot
        axes[-1].set_xlabel(indicator.capitalize())

        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to make room for suptitle
        if save_dir:
            name = f'{prefix}_{indicator.upper()}.png' if prefix else f'{indicator.upper()}.png'
            plt.savefig(path.join(save_dir, name), dpi=600)
            plt.close()
        else:
            plt.show()


def visualize_scalings(dfs, labels=None, columns=None, bins=100):
    """
    Visualize histograms of specified columns across multiple DataFrames for comparison.

    Parameters:
    - dfs: list of pandas DataFrames (must have the same column names)
    - labels: optional list of strings to label each DataFrame (defaults to "Dataframe 1", etc.)
    - columns: optional list of column names to plot (defaults to all numeric columns in the first DataFrame)
    - bins: number of bins for the histogram (default: 100)
    """
    if not isinstance(dfs, list):
        dfs = [dfs]

    num_dfs = len(dfs)

    if labels is None:
        labels = [f"Dataframe {i + 1}" for i in range(num_dfs)]
    elif len(labels) != num_dfs:
        raise ValueError("Number of labels must match the number of DataFrames.")

    if columns is None:
        columns = dfs[0].select_dtypes(include=["number"]).columns.tolist()

    # Set a nice plot style
    sns.set_style("whitegrid")

    # --- Iterate through each indicator and create its plot ---
    for indicator in columns:
        # Check if the indicator exists in all DataFrames
        if any(indicator not in df.columns for df in dfs):
            print(f"Warning: Column '{indicator}' not found in all DataFrames. Skipping.")
            continue

        fig, axes = plt.subplots(
            num_dfs, 1,
            figsize=(12, num_dfs * 3),
            sharex=False  # All subplots will share the same x-axis
        )

        # If there's only one DataFrame, axes will not be an array, so we wrap it
        if num_dfs == 1:
            axes = [axes]

        fig.suptitle(f'Distribution of "{indicator.upper()}" Across Datasets', fontsize=18, y=0.99)

        # --- Create a subplot for each DataFrame ---
        for i, (df, label) in enumerate(zip(dfs, labels)):
            ax = axes[i]

            # Get the data_ for the indicator
            data = df[indicator]

            # Plot the histogram with a Kernel Density Estimate (KDE) overlay
            sns.histplot(
                data=data,
                bins=bins,
                kde=True,
                ax=ax,
                alpha=0.7
            )

            # Calculate mean and std dev for context
            mean_val = data.mean()
            std_val = data.std()

            ax.set_title(f"{label} (μ: {mean_val:.2f}, σ: {std_val:.2f})")
            ax.set_xlabel("")  # Remove x-label for all but the last plot
            ax.set_ylabel("Frequency")
            ax.axvline(mean_val, color='r', linestyle='--', linewidth=1.5, label=f'Mean ({mean_val:.2f})')
            ax.legend()

        # Add a single x-label to the last subplot
        axes[-1].set_xlabel(indicator.capitalize())

        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to make room for suptitle
        # plt.savefig(f'{indicator.upper()}_scalings.png', dpi=600)
        plt.show()



def plot_dfs(*dfs, line=None, dark=False, reversals=None):
    """
    line: dict {ax_index: y_levels}
    Each df is assumed to have a DatetimeIndex.
    """
    if dark:
        plt.style.use("dark_background")
    dfs = list(dfs)
    for i in range(len(dfs)):
        dfs[i] = dfs[i].copy()
        dfs[i].index = pd.to_datetime(dfs[i].index).tz_localize(None)
    if line is None:
        line = {}

    fig, ax = plt.subplots(nrows=len(dfs), sharex=True, figsize=(10, 6))
    if isinstance(ax, plt.Axes):
        ax = [ax]

    # Plot all series
    for i, df in enumerate(dfs):
        df.plot(ax=ax[i])
        ax[i].legend()
        if reversals is not None:
            rev = reversals[i]
            rev  = rev[rev.index > df.index[0].date()]
            top = rev[rev['type'] == 'top']['value']
            bottom = rev[rev['type'] == 'bottom']['value']
            ax[i].scatter(top.index, top.values, label='Top', color='lime', marker='^', s=80, edgecolor='black')
            ax[i].scatter(bottom.index, bottom.values, label='Bottom', color='orange', marker='v', s=80, edgecolor='black')

        # Optional horizontal lines
        if i in line:
            vals = line[i]
            if not isinstance(vals, (tuple, list)):
                vals = [vals]
            for level in vals:
                ax[i].axhline(level, color='k', ls='--', linewidth=0.8)

    # Space for slider
    plt.subplots_adjust(bottom=0.15)

    # Slider axis
    slider_ax = fig.add_axes([0.1, 0.05, 0.8, 0.03])

    # Use index of first df for slider (assumes same index across dfs)
    date_index = dfs[0].index
    date_nums = mdates.date2num(date_index)

    slider = RangeSlider(
        ax=slider_ax,
        label="Date Range",
        valmin=date_nums[0],
        valmax=date_nums[-1],
        valinit=(date_nums[0], date_nums[-1]),
    )
    stats = []
    for df in dfs:
        stats.append({
            "index": df.index,
            "min": df.min(axis=1),
            "max": df.max(axis=1)
        })

    def update(val):
        start_num, end_num = slider.val
        # Update x-limits (matplotlib date numbers)
        for ax_i in ax:
            ax_i.set_xlim(start_num, end_num)

        start_dt = pd.to_datetime(mdates.num2date(start_num)).to_datetime64()
        end_dt = pd.to_datetime(mdates.num2date(end_num)).to_datetime64()

        for i, ax_i in enumerate(ax):

            s = stats[i]
            mask = (s["index"] >= start_dt) & (s["index"] <= end_dt)
            y_min = s["min"][mask].min()
            y_max = s["max"][mask].max()

            # Avoid zero-height range
            if y_min == y_max:
                eps = 1e-6
                y_min -= eps
                y_max += eps

            # Optional margin
            margin = 0.05 * (y_max - y_min)
            ax_i.set_ylim(y_min - margin, y_max + margin)

        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()


def plot_df_hists(*dfs, sharex_ax=None, bins=100, titles=None):
    if sharex_ax is None:
        sharex_ax = []
    fig, ax = plt.subplots(nrows=len(dfs))
    if isinstance(ax, plt.Axes):
        ax = [ax]
    if sharex_ax:
        master = ax[sharex_ax[0]]
        for idx in sharex_ax[1:]:
            ax[idx].sharex(master)
    for i, df in enumerate(dfs):
        df.hist(ax=ax[i], bins=bins, density=True)
        if titles:
            ax[i].set_title(titles[i])
    return fig, ax


