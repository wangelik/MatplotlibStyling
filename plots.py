# import table
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplcyberpunk
import numpy as np
import seaborn as sns

from utils import make_scatter_glow


def generate_plots_mpl(figsize: tuple[int, int] = (17, 17), focus: str = "black"):
    """Utility function to generate different matplotlib plot types"""

    # load, prepare and convert sample data
    data_diamonds = sns.load_dataset("diamonds")
    clarity_ranking = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

    data_fmri = sns.load_dataset("fmri")

    data_networks = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)
    used_networks = [1, 3, 4, 5, 6, 7, 8, 11, 12, 13, 16, 17]
    used_columns = (
        data_networks.columns.get_level_values("network")
        .astype(int)
        .isin(used_networks)
    )
    data_networks = data_networks.loc[:, used_columns]
    corr_networks = data_networks.corr().groupby(level="network").mean()
    corr_networks.index = corr_networks.index.astype(int)
    corr_networks = corr_networks.sort_index().T

    N, r0, cn = 100, 0.6, 10
    x = 0.9 * np.random.rand(N)
    y = 0.9 * np.random.rand(N)
    area = (20 * np.random.rand(N)) ** 2
    c, r = np.sqrt(area), np.sqrt(x**2 + y**2)
    data_area1 = np.ma.masked_where(r < r0, area)
    data_area2 = np.ma.masked_where(r >= r0, area)
    curves, cx = np.empty((cn, N)), np.linspace(0, 1, 100)
    for i in range(1, cn + 1):
        curves[i - 1, :] = 1 + i * 0.1 + 0.15 * np.sin(cx * 2 * np.pi)

    data_flights = sns.load_dataset("flights")
    data_flights = data_flights.pivot(
        index="month", columns="year", values="passengers"
    )

    # Create figure and axes template
    f, ax = plt.subplots(nrows=3, ncols=2, figsize=figsize)

    # 1. Generate seaborn lineplots
    sns.lineplot(
        x="timepoint",
        y="signal",
        hue="region",
        style="event",
        data=data_fmri,
        ax=ax[0, 0],
    )

    # 2. Generate seaborn scatter plots
    sns.scatterplot(
        x="carat",
        y="price",
        hue="clarity",
        size="depth",
        palette="ch:r=-.2,d=.3_r",
        hue_order=clarity_ranking,
        sizes=(1, 8),
        linewidth=0,
        data=data_diamonds,
        ax=ax[0, 1],
    )

    # 3. Generate seaborn histogram charts
    sns.histplot(
        data_diamonds,
        x="price",
        hue="cut",
        multiple="stack",
        palette="light:m_r",
        edgecolor=".3",
        linewidth=0.5,
        log_scale=True,
        ax=ax[1, 0],
    )
    ax[1, 0].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax[1, 0].set_xticks([500, 1000, 2000, 5000, 10000])

    # 4. Generate seaborn violin plots
    sns.violinplot(data=corr_networks, bw_adjust=0.5, cut=1, linewidth=1, ax=ax[1, 1])

    # 5. Generate custom matplotlib scatter plots
    ax[2, 0].scatter(x, y, s=data_area1, marker="^", c=c)
    ax[2, 0].scatter(x, y, s=data_area2, marker="o", c=c)
    for i in range(cn):
        ax[2, 0].plot(cx, curves[i], lw=3)

    ax[2, 0].set_xlabel("x")
    ax[2, 0].set_ylabel("y")

    # 6. Generate seaborn heatmap
    sns.heatmap(
        data_flights, annot=True, fmt="d", linecolor=focus, linewidths=0.5, ax=ax[2, 1]
    )

    # general settings
    sns.despine(f, left=True, bottom=True)
    plt.tight_layout()


def generate_plots_cp(figsize: tuple[int, int] = (17, 10)):
    """Utility function to generate different matplotlib cyberpunk plots"""

    # load, prepare and convert sample data
    lines = [[4, 3, 5, 6, 3, 3, 5], [1, 2, 4, 2, 4, 2, 2], [2, 1, 0.5, 1, 2, 2, 1]]

    categories = ["A", "B", "C", "D", "E"]
    pos = np.arange(len(categories))
    values = [[25, 67, 19, 45, 10], [30, 50, 25, 40, 20]]

    data_networks = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)
    used_networks = [1, 3, 4, 5, 6, 7, 8, 11, 12, 13, 16, 17]
    used_columns = (
        data_networks.columns.get_level_values("network")
        .astype(int)
        .isin(used_networks)
    )
    data_networks = data_networks.loc[:, used_columns]
    corr_networks = data_networks.corr().groupby(level="network").mean()
    corr_networks.index = corr_networks.index.astype(int)
    corr_networks = corr_networks.sort_index().T

    N, r0 = 100, 0.6
    x = 0.9 * np.random.rand(N)
    y = 0.9 * np.random.rand(N)
    area = (20 * np.random.rand(N)) ** 2
    c, r = np.sqrt(area), np.sqrt(x**2 + y**2)
    mask = r < r0

    # Create figure and axes template
    f, ax = plt.subplots(nrows=2, ncols=3, figsize=figsize)

    # 1. Generate matplotlib lineplots (lines glowing)
    ax[0, 0].plot(lines[0], marker="o", label="Camp. 1")
    ax[0, 0].plot(lines[1], marker="o", label="Camp. 2")
    ax[0, 0].plot(lines[2], marker="o", label="Camp. 3")
    ax[0, 0].set_xlabel("Time")
    ax[0, 0].set_ylabel("Impact")
    ax[0, 0].legend(loc="upper left")
    mplcyberpunk.make_lines_glow(ax=ax[0, 0])
    # mplcyberpunk.add_underglow()

    # 2. Generate matplotlib lineplots (area filling)
    ax[0, 1].plot(lines[0], marker="o", label="Camp. 1")
    ax[0, 1].plot(lines[1], marker="o", label="Camp. 2")
    ax[0, 1].plot(lines[2], marker="o", label="Camp. 3")
    ax[0, 1].set_xlabel("Time")
    ax[0, 1].set_ylabel("Impact")
    ax[0, 1].legend(loc="upper left")
    mplcyberpunk.add_gradient_fill(ax=ax[0, 1], alpha_gradientglow=0.5)

    # 3. Generate matplotlib lineplots (area filled)
    ax[0, 2].plot(lines[0], marker="o", label="Camp. 1")
    ax[0, 2].plot(lines[1], marker="o", label="Camp. 2")
    ax[0, 2].plot(lines[2], marker="o", label="Camp. 3")
    ax[0, 2].set_xlabel("Time")
    ax[0, 2].set_ylabel("Impact")
    ax[0, 2].legend(loc="upper left")
    mplcyberpunk.add_glow_effects(ax=ax[0, 2])
    # mplcyberpunk.add_glow_effects(ax=ax[0,2], gradient_fill=True)

    # 4. Generate custom matplotlib scatter plots
    ax[1, 0].scatter(x[mask], y[mask], s=area[mask], marker="o", c=c[mask])
    ax[1, 0].set_xlabel("x")
    ax[1, 0].set_ylabel("y")
    make_scatter_glow(ax=ax[1, 0])

    # 5. Generate seaborn violin plots
    sns.violinplot(data=corr_networks, bw_adjust=0.5, cut=1, linewidth=1, ax=ax[1, 1])
    mplcyberpunk.make_lines_glow(ax=ax[1, 1])

    # 6. Generate matplotlib clustered bar plots
    bars1 = ax[1, 2].bar(pos - 0.35 / 2, values[0], width=0.35, label="Set 1", zorder=2)
    bars2 = ax[1, 2].bar(pos + 0.35 / 2, values[1], width=0.35, label="Set 2", zorder=2)
    ax[1, 2].set_xticks(pos)
    ax[1, 2].set_xticklabels(categories)
    ax[1, 2].set_ylabel("Sales")
    ax[1, 2].legend()
    mplcyberpunk.add_bar_gradient(bars=bars1 + bars2)

    # general settings
    sns.despine(f, left=True, bottom=True)
    plt.tight_layout()
