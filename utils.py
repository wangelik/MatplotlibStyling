# import table
from typing import Optional

import matplotlib.pyplot as plt


# adjusted/fixed scatter_glow function from mplcyberpunk package
def make_scatter_glow(
    ax: Optional[plt.Axes] = None,
    n_glow_lines: int = 10,
    diff_dotwidth: float = 1.2,
    alpha: float = 0.3,
) -> None:
    """Add glow effect to dots in scatter plot.

    Each plot is redrawn 10 times with increasing width to create glow effect."""

    if not ax:
        ax = plt.gca()

    scatterpoints = ax.collections[-1]
    x, y = scatterpoints.get_offsets().data.T
    dot_color = scatterpoints.get_array()
    dot_size = scatterpoints.get_sizes()

    alpha = alpha / n_glow_lines

    for i in range(1, n_glow_lines):
        ax.scatter(x, y, s=dot_size * (diff_dotwidth**i), c=dot_color, alpha=alpha)
