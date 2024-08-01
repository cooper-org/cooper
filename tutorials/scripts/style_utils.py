"""
Consistent style format for plots in example gallery.
"""

import matplotlib.pyplot as plt

SMALL_SIZE = 13
MEDIUM_SIZE = 16
BIG_SIZE = 22

COLOR_DICT = {
    "red": "#92140C",
    "green": "#72A98F",
    "blue": "#0035f5",
    "yellow": "#D4AF37",
}


def set_plot_style():
    plt.style.use("seaborn-v0_8-white")

    plt.rc("font", size=SMALL_SIZE)
    plt.rc("axes", titlesize=MEDIUM_SIZE)
    plt.rc("axes", labelsize=MEDIUM_SIZE)
    plt.rc("xtick", labelsize=SMALL_SIZE)
    plt.rc("ytick", labelsize=SMALL_SIZE)
    plt.rc("legend", fontsize=SMALL_SIZE)
    plt.rc("figure", titlesize=BIG_SIZE)
