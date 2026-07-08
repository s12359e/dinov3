"""Training-curve plotting (static PNG diagnostic, light mode).

Follows the dataviz method: line chart for change-over-time; ONE axis per panel
(loss vs lr/temp/momentum have different scales, so they get their own small
multiples instead of a dual axis); validated reference categorical palette in
fixed slot order; thin 2px-ish lines; recessive grid; text in ink tokens.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e7e6e2"
SPINE = "#c6c5c0"
# Reference categorical palette, fixed slot order (validated set).
SERIES = {"total": "#2a78d6", "ref2ref": "#1baf7a", "target2ref": "#eda100",
          "traditional": "#008300", "repel": "#4a3aa7"}


def _style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SPINE)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_title(title, color=INK, fontsize=10, loc="left")


def plot_training_curves(log, out_path):
    """log: list of per-step dicts with keys
    step, loss, L_refref, L_target2ref, L_trad, lr, teacher_temp, momentum[, repel]."""
    steps = [r["step"] for r in log]
    fig = plt.figure(figsize=(9, 6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.4)

    # Loss panel (all series share the same unit -> one axis).
    ax = fig.add_subplot(gs[0, :])
    _style(ax, "Loss per step")
    for key, label in (("loss", "total"), ("L_refref", "ref2ref"),
                       ("L_target2ref", "target2ref"), ("L_trad", "traditional")):
        ax.plot(steps, [r[key] for r in log], color=SERIES[label],
                linewidth=1.8, label=label)
    if any("repel" in r for r in log):
        ax.plot(steps, [r.get("repel", float("nan")) for r in log],
                color=SERIES["repel"], linewidth=1.8, label="repel")
    # Legend outside the plot area so it can never collide with the curves.
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=8,
              frameon=False, ncol=5, labelcolor=INK2)
    ax.set_xlabel("step", color=INK2, fontsize=8)

    # Schedules: different scales -> small multiples, one axis each (never dual-axis).
    for i, (key, title) in enumerate((("lr", "Head LR"),
                                      ("teacher_temp", "Teacher temperature"),
                                      ("momentum", "EMA momentum"))):
        ax = fig.add_subplot(gs[1, i])
        _style(ax, title)
        ax.plot(steps, [r[key] for r in log], color=SERIES["total"], linewidth=1.8)
        ax.set_xlabel("step", color=INK2, fontsize=8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
