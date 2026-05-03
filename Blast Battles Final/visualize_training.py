"""
visualize_training.py
=====================
Parse TensorBoard logs from train.py and save PNG charts.

Charts produced
---------------
  charts/01_reward_curve.png     — Mean episode reward over all training steps
  charts/02_winrate_hard.png     — Win rate vs hard heuristic bot over all steps
  charts/03_selfplay_winrates.png — Self-play stage: win rate vs self + vs hard (two lines)
  charts/04_episode_length.png   — Average episode length over all training steps
  charts/combined_dashboard.png  — All four charts in a 2×2 grid

Usage
-----
  python visualize_training.py
  python visualize_training.py --logdir tensorboard_logs --outdir charts
  python visualize_training.py --show

Requirements
------------
  pip install tensorboard matplotlib numpy
"""

import argparse, os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Colour palette (matches the game's dark UI) ───────────────
C_REWARD  = "#44ff88"   # green  — rollout reward
C_EVAL    = "#e8b84b"   # gold   — EvalCallback reward
C_WR_HARD = "#4ab8ff"   # blue   — win rate vs hard bot
C_WR_SELF = "#c44bff"   # purple — win rate vs self
C_EP_LEN  = "#ff8844"   # orange — episode length
C_GATE    = "#ff4444"   # red    — 50% reference line
C_BG      = "#0a0c0f"
C_SURFACE = "#111418"
C_BORDER  = "#2a3040"
C_TEXT    = "#e8eaf0"
C_MUTED   = "#6b7585"

plt.rcParams.update({
    "figure.facecolor": C_BG,
    "axes.facecolor":   C_SURFACE,
    "axes.edgecolor":   C_BORDER,
    "axes.labelcolor":  C_TEXT,
    "xtick.color":      C_MUTED,
    "ytick.color":      C_MUTED,
    "text.color":       C_TEXT,
    "grid.color":       C_BORDER,
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "lines.linewidth":  2.0,
    "font.family":      "monospace",
    "legend.framealpha": 0.2,
    "legend.labelcolor": C_TEXT,
    "legend.fontsize":   9,
})


# ══════════════════════════════════════════════════════════════
#  TENSORBOARD READER
# ══════════════════════════════════════════════════════════════

def read_tb(logdir: str, tag: str) -> list[tuple[int, float]]:
    """Return [(step, value), ...] for a scalar tag found anywhere under logdir."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator, STORE_EVERYTHING_SIZE_GUIDANCE)
    except ImportError:
        print("[warn] pip install tensorboard to read logs")
        return []

    points = []
    for fpath in glob.glob(
            os.path.join(logdir, "**", "events.out.*"), recursive=True):
        ea = EventAccumulator(fpath,
                              size_guidance=STORE_EVERYTHING_SIZE_GUIDANCE)
        ea.Reload()
        if tag in ea.Tags().get("scalars", []):
            for ev in ea.Scalars(tag):
                points.append((ev.step, ev.value))
    return sorted(set(points))


def ema(values: np.ndarray, span: int = 7) -> np.ndarray:
    """Exponential moving average smoothing."""
    alpha = 2.0 / (span + 1)
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _unzip(pts):
    if not pts:
        return None, None
    s, v = zip(*pts)
    return np.array(s, dtype=float), np.array(v, dtype=float)


# ══════════════════════════════════════════════════════════════
#  SHARED AXIS FORMATTING
# ══════════════════════════════════════════════════════════════

def _fmt_steps(ax):
    """Format x-axis as k / M steps."""
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k"))


def _fmt_pct(ax):
    """Format y-axis as whole-number percentages."""
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))


def _base(ax, title, xlabel, ylabel):
    ax.set_title(title, color=C_TEXT, fontsize=12, pad=10, fontweight="bold")
    ax.set_xlabel(xlabel, color=C_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=C_MUTED, fontsize=9)
    ax.grid(True)
    ax.spines[:].set_color(C_BORDER)
    _fmt_steps(ax)


def _no_data(ax, msg):
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, color=C_MUTED, fontsize=9,
            linespacing=1.6)


def _line(ax, pts, color, label, raw_alpha=0.2):
    s, v = _unzip(pts)
    if s is None:
        return False
    ax.plot(s, v,       color=color, alpha=raw_alpha, linewidth=1)
    ax.plot(s, ema(v),  color=color, label=label,     linewidth=2)
    ax.fill_between(s, ema(v), alpha=0.07, color=color)
    return True


# ══════════════════════════════════════════════════════════════
#  CHART 1 — REWARD CURVE
# ══════════════════════════════════════════════════════════════

def chart_reward(ax, logdir):
    """
    Mean episode reward over all training steps.
    Shows the raw rollout reward and the EvalCallback reward on the same axes.
    """
    found = False
    for tag, color, label in [
        ("rollout/ep_rew_mean", C_REWARD, "Rollout reward (training envs)"),
        ("eval/mean_reward",    C_EVAL,   "Eval reward (held-out envs)"),
    ]:
        if _line(ax, read_tb(logdir, tag), color, label):
            found = True

    if not found:
        _no_data(ax, "No reward data found.\nRun training first.")

    _base(ax,
          title   = "Mean Episode Reward",
          xlabel  = "Training Steps",
          ylabel  = "Mean Reward")
    ax.legend()


# ══════════════════════════════════════════════════════════════
#  CHART 2 — WIN RATE VS HARD BOT
# ══════════════════════════════════════════════════════════════

def chart_winrate_hard(ax, logdir):
    """
    Win rate against the hard heuristic bot over all training steps.
    Logged by WinRateCallback every 20,000 steps.
    The dashed red line marks the 50% target.
    """
    pts = read_tb(logdir, "eval/win_rate_hard")
    if not _line(ax, pts, C_WR_HARD, "Win rate vs Hard bot"):
        _no_data(ax,
                 "No win rate data found.\n"
                 "Retrain with the updated train.py\n"
                 "to generate this curve.")

    ax.axhline(0.50, color=C_GATE, linestyle="--",
               linewidth=1.5, label="50% target", zorder=3)
    ax.set_ylim(0, 1.0)
    _fmt_pct(ax)
    _base(ax,
          title   = "Win Rate vs Hard Bot",
          xlabel  = "Training Steps",
          ylabel  = "Win Rate")
    ax.legend()


# ══════════════════════════════════════════════════════════════
#  CHART 3 — SELF-PLAY WIN RATES (two lines)
# ══════════════════════════════════════════════════════════════

def chart_selfplay(ax, logdir):
    """
    Self-play stage only: win rate of the self-play agent vs its frozen
    opponent (purple) and vs the hard heuristic bot (blue).

    These use separate TensorBoard tags (selfplay_self, selfplay_hard) so
    they only contain data from the self-play run, not the hard training run.

    The gap between the two lines shows whether the agent is over-fitting
    to its frozen opponent or genuinely improving at the game.
    """
    found = False
    for tag, color, label in [
        ("eval/win_rate_selfplay_self", C_WR_SELF, "vs Frozen self-play opponent"),
        ("eval/win_rate_selfplay_hard", C_WR_HARD, "vs Hard heuristic bot"),
    ]:
        if _line(ax, read_tb(logdir, tag), color, label):
            found = True

    if not found:
        _no_data(ax,
                 "No self-play data found.\n"
                 "Run: python train.py --selfplay\n"
                 "     --checkpoint checkpoints\\hard\\best_winrate_model.zip")

    ax.axhline(0.50, color=C_GATE, linestyle="--",
               linewidth=1.5, label="50% reference", zorder=3)
    ax.set_ylim(0, 1.05)
    _fmt_pct(ax)
    _base(ax,
          title   = "Self-Play Win Rates",
          xlabel  = "Training Steps",
          ylabel  = "Win Rate")
    ax.legend()


# ══════════════════════════════════════════════════════════════
#  CHART 4 — EPISODE LENGTH
# ══════════════════════════════════════════════════════════════

def chart_ep_length(ax, logdir):
    """
    Average episode length in turns over all training steps.
    Rising length means the agent is surviving longer and playing
    more competitively — games last up to 50 turns.
    """
    pts = read_tb(logdir, "rollout/ep_len_mean")
    if not _line(ax, pts, C_EP_LEN, "Avg episode length"):
        _no_data(ax, "No episode length data found.\nRun training first.")

    _base(ax,
          title   = "Average Episode Length",
          xlabel  = "Training Steps",
          ylabel  = "Episode Length (turns)")
    ax.legend()


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Generate training charts from TensorBoard logs.")
    p.add_argument("--logdir", default="tensorboard_logs",
                   help="TensorBoard log directory (default: tensorboard_logs)")
    p.add_argument("--outdir", default="charts",
                   help="Output directory for PNG files (default: charts)")
    p.add_argument("--show", action="store_true",
                   help="Open chart windows in addition to saving")
    args = p.parse_args()

    if not os.path.isdir(args.logdir):
        print(f"[error] Log directory not found: {args.logdir}")
        print("  Run training first: python train.py --difficulty hard ...")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    if args.show:
        matplotlib.use("TkAgg")

    charts = [
        ("01_reward_curve",      chart_reward,      "Mean Episode Reward"),
        ("02_winrate_hard",      chart_winrate_hard, "Win Rate vs Hard Bot"),
        ("03_selfplay_winrates", chart_selfplay,     "Self-Play Win Rates"),
        ("04_episode_length",    chart_ep_length,    "Average Episode Length"),
    ]

    # ── Individual charts ──────────────────────────────────────
    for fname, fn, _ in charts:
        fig, ax = plt.subplots(figsize=(11, 5))
        fn(ax, args.logdir)
        fig.tight_layout(pad=1.8)
        out = os.path.join(args.outdir, f"{fname}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[chart] Saved → {out}")
        if args.show:
            plt.show()
        plt.close(fig)

    # ── Combined 2×2 dashboard ─────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    fig.suptitle("Blast Battles — RL Training Results",
                 color=C_TEXT, fontsize=15, fontweight="bold", y=1.01)

    chart_reward(     axes[0, 0], args.logdir)
    chart_winrate_hard(axes[0, 1], args.logdir)
    chart_selfplay(   axes[1, 0], args.logdir)
    chart_ep_length(  axes[1, 1], args.logdir)

    fig.tight_layout(pad=2.5)
    out = os.path.join(args.outdir, "combined_dashboard.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[chart] Dashboard → {out}")
    if args.show:
        plt.show()
    plt.close(fig)

    print(f"\nAll charts saved to ./{args.outdir}/")
    print(f"Live view during training:  tensorboard --logdir {args.logdir}")


if __name__ == "__main__":
    main()
