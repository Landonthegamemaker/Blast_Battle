"""
visualize_training.py
=====================
Generate training charts from TensorBoard logs produced by train.py.

Each data point is assigned to its training stage using the stage_index tag,
so the three stages (hard training, fine-tune, self-play) never overlap.
Self-play x-values are shifted to continue sequentially after fine-tune.

Charts produced
---------------
  charts/01_reward_curve.png      — Mean episode reward, all three stages
  charts/02_winrate_hard.png      — Win rate vs hard bot, hard + fine-tune
  charts/03_selfplay_winrates.png — Self-play: win rate vs self and vs hard
  charts/04_episode_length.png    — Average episode length, all three stages
  charts/combined_dashboard.png   — All four charts in a 2×2 grid

Usage
-----
  python visualize_training.py
  python visualize_training.py --logdir tensorboard_logs --outdir charts
  python visualize_training.py --show

Requirements
------------
  pip install tensorboard matplotlib numpy
"""

import argparse, os, sys, glob, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Colours ───────────────────────────────────────────────────
C_STAGE0  = "#44ff88"   # green  — hard training
C_STAGE1  = "#e8b84b"   # gold   — fine-tune
C_STAGE2  = "#4ab8ff"   # blue   — self-play
C_WR_SELF = "#c44bff"   # purple — win rate vs self-play opponent
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
    "legend.fontsize":   9,
    "legend.labelcolor": C_TEXT,
})


# ══════════════════════════════════════════════════════════════
#  TENSORBOARD READER
# ══════════════════════════════════════════════════════════════

def load_all_events(logdir: str) -> dict:
    """Load all scalar events from all event files under logdir."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator, STORE_EVERYTHING_SIZE_GUIDANCE)
    except ImportError:
        print("[error] pip install tensorboard")
        sys.exit(1)

    all_events = collections.defaultdict(list)
    for fpath in glob.glob(
            os.path.join(logdir, "**", "events.out.*"), recursive=True):
        ea = EventAccumulator(fpath,
                              size_guidance=STORE_EVERYTHING_SIZE_GUIDANCE)
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            all_events[tag].extend(
                (ev.step, ev.value) for ev in ea.Scalars(tag))
    return all_events


def deduped(events: dict, tag: str) -> list[tuple[int, float]]:
    """Return sorted, deduplicated [(step, value)] for a tag."""
    pts = sorted(events.get(tag, []))
    seen = set(); out = []
    for s, v in pts:
        if s not in seen:
            seen.add(s); out.append((s, v))
    return out


def build_stage_map(events: dict) -> list[tuple[int, int]]:
    """Return sorted [(step, stage_index)] from train/stage_index tag."""
    return [(s, int(round(v)))
            for s, v in deduped(events, "train/stage_index")]


def stage_at(step: int, stage_sorted: list) -> int:
    """Return the stage active at a given step."""
    stage = 0
    for s, st in stage_sorted:
        if s <= step:
            stage = st
        else:
            break
    return stage


def by_stage(events: dict, tag: str, stage: int,
             stage_sorted: list) -> list[tuple[int, float]]:
    """Return only the points for a given stage."""
    return [(s, v) for s, v in deduped(events, tag)
            if stage_at(s, stage_sorted) == stage]


# ══════════════════════════════════════════════════════════════
#  STAGE BOUNDARIES
# ══════════════════════════════════════════════════════════════

def get_boundaries(stage_sorted: list) -> dict:
    """Return start/end step for each stage, plus self-play offset."""
    bounds = {}
    for stage in (0, 1, 2):
        pts = [s for s, st in stage_sorted if st == stage]
        if pts:
            bounds[stage] = (min(pts), max(pts))

    # Self-play x-offset: shift stage 2 to start right after stage 1 ends
    if 1 in bounds and 2 in bounds:
        bounds["sp_offset"] = bounds[1][1] - bounds[2][0]
    else:
        bounds["sp_offset"] = 0

    return bounds


def normalize_sp(pts: list, sp_offset: int) -> list:
    """Shift self-play steps so they continue after fine-tune."""
    return [(s + sp_offset, v) for s, v in pts]


# ══════════════════════════════════════════════════════════════
#  PLOT HELPERS
# ══════════════════════════════════════════════════════════════

def ema(values: np.ndarray, span: int = 7) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def fmt_steps(ax):
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k"))


def fmt_pct(ax):
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))


def base(ax, title, xlabel, ylabel):
    ax.set_title(title, color=C_TEXT, fontsize=12, pad=10, fontweight="bold")
    ax.set_xlabel(xlabel, color=C_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=C_MUTED, fontsize=9)
    ax.grid(True)
    ax.spines[:].set_color(C_BORDER)
    fmt_steps(ax)


def line(ax, pts, color, label, raw_alpha=0.2):
    if not pts:
        return
    s, v = zip(*pts)
    s, v = np.array(s, dtype=float), np.array(v, dtype=float)
    ax.plot(s, v, color=color, alpha=raw_alpha, linewidth=1)
    ax.plot(s, ema(v), color=color, label=label, linewidth=2)
    ax.fill_between(s, ema(v), alpha=0.07, color=color)


def divider(ax, x, label):
    ax.axvline(x, color=C_MUTED, linestyle=":", linewidth=1.2, alpha=0.7)
    ylim = ax.get_ylim()
    ax.text(x, ylim[0] + (ylim[1] - ylim[0]) * 0.96, label,
            color=C_MUTED, fontsize=7, ha="center", va="top",
            bbox=dict(facecolor=C_SURFACE, edgecolor="none",
                      alpha=0.8, pad=2))


def no_data(ax, msg):
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, color=C_MUTED,
            fontsize=9, linespacing=1.6)


# ══════════════════════════════════════════════════════════════
#  CHART FUNCTIONS
# ══════════════════════════════════════════════════════════════

def chart_reward(ax, ev, ss, bounds):
    """
    Mean episode reward — all three stages as coloured segments.
    Green = hard training, Gold = fine-tune, Blue = self-play.
    Self-play is shifted to continue sequentially after fine-tune.
    """
    sp_offset = bounds.get("sp_offset", 0)
    s0 = by_stage(ev, "rollout/ep_rew_mean", 0, ss)
    s1 = by_stage(ev, "rollout/ep_rew_mean", 1, ss)
    s2 = normalize_sp(by_stage(ev, "rollout/ep_rew_mean", 2, ss), sp_offset)

    found = any([s0, s1, s2])
    line(ax, s0, C_STAGE0, "Hard training")
    line(ax, s1, C_STAGE1, "Fine-tune")
    line(ax, s2, C_STAGE2, "Self-play")

    if 0 in bounds and 1 in bounds:
        divider(ax, bounds[0][1], "Fine-tune →")
        divider(ax, bounds[1][1], "Self-play →")

    if not found:
        no_data(ax, "No reward data.\nRun training first.")

    base(ax, "Mean Episode Reward — All Training Stages",
         "Training Steps", "Mean Reward")
    ax.legend()


def chart_winrate_hard(ax, ev, ss, bounds):
    """
    Win rate vs hard heuristic bot — hard training and fine-tune only.
    Self-play is excluded because it trains against itself, not the hard bot.
    Dashed red line marks the 50% target.
    """
    s0 = by_stage(ev, "eval/win_rate_hard", 0, ss)
    s1 = by_stage(ev, "eval/win_rate_hard", 1, ss)

    found = any([s0, s1])
    line(ax, s0, C_STAGE0, "Hard training")
    line(ax, s1, C_STAGE1, "Fine-tune")

    if 0 in bounds:
        divider(ax, bounds[0][1], "Fine-tune →")

    ax.axhline(0.50, color=C_GATE, linestyle="--",
               linewidth=1.5, label="50% target", zorder=3)
    ax.set_ylim(0, 1.0)
    fmt_pct(ax)

    if not found:
        no_data(ax, "No win rate data.\nRetrain with updated train.py.")

    base(ax, "Win Rate vs Hard Bot — Hard Training & Fine-Tune",
         "Training Steps", "Win Rate")
    ax.legend()


def chart_selfplay(ax, ev, ss, bounds):
    """
    Self-play stage only — two lines on the same axes:
      Purple: win rate vs the frozen self-play opponent
      Blue:   win rate vs the hard heuristic bot

    X-axis resets to zero (self-play training steps, not global steps).
    The gap between the two lines shows whether the agent is over-fitting
    to its training opponent or generalising to the hard bot.
    """
    raw_self = by_stage(ev, "eval/win_rate_selfplay_self", 2, ss)
    raw_hard = by_stage(ev, "eval/win_rate_selfplay_hard", 2, ss)

    # Reset x-axis to start from 0
    if raw_self:
        sp_start = min(s for s, _ in raw_self)
        sp_self = [(s - sp_start, v) for s, v in raw_self]
        sp_hard = [(s - sp_start, v) for s, v in raw_hard]
    else:
        sp_self = sp_hard = []

    line(ax, sp_self, C_WR_SELF, "vs Frozen self-play opponent")
    line(ax, sp_hard, C_STAGE2,  "vs Hard heuristic bot")

    ax.axhline(0.50, color=C_GATE, linestyle="--",
               linewidth=1.5, label="50% reference", zorder=3)
    ax.set_ylim(0, 1.05)
    fmt_pct(ax)

    if not sp_self:
        no_data(ax, "No self-play data.\n"
                    "Run: python train.py --selfplay\n"
                    "     --checkpoint checkpoints\\hard\\best_winrate_model.zip")

    base(ax, "Self-Play Win Rates",
         "Self-Play Training Steps", "Win Rate")
    ax.legend()

    return sp_self, sp_hard  # return for reuse in dashboard


def chart_ep_length(ax, ev, ss, bounds):
    """
    Average episode length in turns — all three stages.
    Rising length means the agent survives longer and plays more
    competitively. Self-play games are typically longer since both
    sides are competent.
    """
    sp_offset = bounds.get("sp_offset", 0)
    s0 = by_stage(ev, "rollout/ep_len_mean", 0, ss)
    s1 = by_stage(ev, "rollout/ep_len_mean", 1, ss)
    s2 = normalize_sp(by_stage(ev, "rollout/ep_len_mean", 2, ss), sp_offset)

    found = any([s0, s1, s2])
    line(ax, s0, C_STAGE0, "Hard training")
    line(ax, s1, C_STAGE1, "Fine-tune")
    line(ax, s2, C_STAGE2, "Self-play")

    if 0 in bounds and 1 in bounds:
        divider(ax, bounds[0][1], "Fine-tune →")
        divider(ax, bounds[1][1], "Self-play →")

    if not found:
        no_data(ax, "No episode length data.\nRun training first.")

    base(ax, "Average Episode Length — All Training Stages",
         "Training Steps", "Episode Length (turns)")
    ax.legend()


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Generate Blast Battles training charts from TensorBoard logs.")
    p.add_argument("--logdir", default="tensorboard_logs")
    p.add_argument("--outdir", default="charts")
    p.add_argument("--show",   action="store_true",
                   help="Open chart windows in addition to saving")
    args = p.parse_args()

    if not os.path.isdir(args.logdir):
        print(f"[error] Log directory not found: {args.logdir}")
        print("  Run training first: python train.py --difficulty hard ...")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    if args.show:
        matplotlib.use("TkAgg")

    print(f"[load] Reading logs from {args.logdir} ...")
    ev = load_all_events(args.logdir)
    ss = build_stage_map(ev)
    bounds = get_boundaries(ss)

    stages_found = sorted(set(st for _, st in ss))
    print(f"[load] Stages found: {stages_found}")
    for stage, label in [(0, "Hard training"), (1, "Fine-tune"), (2, "Self-play")]:
        if stage in bounds:
            start, end = bounds[stage]
            print(f"  Stage {stage} ({label}): {start:>8,} → {end:>8,} steps")

    def save(fig, name):
        path = os.path.join(args.outdir, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[chart] Saved → {path}")
        if args.show:
            plt.show()
        plt.close(fig)

    # ── Individual charts ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    chart_reward(ax, ev, ss, bounds)
    fig.tight_layout(pad=1.8); save(fig, "01_reward_curve")

    fig, ax = plt.subplots(figsize=(11, 5))
    chart_winrate_hard(ax, ev, ss, bounds)
    fig.tight_layout(pad=1.8); save(fig, "02_winrate_hard")

    fig, ax = plt.subplots(figsize=(11, 5))
    sp_self, sp_hard = chart_selfplay(ax, ev, ss, bounds)
    fig.tight_layout(pad=1.8); save(fig, "03_selfplay_winrates")

    fig, ax = plt.subplots(figsize=(13, 5))
    chart_ep_length(ax, ev, ss, bounds)
    fig.tight_layout(pad=1.8); save(fig, "04_episode_length")

    # ── Combined 2×2 dashboard ─────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(22, 10))
    fig.suptitle("Blast Battles — RL Training Results",
                 color=C_TEXT, fontsize=15, fontweight="bold", y=1.01)

    chart_reward(    axes[0, 0], ev, ss, bounds)
    chart_winrate_hard(axes[0, 1], ev, ss, bounds)

    # Reuse pre-computed self-play series for dashboard
    ax = axes[1, 0]
    line(ax, sp_self, C_WR_SELF, "vs Self-play opponent")
    line(ax, sp_hard, C_STAGE2,  "vs Hard heuristic bot")
    ax.axhline(0.50, color=C_GATE, linestyle="--",
               linewidth=1.5, label="50% reference", zorder=3)
    ax.set_ylim(0, 1.05); fmt_pct(ax)
    base(ax, "Self-Play Win Rates", "Self-Play Training Steps", "Win Rate")
    ax.legend()

    chart_ep_length( axes[1, 1], ev, ss, bounds)

    fig.tight_layout(pad=2.5)
    save(fig, "combined_dashboard")

    print(f"\nAll charts saved to ./{args.outdir}/")
    print(f"Live TensorBoard:  tensorboard --logdir {args.logdir}")


if __name__ == "__main__":
    main()
