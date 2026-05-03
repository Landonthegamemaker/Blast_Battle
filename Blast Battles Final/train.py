"""
train.py
========
Train a MaskablePPO agent on BlastBattlesEnv (Stable Baselines 3 + sb3-contrib).

──────────────────────────────────────────────────────────────────
Workflow
──────────────────────────────────────────────────────────────────
  # Step 1 — Initial hard training (fresh start)
  python train.py --difficulty hard --timesteps 50000 --chunks 5

  # Step 2 — Fine-tune from best checkpoint
  python train.py --difficulty hard --timesteps 50000 --chunks 10 \\
                  --checkpoint checkpoints/hard/best_winrate_model.zip

  # Step 3 — Self-play (tracks win rate vs self + vs hard bot)
  python train.py --selfplay --checkpoint checkpoints/hard/best_winrate_model.zip \\
                  --timesteps 50000 --chunks 10

  # Utilities
  python train.py --export-onnx checkpoints/hard/best_winrate_model.zip
  python train.py --eval-only   checkpoints/hard/best_winrate_model.zip
  python train.py --sanity

──────────────────────────────────────────────────────────────────
How win rate is calculated
──────────────────────────────────────────────────────────────────
  WinRateCallback fires every WINRATE_EVAL_FREQ global steps and:
    1. Creates a fresh BlastBattlesEnv at the target difficulty
    2. Runs N_WINRATE_EPISODES episodes with the model acting deterministically
    3. Normalises each observation using the live VecNormalize stats from the
       training env — this ensures the model sees the same obs scale it was
       trained on (raw obs would be out-of-distribution and depress the number)
    4. Counts episodes where info["winner"] == "player", divides by N
    5. Logs the result to TensorBoard as eval/win_rate_hard or eval/win_rate_self

  This is separate from ep_rew_mean (always logged by SB3), which is a shaped
  reward signal.  Win rate is the cleaner metric because a model can have high
  reward while still losing if it deals damage before being defeated.
──────────────────────────────────────────────────────────────────
"""

import argparse, os
import numpy as np

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback,
)
from sb3_contrib import MaskablePPO
from blast_battles_env import BlastBattlesEnv, OBS_DIM, N_ACTIONS

# ── Paths ─────────────────────────────────────────────────────
SAVE_DIR          = "checkpoints"
LOG_DIR           = "tensorboard_logs"
ONNX_HARD         = "blast_battles_policy.onnx"       # Impossible difficulty
ONNX_SELFPLAY     = "blast_battles_selfplay.onnx"     # Legendary difficulty

# ── Training config ───────────────────────────────────────────
N_ENVS            = 8
EVAL_FREQ         = 20_000
N_EVAL_EPISODES   = 50
CHECKPOINT_FREQ   = 100_000
WINRATE_EVAL_FREQ  = 20_000  # how often callbacks fire (global steps)
N_WINRATE_EPS      = 200     # episodes per win-rate evaluation (logs + save decision)

WIN_RATE_GATE     = 0.50     # informational only

# ── PPO hyperparameters ───────────────────────────────────────
PPO_KWARGS = dict(
    learning_rate = 3e-4,
    n_steps       = 1024,
    batch_size    = 512,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.25,
    vf_coef       = 0.5,
    max_grad_norm = 0.5,
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    verbose       = 1,
    tensorboard_log = LOG_DIR,
    device        = "cpu",
)

FINETUNE_OVERRIDES = dict(
    learning_rate = 5e-5,
    ent_coef      = 0.05,
)


# ══════════════════════════════════════════════════════════════
#  CALLBACKS
# ══════════════════════════════════════════════════════════════

class StageCallback(BaseCallback):
    STAGE_IDX = {"hard": 0, "finetune": 1, "selfplay": 2}

    def __init__(self, stage_name, verbose=0):
        super().__init__(verbose)
        self.stage_name = stage_name

    def _on_step(self):
        self.logger.record("train/stage_index",
                           self.STAGE_IDX.get(self.stage_name, -1))
        return True


class WinRateCallback(BaseCallback):
    """
    Evaluates win rate every eval_freq steps using normalised observations.
    Logs to TensorBoard as eval/win_rate_{tag} AND saves the model if a
    new peak is reached (tag='hard' only). Both happen from the same
    N_WINRATE_EPS (200) episode batch — no redundant evaluation.
    """
    def __init__(self, train_env, save_path, difficulty="hard",
                 opponent=None, eval_freq=WINRATE_EVAL_FREQ,
                 n_episodes=N_WINRATE_EPS, tag="hard", verbose=0):
        super().__init__(verbose)
        self.train_env  = train_env
        self.save_path  = save_path
        self.difficulty = difficulty
        self.opponent   = opponent
        self.eval_freq  = eval_freq
        self.n_episodes = n_episodes
        self.tag        = tag
        self.best_wr    = 0.0
        self._last_eval = 0

    def _on_step(self):
        if self.num_timesteps - self._last_eval < self.eval_freq:
            return True
        self._last_eval = self.num_timesteps

        env = BlastBattlesEnv(difficulty=self.difficulty,
                              bot_model=self.opponent)
        wins = 0
        for _ in range(self.n_episodes):
            obs, _ = env.reset()
            obs_n  = self.train_env.normalize_obs(obs)
            done   = False
            while not done:
                masks = env.action_masks()
                action, _ = self.model.predict(obs_n, deterministic=True,
                                               action_masks=masks)
                obs, _, term, trunc, info = env.step(int(action))
                obs_n = self.train_env.normalize_obs(obs)
                done  = term or trunc
            if info.get("winner") == "player":
                wins += 1

        wr = wins / self.n_episodes
        self.logger.record(f"eval/win_rate_{self.tag}", wr)
        # Save on new peak (only for the primary hard-bot metric)
        if self.tag in ("hard", "selfplay_hard") and wr > self.best_wr:
            self.best_wr = wr
            os.makedirs(self.save_path, exist_ok=True)
            path = os.path.join(self.save_path, "best_winrate_model")
            self.model.save(path)
            print(f"\n  ★ New best win rate: {wr*100:.1f}%  → {path}.zip")
        return True




# ══════════════════════════════════════════════════════════════
#  ENV FACTORIES
# ══════════════════════════════════════════════════════════════

def make_vec(difficulty, n=N_ENVS):
    return make_vec_env(
        lambda: BlastBattlesEnv(difficulty=difficulty), n_envs=n)


def make_vec_selfplay(opponent, n=N_ENVS):
    return make_vec_env(
        lambda: BlastBattlesEnv(difficulty="impossible", bot_model=opponent),
        n_envs=n)


# ══════════════════════════════════════════════════════════════
#  NORMALISED WIN-RATE (for per-chunk terminal output)
# ══════════════════════════════════════════════════════════════


def _eval_norm(model, train_env, n_episodes, difficulty, opponent=None):
    env = BlastBattlesEnv(difficulty=difficulty, bot_model=opponent)
    wins = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        obs_n  = train_env.normalize_obs(obs)
        done   = False
        while not done:
            masks = env.action_masks()
            action, _ = model.predict(obs_n, deterministic=True,
                                      action_masks=masks)
            obs, _, term, trunc, info = env.step(int(action))
            obs_n = train_env.normalize_obs(obs)
            done  = term or trunc
        if info.get("winner") == "player":
            wins += 1
    return wins / n_episodes


# ══════════════════════════════════════════════════════════════
#  TRAIN SINGLE (hard + fine-tune)
# ══════════════════════════════════════════════════════════════

def train_single(difficulty, timesteps, n_chunks=5, checkpoint=None):
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,  exist_ok=True)

    finetune   = checkpoint is not None
    stage_name = "finetune" if finetune else "hard"
    kwargs     = dict(PPO_KWARGS)
    if finetune:
        kwargs.update(FINETUNE_OVERRIDES)

    print(f"\n{'='*60}")
    print(f"  {'FINE-TUNE' if finetune else 'INITIAL TRAINING'} vs {difficulty.upper()}")
    print(f"  {timesteps:,} steps/chunk × {n_chunks} = {timesteps*n_chunks:,} total")
    print(f"  LR={kwargs['learning_rate']:.0e}  ent_coef={kwargs['ent_coef']}")
    print(f"  Win rate logged to TensorBoard every {WINRATE_EVAL_FREQ:,} steps")
    if finetune:
        print(f"  Resuming from: {checkpoint}")
    print(f"{'='*60}")

    train_env = VecNormalize(make_vec(difficulty),
                             norm_obs=True, norm_reward=True, clip_obs=10.0)
    eval_env  = VecNormalize(make_vec(difficulty, n=4),
                             norm_obs=True, norm_reward=False,
                             clip_obs=10.0, training=False)
    best_save = os.path.join(SAVE_DIR, difficulty)

    if finetune and checkpoint and os.path.exists(checkpoint):
        model = MaskablePPO.load(
            checkpoint, env=train_env,
            **{k: v for k, v in kwargs.items()
               if k not in ("verbose", "tensorboard_log", "device", "policy_kwargs")})
        model.learning_rate = kwargs["learning_rate"]
        model.ent_coef      = kwargs["ent_coef"]
        print("  Checkpoint loaded ✓\n")
    else:
        if finetune:
            print(f"  [warn] Checkpoint not found — starting fresh\n")
        model = MaskablePPO("MlpPolicy", train_env, **kwargs)

    total    = 0
    best_wr  = 0.0
    best_rew = -float('inf')  # track peak reward across all chunks

    print(f"  {'Steps':>10}  {'Win%':>6}  {'Reward':>8}  Bar")
    print(f"  {'-'*52}")

    # Instantiate once outside the loop so best_wr persists across all chunks
    win_rate_cb = WinRateCallback(
        train_env=train_env, save_path=best_save,
        difficulty="hard", tag="hard",
        eval_freq=WINRATE_EVAL_FREQ)

    for _ in range(n_chunks):
        cbs = CallbackList([
            CheckpointCallback(
                save_freq=max(CHECKPOINT_FREQ // N_ENVS, 1),
                save_path=SAVE_DIR, name_prefix=f"ppo_{stage_name}"),
            EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(SAVE_DIR, "_eval_tmp"),
                log_path=os.path.join(LOG_DIR, stage_name),
                eval_freq=max(EVAL_FREQ // N_ENVS, 1),
                n_eval_episodes=N_EVAL_EPISODES,
                deterministic=True, verbose=0),
            win_rate_cb,
            StageCallback(stage_name),
        ])
        model.learn(timesteps, callback=cbs, progress_bar=True,
                    reset_num_timesteps=False, tb_log_name="BlastBattles")
        total += timesteps

        # Win rate (displayed each chunk)
        wr      = _eval_norm(model, train_env, 500, "hard")
        best_wr = max(best_wr, wr)
        bar     = "█" * int(wr * 20) + "░" * (20 - int(wr * 20))
        gate    = "✓ GATE" if wr >= WIN_RATE_GATE else ""

        # Reward from rollout buffer — check if new peak this chunk
        buf_rew  = [ep["r"] for ep in model.ep_info_buffer] if model.ep_info_buffer else [0]
        chunk_rew = float(np.mean(buf_rew))
        new_best  = chunk_rew > best_rew
        if new_best:
            best_rew = chunk_rew
        star = "  ★ new best reward" if new_best else ""

        print(f"  {total:>10,}  {wr*100:>5.1f}%  {chunk_rew:>8.2f}  {bar} {gate}{star}")

    print(f"\n  Done. Best win rate: {best_wr*100:.1f}%  Best reward: {best_rew:.2f}")

    best_winrate_zip = os.path.join(best_save, "best_winrate_model.zip")
    if os.path.exists(best_winrate_zip):
        try:
            best = MaskablePPO.load(best_winrate_zip)
            wr_b = _eval_norm(best, train_env, 200, "hard")
            print(f"[export] best_winrate_model.zip → {wr_b*100:.1f}% vs Hard")
            export_onnx(best, ONNX_HARD)
        except Exception as e:
            print(f"[export] Could not load checkpoint ({e}) — exporting current model")
            export_onnx(model, ONNX_HARD)
    else:
        print("[export] No best_winrate_model.zip found — exporting current model")
        export_onnx(model, ONNX_HARD)

    return model, train_env


# ══════════════════════════════════════════════════════════════
#  SELF-PLAY TRAINING
# ══════════════════════════════════════════════════════════════

def train_selfplay(checkpoint, timesteps=50_000, n_chunks=10):
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,  exist_ok=True)

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print(f"\n{'='*60}")
    print(f"  SELF-PLAY TRAINING")
    print(f"  Opponent (frozen): {checkpoint}")
    print(f"  {timesteps:,} steps/chunk × {n_chunks} = {timesteps*n_chunks:,} total")
    print(f"  LR={FINETUNE_OVERRIDES['learning_rate']:.0e}  "
          f"ent_coef={FINETUNE_OVERRIDES['ent_coef']}")
    print(f"  Tracking: win_rate_self + win_rate_hard → TensorBoard")
    print(f"{'='*60}")

    opponent    = MaskablePPO.load(checkpoint)
    opp_vs_hard = evaluate(opponent, 200, "hard", silent=True)
    print(f"\n[selfplay] Opponent baseline vs Hard: {opp_vs_hard*100:.1f}%\n")

    train_env = VecNormalize(make_vec_selfplay(opponent),
                             norm_obs=True, norm_reward=True, clip_obs=10.0)
    eval_env  = VecNormalize(make_vec_selfplay(opponent, n=4),
                             norm_obs=True, norm_reward=False,
                             clip_obs=10.0, training=False)

    kwargs = dict(PPO_KWARGS)
    kwargs.update(FINETUNE_OVERRIDES)

    model = MaskablePPO.load(
        checkpoint, env=train_env,
        **{k: v for k, v in kwargs.items()
           if k not in ("verbose", "tensorboard_log", "device", "policy_kwargs")})
    model.learning_rate = kwargs["learning_rate"]
    model.ent_coef      = kwargs["ent_coef"]
    print(f"[selfplay] Training agent loaded ✓\n")

    best_save    = os.path.join(SAVE_DIR, "selfplay")
    best_wr_self = 0.0
    best_rew     = -float('inf')
    total        = 0

    print(f"  {'Steps':>10}  {'vs Self':>7}  {'vs Hard':>7}  {'Reward':>8}  Bar (vs self)")
    print(f"  {'-'*62}")

    # Instantiate once outside the loop so best_wr persists across all chunks
    wr_self_cb = WinRateCallback(
        train_env=train_env, save_path=best_save,
        difficulty="impossible", opponent=opponent,
        tag="selfplay_self", eval_freq=WINRATE_EVAL_FREQ)
    wr_hard_cb = WinRateCallback(
        train_env=train_env, save_path=best_save,
        difficulty="hard", tag="selfplay_hard",
        eval_freq=WINRATE_EVAL_FREQ)

    for _ in range(n_chunks):
        cbs = CallbackList([
            CheckpointCallback(
                save_freq=max(CHECKPOINT_FREQ // N_ENVS, 1),
                save_path=SAVE_DIR, name_prefix="ppo_selfplay"),
            EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(SAVE_DIR, "_eval_tmp"),
                log_path=os.path.join(LOG_DIR, "selfplay"),
                eval_freq=max(EVAL_FREQ // N_ENVS, 1),
                n_eval_episodes=N_EVAL_EPISODES,
                deterministic=True, verbose=0),
            wr_self_cb,
            wr_hard_cb,
            StageCallback("selfplay"),
        ])
        model.learn(timesteps, callback=cbs, progress_bar=True,
                    reset_num_timesteps=False, tb_log_name="BlastBattles")
        total += timesteps

        wr_self      = _eval_norm(model, train_env, 300, "impossible", opponent)
        wr_hard      = evaluate(model, 100, "hard", silent=True)
        best_wr_self = max(best_wr_self, wr_self)
        bar          = "█" * int(wr_self * 20) + "░" * (20 - int(wr_self * 20))

        buf_rew   = [ep["r"] for ep in model.ep_info_buffer] if model.ep_info_buffer else [0]
        chunk_rew = float(np.mean(buf_rew))
        new_best  = chunk_rew > best_rew
        if new_best:
            best_rew = chunk_rew
        star = "  ★" if new_best else ""

        print(f"  {total:>10,}  {wr_self*100:>6.1f}%  {wr_hard*100:>6.1f}%  {chunk_rew:>8.2f}  {bar}{star}")

    print(f"\n  Done. Best vs self: {best_wr_self*100:.1f}%  Best reward: {best_rew:.2f}")

    best_winrate_zip = os.path.join(best_save, "best_winrate_model.zip")
    export_target = model
    if os.path.exists(best_winrate_zip):
        try:
            export_target = MaskablePPO.load(best_winrate_zip)
            print(f"[selfplay] best_winrate_model.zip loaded ✓")
        except Exception as e:
            print(f"[selfplay] Could not load checkpoint ({e})")

    compare_models(export_target, opponent, 300, train_env)
    export_onnx(export_target, ONNX_SELFPLAY)
    print(f"\n[selfplay] Hard model → {ONNX_HARD} (Impossible difficulty)")
    print(f"[selfplay] Self-play  → {ONNX_SELFPLAY} (Legendary difficulty)")

    return export_target


# ══════════════════════════════════════════════════════════════
#  COMPARISON TABLE
# ══════════════════════════════════════════════════════════════

def compare_models(selfplay_model, hard_model, n_episodes=300, train_env=None):
    print(f"\n{'='*60}")
    print(f"  MODEL COMPARISON  ({n_episodes} episodes each)")
    print(f"{'='*60}")
    print(f"  {'Opponent':<30}  {'Hard-trained':>12}  {'Self-play':>10}")
    print(f"  {'-'*56}")

    for diff in ("hard", "medium"):
        wh = evaluate(hard_model,     n_episodes, diff, silent=True)
        ws = evaluate(selfplay_model, n_episodes, diff, silent=True)
        print(f"  {diff.capitalize()+' heuristic bot':<30}  "
              f"{wh*100:>11.1f}%  {ws*100:>9.1f}%")

    print(f"\n  Head-to-head (self-play vs hard-trained):")
    env = BlastBattlesEnv(difficulty="impossible", bot_model=hard_model)
    wins = losses = draws = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        while not done:
            masks = env.action_masks()
            action, _ = selfplay_model.predict(obs, deterministic=True,
                                               action_masks=masks)
            obs, _, term, trunc, info = env.step(int(action))
            done = term or trunc
        w = info.get("winner")
        if w == "player": wins += 1
        elif w == "bot":  losses += 1
        else:             draws += 1

    print(f"  Self-play wins: {wins/n_episodes*100:.1f}%  |  "
          f"Hard-trained wins: {losses/n_episodes*100:.1f}%  |  "
          f"Draws: {draws/n_episodes*100:.1f}%")

    delta = (evaluate(selfplay_model, n_episodes, "hard", silent=True)
           - evaluate(hard_model,     n_episodes, "hard", silent=True))
    verdict = ("✓ Self-play IMPROVED vs hard bot"        if delta >  0.02 else
               "✗ Self-play did NOT improve vs hard bot" if delta < -0.02 else
               "≈ Self-play had NO CLEAR EFFECT")
    print(f"\n{'='*60}")
    print(f"  {verdict}  (Δ {delta*100:+.1f}%)")
    print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════

def evaluate(model, n_episodes=200, difficulty="hard", silent=False):
    if not silent:
        print(f"\n[eval] {n_episodes} eps vs {difficulty.upper()} ...")
    env = BlastBattlesEnv(difficulty=difficulty)
    wins = losses = draws = 0
    total_r, lengths = 0.0, []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False; ep_r = 0; steps = 0
        while not done:
            masks = env.action_masks()
            action, _ = model.predict(obs, deterministic=True,
                                      action_masks=masks)
            obs, r, term, trunc, info = env.step(int(action))
            ep_r += r; steps += 1; done = term or trunc
        total_r += ep_r; lengths.append(steps)
        w = info.get("winner")
        if w == "player": wins += 1
        elif w == "bot":  losses += 1
        else:             draws += 1
    wr = wins / n_episodes
    if not silent:
        print(f"  Win {wr*100:.1f}% | Loss {losses/n_episodes*100:.1f}% | "
              f"Draw {draws/n_episodes*100:.1f}% | "
              f"Avg reward {total_r/n_episodes:.2f} | "
              f"Avg steps {np.mean(lengths):.1f}")
    return wr


# ══════════════════════════════════════════════════════════════
#  ONNX EXPORT
# ══════════════════════════════════════════════════════════════

def export_onnx(model, path=ONNX_HARD):
    try:
        import torch, torch.nn as nn, onnx

        class PolicyNet(nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.mlp_extractor = policy.mlp_extractor
                self.action_net    = policy.action_net

            def forward(self, obs):
                latent_pi, _ = self.mlp_extractor(obs)
                return self.action_net(latent_pi)

        net   = PolicyNet(model.policy).eval().cpu()
        dummy = torch.zeros(1, OBS_DIM, dtype=torch.float32)
        with torch.no_grad():
            torch.onnx.export(
                net, dummy, path,
                input_names=["obs"], output_names=["logits"],
                dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=17, export_params=True)

        m = onnx.load(path)
        onnx.save_model(m, path, save_as_external_data=False)
        size_mb = os.path.getsize(path) / 1e6
        print(f"[onnx] ✓ {path}  ({size_mb:.1f} MB  "
              f"obs:{OBS_DIM}  actions:{N_ACTIONS})")
    except Exception as e:
        print(f"[onnx] Export failed: {e}")


# ══════════════════════════════════════════════════════════════
#  SANITY CHECK
# ══════════════════════════════════════════════════════════════

def sanity_check():
    print("[sanity] 5 episodes × 4 difficulties ...")
    for diff in ("training", "easy", "medium", "hard"):
        env = BlastBattlesEnv(difficulty=diff)
        for ep in range(5):
            obs, _ = env.reset(seed=ep)
            assert obs.shape == (OBS_DIM,)
            assert np.all(np.isfinite(obs))
            done = False
            while not done:
                obs, r, term, trunc, info = env.step(env.action_space.sample())
                assert np.all(np.isfinite(obs)) and np.isfinite(r)
                done = term or trunc
        print(f"  {diff.upper():<10} OK")
    print("[sanity] All passed ✓\n")


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train / evaluate / export the Blast Battles RL bot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python train.py --difficulty hard --timesteps 50000 --chunks 5
  python train.py --difficulty hard --timesteps 50000 --chunks 10 \\
                  --checkpoint checkpoints/hard/best_winrate_model.zip
  python train.py --selfplay --checkpoint checkpoints/hard/best_winrate_model.zip \\
                  --timesteps 50000 --chunks 10
  python train.py --export-onnx checkpoints/hard/best_winrate_model.zip
  python train.py --eval-only   checkpoints/hard/best_winrate_model.zip
  python train.py --sanity
""")
    p.add_argument("--difficulty",
                   choices=["training", "easy", "medium", "hard"])
    p.add_argument("--timesteps",   type=int, default=50_000)
    p.add_argument("--chunks",      type=int, default=5)
    p.add_argument("--checkpoint",  type=str, default=None)
    p.add_argument("--selfplay",    action="store_true")
    p.add_argument("--eval-only",   type=str, metavar="MODEL")
    p.add_argument("--export-onnx", type=str, metavar="MODEL")
    p.add_argument("--sanity",      action="store_true")
    args = p.parse_args()

    if args.sanity:
        sanity_check()

    elif args.eval_only:
        m = MaskablePPO.load(args.eval_only)
        for d in ("easy", "medium", "hard"):
            evaluate(m, 200, d)

    elif args.export_onnx:
        export_onnx(MaskablePPO.load(args.export_onnx))

    elif args.selfplay:
        if not args.checkpoint:
            p.error("--selfplay requires --checkpoint")
        train_selfplay(args.checkpoint, args.timesteps, args.chunks)

    elif args.difficulty:
        sanity_check()
        model, train_env = train_single(args.difficulty, args.timesteps,
                                        args.chunks, args.checkpoint)

        # Evaluate the exported checkpoint, not the final model weights
        best_save        = os.path.join(SAVE_DIR, args.difficulty)
        best_winrate_zip = os.path.join(best_save, "best_winrate_model.zip")

        if os.path.exists(best_winrate_zip):
            try:
                eval_model = MaskablePPO.load(best_winrate_zip)
                print(f"\n[eval] Win rates for best_winrate_model.zip:")
                for d in ("easy", "medium", "hard"):
                    wr = _eval_norm(eval_model, train_env, 200, d)
                    print(f"  vs {d.upper():<8} {wr*100:.1f}%")
            except Exception as e:
                print(f"[eval] Could not load checkpoint ({e}) — using final model")
                for d in ("easy", "medium", "hard"):
                    wr = _eval_norm(model, train_env, 200, d)
                    print(f"  vs {d.upper():<8} {wr*100:.1f}%")
        else:
            print("\n[eval] Win rates for final model:")
            for d in ("easy", "medium", "hard"):
                wr = _eval_norm(model, train_env, 200, d)
                print(f"  vs {d.upper():<8} {wr*100:.1f}%")

    else:
        p.print_help()
