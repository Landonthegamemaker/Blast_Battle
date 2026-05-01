"""
train.py
========
Train a PPO agent on BlastBattlesEnv using Stable Baselines 3.

Curriculum schedule (difficulty escalates as agent improves):
    Stage 1 — training   (bot skips 100%,   0% active): gate 90%
    Stage 2 — easy       (bot skips  75%,  25% active): gate 60%
    Stage 3 — medium     (bot skips  50%,  50% active): gate 45%
    Stage 4 — hard       (bot skips   0%, 100% active): gate 30% — surpass heuristic
    Stage 5 — impossible (trained agent self-play):      no gate

Usage:
    python train.py                        # full curriculum run
    python train.py --difficulty easy      # single-difficulty run
    python train.py --timesteps 50000      # quick smoke test
    python train.py --eval-only model.zip  # evaluate saved model
    python train.py --export-onnx model.zip
    python train.py --sanity               # env check, no ML deps
"""

import argparse, os, time
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback,
)
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy as maskable_evaluate_policy
from blast_battles_env import BlastBattlesEnv, OBS_DIM

# ── Config ────────────────────────────────────────────────────
CURRICULUM = [
    ("training",    50_000),
    ("easy",       200_000),
    ("medium",     200_000),
    ("semi_hard",  200_000),   # bridge: 20% skip, optimal heuristic
    ("hard",       300_000),
    ("impossible", 200_000),
]

# Win rate gates — HARD STOPS. DO NOT CHANGE WITHOUT USER INSTRUCTION.
WIN_RATE_GATES = {
    "training":   0.90,
    "easy":       0.75,   # ceiling 81%
    "medium":     0.60,   # ceiling 63%
    "semi_hard":  0.42,   # ceiling 45%
    "hard":       0.42,   # proven ceiling via direct training
    "impossible": 0.0,
}

# Max chunks per stage
MAX_CHUNKS = {
    "training":   6,
    "easy":       6,
    "medium":     6,
    "semi_hard":  6,
    "hard":       10,   # 10 × 300k = 3M steps max
    "impossible": 4,
}
# Typical run: ~15-20 min total
N_ENVS          = 8
EVAL_FREQ       = 20_000
N_EVAL_EPISODES = 50
CHECKPOINT_FREQ = 100_000
SAVE_DIR        = "checkpoints"
LOG_DIR         = "tensorboard_logs"
FINAL_MODEL     = "blast_battles_ppo"
ONNX_PATH       = "blast_battles_policy.onnx"

PPO_KWARGS = dict(
    learning_rate   = 3e-4,
    n_steps         = 1024,
    batch_size      = 512,
    n_epochs        = 10,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    ent_coef        = 0.25,
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,
    policy_kwargs   = dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),  # 256 trains 2x faster than 512
    verbose         = 1,
    tensorboard_log = LOG_DIR,
    device          = "cpu",
)

# ── Curriculum callback ───────────────────────────────────────
class CurriculumCallback(BaseCallback):
    """Logs current difficulty stage to TensorBoard."""
    DIFF_IDX = {"training": 0, "easy": 1, "medium": 2, "semi_hard": 3, "hard": 4, "impossible": 5}
    def __init__(self, difficulty, verbose=0):
        super().__init__(verbose)
        self.difficulty = difficulty
    def _on_step(self):
        self.logger.record("curriculum/stage_index", self.DIFF_IDX.get(self.difficulty, -1))
        self.logger.record("curriculum/stage", self.difficulty)
        return True

# ── Env factory ───────────────────────────────────────────────
def make_vec(difficulty, n=N_ENVS, bot_model=None):
    return make_vec_env(
        lambda: BlastBattlesEnv(difficulty=difficulty, bot_model=bot_model),
        n_envs=n
    )

# Resume curriculum — skips easy stages, focuses on hard
CURRICULUM_RESUME = [
    ("medium",     500_000),
    ("hard",       2_000_000),
    ("impossible", 500_000),
]

# ── Normalized gate evaluation ────────────────────────────────
def _evaluate_normalized(model, train_env, n_episodes, difficulty):
    """Evaluate win rate using the same VecNormalize stats as training.
    This ensures gate checks see the same observation scale the model was trained on."""
    env = BlastBattlesEnv(difficulty=difficulty)
    wins = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        # Normalize obs using training env stats
        obs_n = train_env.normalize_obs(obs)
        done = False
        while not done:
            masks = env.action_masks()
            action, _ = model.predict(obs_n, deterministic=True, action_masks=masks)
            obs, r, term, trunc, info = env.step(int(action))
            obs_n = train_env.normalize_obs(obs)
            done = term or trunc
        if info.get("winner") == "player":
            wins += 1
    return wins / n_episodes

# ── Curriculum training ───────────────────────────────────────
def train_curriculum():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,  exist_ok=True)
    model, t0 = None, time.time()

    for i, (difficulty, stage_steps) in enumerate(CURRICULUM):
        gate = WIN_RATE_GATES.get(difficulty, 0.0)
        max_chunks = MAX_CHUNKS.get(difficulty, 4)
        max_steps = stage_steps * max_chunks

        print(f"\n{'='*56}")
        print(f"  STAGE {i+1}/{len(CURRICULUM)} — {difficulty.upper()}  "
              f"({stage_steps:,} steps/chunk × {max_chunks} max, gate={gate*100:.0f}%)")
        print(f"{'='*56}\n")

        # For impossible stage — load the saved hard model as self-play bot
        impossible_bot = None
        if difficulty == "impossible":
            impossible_bot_path = os.path.join(SAVE_DIR, "impossible_bot.zip")
            if os.path.exists(impossible_bot_path):
                impossible_bot = MaskablePPO.load(impossible_bot_path)
                print(f"[impossible] Self-play bot loaded → training against trained agent")
            else:
                print("[impossible] No self-play bot found — using hard heuristic as fallback")

        train_env = VecNormalize(make_vec(difficulty, bot_model=impossible_bot),
                                 norm_obs=True, norm_reward=True, clip_obs=10.0)
        eval_env  = VecNormalize(make_vec(difficulty, 4, bot_model=impossible_bot),
                                 norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

        if model is None:
            model = MaskablePPO("MlpPolicy", train_env, **PPO_KWARGS)
        else:
            model.set_env(train_env)

        steps_trained = 0
        while steps_trained < max_steps:
            chunk = min(stage_steps, max_steps - steps_trained)

            cbs = CallbackList([
                CheckpointCallback(CHECKPOINT_FREQ // N_ENVS, SAVE_DIR,
                                   f"ppo_{difficulty}", save_vecnormalize=True),
                EvalCallback(eval_env,
                             best_model_save_path=os.path.join(SAVE_DIR, difficulty),
                             log_path=os.path.join(LOG_DIR, difficulty),
                             eval_freq=EVAL_FREQ // N_ENVS,
                             n_eval_episodes=N_EVAL_EPISODES,
                             deterministic=True),
                CurriculumCallback(difficulty),
            ])
            model.learn(chunk, callback=cbs, progress_bar=True,
                        reset_num_timesteps=False, tb_log_name="BlastBattles")
            steps_trained += chunk

            # Gate check — must use normalized observations matching training
            # Raw evaluate() won't work since model was trained with VecNormalize
            wr = _evaluate_normalized(model, train_env, 500, difficulty)
            print(f"  [{difficulty}] Win rate: {wr*100:.1f}% (gate: {gate*100:.0f}%)")

            if wr >= gate or steps_trained >= max_steps:
                if wr >= gate:
                    print(f"  ✓ Gate passed at {steps_trained:,} steps")
                else:
                    print(f"\n{'='*56}")
                    print(f"  ✗ GATE NOT MET — {difficulty.upper()} stage halted")
                    print(f"  Required: {gate*100:.0f}%  |  Achieved: {wr*100:.1f}%")
                    print(f"  Adjust reward incentives before retraining.")
                    print(f"{'='*56}\n")
                    model.save(f"{FINAL_MODEL}_halted_{difficulty}")
                    return model
                break
            else:
                print(f"  Gate not met — extending {stage_steps:,} more steps ...")

        spath = os.path.join(SAVE_DIR, f"stage{i+1}_{difficulty}")
        model.save(spath); train_env.save(f"{spath}_vecnorm.pkl")
        print(f"[stage {i+1}] Saved → {spath}.zip")

        # After hard stage passes, save model as the impossible bot (self-play)
        if difficulty == "hard":
            impossible_bot_path = os.path.join(SAVE_DIR, "impossible_bot")
            model.save(impossible_bot_path)
            print(f"[stage {i+1}] Saved impossible (self-play) bot → {impossible_bot_path}.zip")

    print(f"\n[train] Curriculum done in {(time.time()-t0)/60:.1f} min")
    model.save(FINAL_MODEL)
    return model

# ── Resume training ───────────────────────────────────────────
def train_resume(model_path):
    """Continue training from a saved model using the hard-focus curriculum."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,  exist_ok=True)
    curriculum = CURRICULUM_RESUME
    print(f"[resume] Loading model from {model_path} ...")
    t0 = time.time()
    model = None
    for i, (difficulty, stage_steps) in enumerate(curriculum):
        print(f"\n{'='*56}")
        print(f"  RESUME STAGE {i+1}/{len(curriculum)} — {difficulty.upper()}  ({stage_steps:,} steps)")
        print(f"{'='*56}\n")

        train_env = VecNormalize(make_vec(difficulty),
                                 norm_obs=True, norm_reward=True, clip_obs=10.0)
        eval_env  = VecNormalize(make_vec(difficulty, 4),
                                 norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

        if model is None:
            model = MaskablePPO.load(model_path, env=train_env)
            print(f"[resume] Model loaded. Continuing training...")
        else:
            model.set_env(train_env)

        cbs = CallbackList([
            CheckpointCallback(CHECKPOINT_FREQ // N_ENVS, SAVE_DIR,
                               f"resume_{difficulty}", save_vecnormalize=True),
            EvalCallback(eval_env,
                         best_model_save_path=os.path.join(SAVE_DIR, f"resume_{difficulty}"),
                         log_path=os.path.join(LOG_DIR, f"resume_{difficulty}"),
                         eval_freq=EVAL_FREQ // N_ENVS,
                         n_eval_episodes=N_EVAL_EPISODES,
                         deterministic=True),
            CurriculumCallback(difficulty),
        ])

        model.learn(stage_steps, callback=cbs, progress_bar=True,
                    reset_num_timesteps=False, tb_log_name="BlastBattles")

        spath = os.path.join(SAVE_DIR, f"resume_stage{i+1}_{difficulty}")
        model.save(spath); train_env.save(f"{spath}_vecnorm.pkl")
        wr = evaluate(model, 100, difficulty, silent=True)
        print(f"[resume stage {i+1}] Win rate vs {difficulty}: {wr*100:.1f}%  → saved {spath}.zip")

    print(f"\n[resume] Done in {(time.time()-t0)/60:.1f} min")
    model.save("blast_battles_ppo_resumed")
    return model
def train_single(difficulty, timesteps, checkpoint=None, n_chunks=5):
    """
    Train directly against one difficulty, reporting win rate every chunk.
    Optionally resume from a checkpoint with fine-tuning hyperparameters.
    Always runs exactly n_chunks then stops (regardless of gate).
    """
    os.makedirs(SAVE_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)

    gate       = WIN_RATE_GATES.get(difficulty, 0.0)
    max_steps  = timesteps * n_chunks
    finetune   = checkpoint is not None

    # Fine-tuning uses lower LR and entropy to stabilize near the learned peak
    kwargs = dict(PPO_KWARGS)
    if finetune:
        kwargs["learning_rate"] = 5e-5   # 6× lower — stabilises near peak
        kwargs["ent_coef"]      = 0.05   # 5× lower — exploit learned strategy

    print(f"\n{'='*56}")
    print(f"  {'FINE-TUNING' if finetune else 'DIRECT TRAINING'} vs {difficulty.upper()}")
    print(f"  {timesteps:,} steps/chunk × {n_chunks} chunks = {max_steps:,} total")
    print(f"  LR={kwargs['learning_rate']:.0e}  ent={kwargs['ent_coef']}  gate={gate*100:.0f}%")
    if finetune: print(f"  Resuming from: {checkpoint}")
    print(f"{'='*56}")
    print(f"  {'Steps':>10}  {'Win%':>6}  Chart")
    print(f"  {'-'*42}")

    train_env = VecNormalize(make_vec(difficulty), norm_obs=True, norm_reward=True, clip_obs=10.0)
    eval_env  = VecNormalize(make_vec(difficulty, 4), norm_obs=True, norm_reward=False,
                             clip_obs=10.0, training=False)

    if finetune and os.path.exists(checkpoint):
        model = MaskablePPO.load(checkpoint, env=train_env, **{k:v for k,v in kwargs.items()
                                 if k not in ('verbose','tensorboard_log','device','policy_kwargs')})
        model.learning_rate = kwargs["learning_rate"]
        model.ent_coef      = kwargs["ent_coef"]
        print(f"  Checkpoint loaded ✓")
    else:
        if finetune: print(f"  [warn] Checkpoint not found — starting fresh")
        model = MaskablePPO("MlpPolicy", train_env, **kwargs)

    best_wr = 0.0
    steps_trained = 0
    for chunk_i in range(n_chunks):
        cbs = CallbackList([
            EvalCallback(eval_env,
                         best_model_save_path=os.path.join(SAVE_DIR, difficulty),
                         log_path=os.path.join(LOG_DIR, difficulty),
                         eval_freq=max(timesteps // N_ENVS, 1),
                         n_eval_episodes=N_EVAL_EPISODES,
                         deterministic=True, verbose=0),
            CurriculumCallback(difficulty),
        ])
        model.learn(timesteps, callback=cbs, progress_bar=True,
                    reset_num_timesteps=False, tb_log_name="BlastBattles")
        steps_trained += timesteps

        wr = _evaluate_normalized(model, train_env, 500, difficulty)
        best_wr = max(best_wr, wr)
        bar  = '█' * int(wr * 20)
        note = '✓ GATE' if wr >= gate else ''
        print(f"  {steps_trained:>10,}  {wr*100:>5.1f}%  {bar} {note}")

    print(f"\n  Done. Best: {best_wr*100:.1f}%  Gate: {gate*100:.0f}%")
    model.save(f"blast_battles_direct_{difficulty}")
    return model

# ── Evaluation ────────────────────────────────────────────────
def evaluate(model, n_episodes=200, difficulty="medium", silent=False):
    if not silent: print(f"\n[eval] {n_episodes} eps vs {difficulty.upper()} ...")
    env = BlastBattlesEnv(difficulty=difficulty)
    wins = losses = draws = 0
    total_r, lengths = 0.0, []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, ep_r, steps = False, 0.0, 0
        while not done:
            # Use action masks so evaluation matches training behaviour
            masks = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
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
              f"Avg reward {total_r/n_episodes:.2f} | Avg steps {np.mean(lengths):.1f}")
    return wr

# ── ONNX export ───────────────────────────────────────────────
def export_onnx(model, path=ONNX_PATH):
    try:
        import torch, torch.nn as nn, onnx
        from onnx.external_data_helper import convert_model_to_external_data
        from blast_battles_env import OBS_DIM, N_ACTIONS

        class PolicyNet(nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.mlp_extractor = policy.mlp_extractor
                self.action_net    = policy.action_net

            def forward(self, obs):
                latent_pi, _ = self.mlp_extractor(obs)
                return self.action_net(latent_pi)

        net = PolicyNet(model.policy)
        net.eval().cpu()
        dummy = torch.zeros(1, OBS_DIM, dtype=torch.float32)

        with torch.no_grad():
            torch.onnx.export(
                net,
                dummy,
                path,
                input_names   = ["obs"],
                output_names  = ["logits"],
                dynamic_axes  = {"obs": {0: "batch"}, "logits": {0: "batch"}},
                opset_version = 17,
                export_params = True,
            )

        # Inline all external data into the single .onnx file so the browser
        # can load it without needing a companion .data file
        m = onnx.load(path)
        onnx.save_model(m, path, save_as_external_data=False)
        print(f"[onnx] Exported → {path}  (input:{OBS_DIM}  output:{N_ACTIONS}  self-contained)")
    except Exception as e:
        print(f"[onnx] Failed: {e}")

# ── Sanity check ──────────────────────────────────────────────
def sanity_check():
    print("[sanity] 5 episodes × 4 difficulties ...")
    for diff in ("training", "easy", "medium", "hard"):
        env = BlastBattlesEnv(difficulty=diff)
        for ep in range(5):
            obs, _ = env.reset(seed=ep)
            assert obs.shape == (OBS_DIM,), f"Expected ({OBS_DIM},) got {obs.shape}"
            assert np.all(np.isfinite(obs))
            done = False
            while not done:
                obs, r, term, trunc, info = env.step(env.action_space.sample())
                assert np.all(np.isfinite(obs)) and np.isfinite(r)
                done = term or trunc
        print(f"  {diff.upper():<10} OK")
    print("[sanity] All passed ✓\n")

# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps",   type=int)
    p.add_argument("--difficulty",  choices=["training","easy","medium","hard","impossible"])
    p.add_argument("--checkpoint",  type=str, default=None,
                   help="Path to saved model to fine-tune from (uses lower LR/entropy)")
    p.add_argument("--chunks",      type=int, default=5,
                   help="Number of chunks to run in train_single mode (default: 5)")
    p.add_argument("--resume",      type=str, default=None,
                   help="Path to saved model to resume training from (skips easy stages)")
    p.add_argument("--eval-only",   type=str)
    p.add_argument("--export-onnx", type=str)
    p.add_argument("--sanity",      action="store_true")
    args = p.parse_args()

    if args.sanity:
        sanity_check()
    elif args.resume:
        m = train_resume(args.resume)
        print("\n[eval] Resumed model vs all difficulties:")
        for d in ("training","easy","medium","hard"): evaluate(m, 200, d)
        hard_best = os.path.join(SAVE_DIR, "resume_hard", "best_model.zip")
        if os.path.exists(hard_best):
            print("\n[eval] Best resumed hard checkpoint:")
            m_hard = MaskablePPO.load(hard_best)
            for d in ("training","easy","medium","hard"): evaluate(m_hard, 200, d)
        export_onnx(m)
    elif args.eval_only:
        m = MaskablePPO.load(args.eval_only)
        for d in ("easy","medium","hard"): evaluate(m, 200, d)
    elif args.export_onnx:
        export_onnx(MaskablePPO.load(args.export_onnx))
    elif args.difficulty:
        sanity_check()
        m = train_single(args.difficulty, args.timesteps or 50_000,
                         checkpoint=args.checkpoint, n_chunks=args.chunks)
        print("\n[eval] Final model vs all difficulties:")
        for d in ("training","easy","medium","hard"): evaluate(m, 200, d)
        best = os.path.join(SAVE_DIR, args.difficulty, "best_model.zip")
        if os.path.exists(best):
            print(f"\n[eval] Best {args.difficulty} checkpoint:")
            try:
                m_best = MaskablePPO.load(best)
                for d in ("training","easy","medium","hard"): evaluate(m_best, 200, d)
                export_onnx(m_best)
            except ValueError:
                export_onnx(m)
        else:
            export_onnx(m)
    else:
        sanity_check()
        m = train_curriculum()
        print("\n[eval] Final model vs all difficulties:")
        for d in ("training","easy","medium","hard"): evaluate(m, 200, d)
        # Also evaluate best hard checkpoint — may outperform final model
        hard_best = os.path.join(SAVE_DIR, "hard", "best_model.zip")
        if os.path.exists(hard_best):
            print("\n[eval] Best hard checkpoint:")
            try:
                m_hard = MaskablePPO.load(hard_best)
                for d in ("training","easy","medium","hard"): evaluate(m_hard, 200, d)
                print("\n[export] Exporting best hard checkpoint as ONNX (outperforms final model)...")
                export_onnx(m_hard)
            except ValueError as e:
                if "observation shape" in str(e):
                    print(f"  [skip] Checkpoint OBS_DIM mismatch — saved with old dims, retrain to regenerate")
                    export_onnx(m)
                else:
                    raise
