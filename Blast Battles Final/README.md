# Blast Battles

A weapons-based card game with a trained RL agent that achieves **50.6% win rate** against the hard bot.

---

## Playing the Game

> **Important:** The Impossible AI loads `blast_battles_policy.onnx` via `fetch`, which requires HTTP.
> Open a terminal in the project folder and run:
>
> ```bash
> python -m http.server 8080
> ```
> Then open **http://localhost:8080/blast-battles.html** in your browser.

### Difficulty Levels

| Difficulty | Bot Behavior |
|---|---|
| Easy | Skips 75% of phases, random card selection |
| Medium | Skips 50% of phases, random card selection |
| Hard | Never skips, always picks optimal card |
| Impossible | Trained RL agent (PPO, 50.6% vs hard bot) |

### Rules

- 4 combat phases per turn: **Fast → Medium → Slow → Charged**
- Movement phase precedes each combat round
- Higher character speed acts first within each phase
- 50-turn limit — highest HP wins on time
- Cards in play are visible to both sides; hand cards are hidden

---

## RL Training

### Setup

```bash
pip install -r requirements.txt
```

### Reproduce the 50%+ result

Training skips the easy/medium curriculum and goes straight to hard, then
fine-tunes from the best checkpoint.

```bash
# Step 1 — Initial hard training (~25 min on RTX 5090)
python train.py --difficulty hard --timesteps 50000 --chunks 5

# Step 2 — Fine-tune from best checkpoint (lower LR + entropy)
python train.py --difficulty hard --timesteps 50000 --chunks 5 \
                --checkpoint checkpoints/hard/best_model.zip
```

After each run the **best** checkpoint (highest eval reward, not the final weights)
is automatically exported to `blast_battles_policy.onnx`.

### Other commands

```bash
# Export a specific checkpoint to ONNX manually
python train.py --export-onnx checkpoints/hard/best_model.zip

# Evaluate a saved model (200 episodes vs easy / medium / hard)
python train.py --eval-only checkpoints/hard/best_model.zip

# Quick environment sanity check (no ML deps beyond gymnasium)
python train.py --sanity
```

### Visualise training progress

```bash
python visualize_training.py            # reads tensorboard_logs/, writes charts/
python visualize_training.py --show     # also opens matplotlib windows
tensorboard --logdir tensorboard_logs   # live interactive view during training
```

---

## Architecture

### Environment (`blast_battles_env.py`)

| | |
|---|---|
| **Observation** | 101 features — character stats, hand/in-play cards, positions, arena effects, phase, helpers |
| **Actions** | 14 — skip, 4 hand slots, 9 arena movement nodes |
| **Reward** | Per-turn HP delta + terminal HP differential |
| **Episode end** | Either character reaches 0 HP, or turn > 50 |

### Training (`train.py`)

| | |
|---|---|
| **Algorithm** | MaskablePPO (sb3-contrib) |
| **Network** | MLP 101 → 256 → 256 → 14 (pi + vf separate heads) |
| **Initial run** | LR=3e-4, ent_coef=0.25, direct vs Hard bot |
| **Fine-tune** | LR=5e-5, ent_coef=0.05, starting from best_model.zip |
| **ONNX export** | Always exports `best_model.zip`, not the final model |

### Results

| Stage | Gate | Achieved |
|---|---|---|
| Hard (initial) | — | ~45–48% |
| Hard (fine-tuned) | 50% | **50.6%** |

---

## Files

| File | Description |
|---|---|
| `blast-battles.html` | Game UI (load via HTTP for Impossible AI) |
| `blast_battles_env.py` | Gymnasium environment |
| `train.py` | MaskablePPO training pipeline |
| `visualize_training.py` | TensorBoard log → matplotlib charts |
| `requirements.txt` | Python dependencies |
| `blast_battles_policy.onnx` | Exported best model (loaded by browser via ORT Web) |
