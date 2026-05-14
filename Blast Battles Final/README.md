# Blast Battles

A weapons-based card game with a trained RL agent that achieves a 50.6% win rate against the hard bot.

## Playing the Game

**Important:** The Impossible AI loads `blast_battles_policy.onnx` via fetch, which requires HTTP.
Open a terminal in the project folder and run:

```bash
python -m http.server 8080
```

Then open **http://localhost:8080/blast-battles.html** in your browser.

### Difficulty Levels

| Difficulty | Bot Behavior |
|---|---|
| Easy | Skips 75% of phases, random card selection |
| Medium | Skips 50% of phases, random card selection |
| Hard | Never skips, always picks optimal card |
| Impossible | Trained RL agent (MaskablePPO, 50.6% win rate vs hard bot) |

### Rules

- 5 phases per turn: Movement then Fast, Medium, Slow, and Charged combat phases
- Higher character speed acts first within each phase
- 50-turn limit with the highest HP character winning on time
- Cards in play are visible to both sides; hand cards are hidden

## RL Training

### Setup

```bash
pip install -r requirements.txt
```

### Reproduce the 50%+ result

Training skips the easy and medium curriculum and goes straight to hard, then
fine-tunes from the best checkpoint.

```bash
# Step 1: Initial hard training (approximately 25 minutes on an RTX 5060 Ti)
python train.py --difficulty hard --timesteps 50000 --chunks 6

# Step 2: Fine-tune from best checkpoint (lower LR and entropy)
python train.py --difficulty hard --timesteps 50000 --chunks 12 \
                --checkpoint checkpoints/hard/best_winrate_model.zip
```

After each run the best checkpoint by win rate is automatically exported to
`blast_battles_policy.onnx`. This is the model the Impossible difficulty loads in the browser.

### Optional: Self-play stage

```bash
python train.py --selfplay \
                --checkpoint checkpoints/hard/best_winrate_model.zip \
                --timesteps 50000 --chunks 11
```

Note: self-play improved episode rewards but lowered win rate vs the hard bot to
roughly 45%, so the deployed model uses the Stage 2 checkpoint, not self-play.

### Other commands

```bash
# Export a specific checkpoint to ONNX manually
python train.py --export-onnx checkpoints/hard/best_winrate_model.zip

# Evaluate a saved model (200 episodes vs easy, medium, and hard)
python train.py --eval-only checkpoints/hard/best_winrate_model.zip

# Quick environment sanity check (no ML deps beyond gymnasium)
python train.py --sanity
```

### Visualize training progress

```bash
python visualize_training.py              # reads tensorboard_logs/, writes charts/
python visualize_training.py --show       # also opens matplotlib windows
tensorboard --logdir tensorboard_logs     # live interactive view during training
```

## Files

| File | Description |
|---|---|
| `blast-battles.html` | Game UI (must be served over HTTP for Impossible AI) |
| `blast_battles_env.py` | Gymnasium environment |
| `train.py` | MaskablePPO training pipeline |
| `visualize_training.py` | TensorBoard log to matplotlib charts |
| `requirements.txt` | Python dependencies |
| `blast_battles_policy.onnx` | Exported best model (loaded by browser via ORT Web) |
