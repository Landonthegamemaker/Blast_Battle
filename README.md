# Blast Battles

A weapons-based card game with a trained RL agent that achieves 50.6% win rate against the hard bot.

## Game

Open `blast-battles.html` directly in any browser — no server needed.

### Difficulty Levels
| Difficulty | Bot Behavior |
|---|---|
| Easy | Skips 75% of phases, random card selection |
| Medium | Skips 50% of phases, random card selection |
| Hard | Never skips, always picks optimal card |
| Impossible | Trained RL agent (PPO, 50.6% vs hard bot) |

### Game Rules
- 4 combat phases per turn: **Fast → Medium → Slow → Charged**
- Movement phase precedes combat each turn
- Higher character speed acts first each phase
- 50 turn limit — highest HP wins on time
- Cards in play are visible to both sides; hand cards are hidden

---

## RL Training

### Requirements
```bash
pip install -r requirements.txt
```

### Train from scratch (full curriculum)
```bash
python train.py
```

### Train directly vs hard bot
```bash
python train.py --difficulty hard --timesteps 50000 --chunks 5
```

### Fine-tune from best checkpoint
```bash
python train.py --difficulty hard --timesteps 50000 --chunks 5 --checkpoint checkpoints/hard/best_model.zip
```

### Evaluate a saved model
```bash
python train.py --eval-only checkpoints/hard/best_model.zip
```

### Export model to ONNX
```bash
python train.py --export-onnx checkpoints/hard/best_model.zip
```

---

## Architecture

### Environment (`blast_battles_env.py`)
- **Observation:** 101 features (character stats, hand cards, in-play cards, positions, arena effects, phase, helper features)
- **Actions:** 14 (skip, 4 hand slots, 9 arena nodes)
- **Reward:** Per-turn HP delta `(dmg_dealt/bot_maxHP)*5 - (dmg_received/player_maxHP)*5` + terminal HP differential `(player_hp_ratio - bot_hp_ratio)*20`
- **Initiative:** Higher character speed acts first each phase; 50/50 on ties

### Training (`train.py`)
- **Algorithm:** MaskablePPO (Stable Baselines 3)
- **Network:** MLP 101 → 256 → 256 → 14
- **Curriculum:** Training (5% fire) → Easy (75% skip) → Medium (50% skip) → Semi-hard (20% skip) → Hard (0% skip) → Impossible (self-play)
- **Fine-tuning:** LR=5e-5, entropy=0.05 from best hard checkpoint

### Results
| Stage | Gate | Achieved |
|---|---|---|
| Training | 90% | 95%+ |
| Easy | 75% | 76%+ |
| Medium | 60% | 60%+ |
| Hard (fine-tuned) | 50% | **50.6%** |

---

## Files

| File | Description |
|---|---|
| `blast-battles.html` | Game UI (must be served over HTTP for Impossible AI) |
| `blast_battles_env.py` | Gymnasium environment |
| `train.py` | MaskablePPO training pipeline |
| `visualize_training.py` | TensorBoard log to matplotlib charts |
| `requirements.txt` | Python dependencies |
| `blast_battles_policy.onnx` | Exported best model |
| `blast_battles_selfplay.onnx` | Selfplay "Legendary" model |
