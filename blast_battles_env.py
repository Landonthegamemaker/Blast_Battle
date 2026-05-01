"""
blast_battles_env.py
====================
Gymnasium environment for Blast Battles — mirrors the browser game logic exactly.

Observation space  : flat Box (float32), see _obs() for full breakdown
Action space       : Discrete — see ACTION_* constants below
Reward             : shaped dense reward + terminal win/loss bonus
Episode ends       : when either character reaches 0 HP or turn > MAX_TURNS

Usage:
    env = BlastBattlesEnv()
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

Action encoding (Discrete(N)):
    0            : SKIP current phase / STAY PUT in movement
    1..len(hand) : play / fire card at hand_index - 1   (up to 4 slots)
    5..13        : movement — move to arena node 0..8    (9 nodes)

Total discrete actions: 1 + 4 + 9 = 14
"""

import copy, random, math
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ─────────────────────────────────────────────────────────────
#  CONSTANTS  (must match blast-battles.html exactly)
# ─────────────────────────────────────────────────────────────
PHASES = ["movement", "fast", "medium", "slow", "charged"]

ADJACENCY = {
    0: [1],      1: [0, 2],    2: [1, 3, 5, 7],
    3: [2, 4],   4: [3],       5: [2, 6],
    6: [5],      7: [2, 8],    8: [7],
}

# action indices
ACT_SKIP    = 0
ACT_CARD    = slice(1, 5)     # play hand slots 0-3
ACT_MOVE    = slice(5, 14)    # move to arena nodes 0-8

N_ACTIONS   = 14
MAX_TURNS   = 50              # truncation limit

# ─────────────────────────────────────────────────────────────
#  CARD / CHARACTER DATA  (derived from HTML source)
# ─────────────────────────────────────────────────────────────
WEAPON_POOL = [
    # pistols  (range 1)
    {"id":"w1",  "name":"M9 Pistol",     "type":"weapon","subtype":"pistol",        "damage":35,"ammo":6, "speed":"fast",    "range":1},
    {"id":"w2",  "name":"Desert Eagle",  "type":"weapon","subtype":"pistol",        "damage":50,"ammo":4, "speed":"medium",  "range":1},
    {"id":"w3",  "name":"Five-Seven",    "type":"weapon","subtype":"pistol",        "damage":30,"ammo":8, "speed":"fast",    "range":1},
    {"id":"w4",  "name":"Revolver",      "type":"weapon","subtype":"pistol",        "damage":55,"ammo":3, "speed":"medium",  "range":1},
    {"id":"w5",  "name":"Tec-9",         "type":"weapon","subtype":"pistol",        "damage":25,"ammo":12,"speed":"fast",   "range":1},
    # assault rifles  (range 2)
    {"id":"w6",  "name":"AK-47",         "type":"weapon","subtype":"assault_rifle", "damage":65,"ammo":3, "speed":"medium",  "range":2},
    {"id":"w7",  "name":"M4A1",          "type":"weapon","subtype":"assault_rifle", "damage":55,"ammo":4, "speed":"medium",  "range":2},
    {"id":"w8",  "name":"FAMAS",         "type":"weapon","subtype":"assault_rifle", "damage":45,"ammo":5, "speed":"fast",    "range":2},
    {"id":"w9",  "name":"G36C",          "type":"weapon","subtype":"assault_rifle", "damage":50,"ammo":4, "speed":"medium",  "range":2},
    {"id":"w10", "name":"Galil",         "type":"weapon","subtype":"assault_rifle", "damage":60,"ammo":3, "speed":"slow",    "range":2},
    # shotguns  (range 1)
    {"id":"w11", "name":"Mossberg 500",  "type":"weapon","subtype":"shotgun",       "damage":70,"ammo":3, "speed":"slow",    "range":1},
    {"id":"w12", "name":"SPAS-12",       "type":"weapon","subtype":"shotgun",       "damage":80,"ammo":2, "speed":"slow",    "range":1},
    {"id":"w13", "name":"AA-12",         "type":"weapon","subtype":"shotgun",       "damage":65,"ammo":4, "speed":"medium",  "range":1},
    # snipers  (range 3)
    {"id":"w14", "name":"Barrett M82",   "type":"weapon","subtype":"sniper",        "damage":90,"ammo":2, "speed":"charged", "range":3},
    {"id":"w15", "name":"SVD Dragunov",  "type":"weapon","subtype":"sniper",        "damage":75,"ammo":3, "speed":"slow",    "range":3},
    {"id":"w16", "name":"Intervention",  "type":"weapon","subtype":"sniper",        "damage":72,"ammo":4, "speed":"charged", "range":3},
    # grenades/explosives  (range 1)
    {"id":"w17", "name":"Frag Grenade",  "type":"weapon","subtype":"explosive",     "damage":60,"ammo":2, "speed":"slow",    "range":1},
    {"id":"w18", "name":"Flashbang",     "type":"weapon","subtype":"explosive",     "damage":20,"ammo":3, "speed":"fast",    "range":1},
    {"id":"w19", "name":"Smoke Bomb",    "type":"weapon","subtype":"explosive",     "damage":10,"ammo":4, "speed":"fast",    "range":1},
    {"id":"w20", "name":"Sticky Bomb",   "type":"weapon","subtype":"explosive",     "damage":75,"ammo":2, "speed":"charged", "range":1},
    # missiles  (range 2)
    {"id":"w21", "name":"RPG-7",         "type":"weapon","subtype":"missile",       "damage":100,"ammo":2,"speed":"charged", "range":2},
    {"id":"w22", "name":"Stinger SAM",   "type":"weapon","subtype":"missile",       "damage":85, "ammo":2,"speed":"charged", "range":2},
    {"id":"w23", "name":"Javelin",       "type":"weapon","subtype":"missile",       "damage":90, "ammo":2,"speed":"charged", "range":2},
    # melee  (range 0)
    {"id":"w24", "name":"Combat Knife",  "type":"weapon","subtype":"melee",         "damage":30,"ammo":8, "speed":"fast",    "range":0},
    {"id":"w25", "name":"War Hammer",    "type":"weapon","subtype":"melee",         "damage":55,"ammo":5, "speed":"slow",    "range":0},
    {"id":"w26", "name":"Katana",        "type":"weapon","subtype":"melee",         "damage":42,"ammo":6, "speed":"medium",  "range":0},
    {"id":"w27", "name":"Chainsaw",      "type":"weapon","subtype":"melee",         "damage":65,"ammo":4, "speed":"slow",    "range":0},
    {"id":"w28", "name":"Shock Baton",   "type":"weapon","subtype":"melee",         "damage":25,"ammo":10,"speed":"fast",   "range":0},
    {"id":"w29", "name":"Plasma Blade",  "type":"weapon","subtype":"melee",         "damage":50,"ammo":5, "speed":"medium",  "range":0},
    {"id":"w30", "name":"Mini SMG",      "type":"weapon","subtype":"pistol",        "damage":15,"ammo":20,"speed":"fast",   "range":1},
]

# Scale all weapon damage up to match increased character HP pool (300-450 vs old 100-250)
for _w in WEAPON_POOL:
    _w["damage"] = int(_w["damage"] * 1.3)

DEFENSE_POOL = [
    {"id":"d1", "name":"Kevlar Vest",    "type":"defense","subtype":"vest",         "defense":40,"durability":3,"maxDurability":3,"effectiveVs":["pistol","assault_rifle","shotgun"],"healAmount":0},
    {"id":"d2", "name":"Tactical Vest",  "type":"defense","subtype":"vest",         "defense":50,"durability":3,"maxDurability":3,"effectiveVs":["pistol","assault_rifle","shotgun"],"healAmount":0},
    {"id":"d3", "name":"Riot Vest",      "type":"defense","subtype":"vest",         "defense":35,"durability":4,"maxDurability":4,"effectiveVs":["pistol","assault_rifle","shotgun"],"healAmount":0},
    {"id":"d4", "name":"Nano Vest",      "type":"defense","subtype":"vest",         "defense":60,"durability":2,"maxDurability":2,"effectiveVs":["pistol","assault_rifle","shotgun"],"healAmount":0},
    {"id":"d5", "name":"Combat Helmet",  "type":"defense","subtype":"helmet",       "defense":45,"durability":3,"maxDurability":3,"effectiveVs":["sniper"],"healAmount":0},
    {"id":"d6", "name":"Ballistic Helm", "type":"defense","subtype":"helmet",       "defense":55,"durability":3,"maxDurability":3,"effectiveVs":["sniper"],"healAmount":0},
    {"id":"d7", "name":"Full Face Guard","type":"defense","subtype":"helmet",       "defense":35,"durability":4,"maxDurability":4,"effectiveVs":["sniper"],"healAmount":0},
    {"id":"d8", "name":"Exo Helm",       "type":"defense","subtype":"helmet",       "defense":65,"durability":2,"maxDurability":2,"effectiveVs":["sniper"],"healAmount":0},
    {"id":"d9", "name":"Blast Suit",     "type":"defense","subtype":"blast_armor",  "defense":60,"durability":3,"maxDurability":3,"effectiveVs":["explosive","missile"],"healAmount":0},
    {"id":"d10","name":"EOD Gear",       "type":"defense","subtype":"blast_armor",  "defense":70,"durability":2,"maxDurability":2,"effectiveVs":["explosive","missile"],"healAmount":0},
    {"id":"d11","name":"Blast Plate",    "type":"defense","subtype":"blast_armor",  "defense":50,"durability":4,"maxDurability":4,"effectiveVs":["explosive","missile"],"healAmount":0},
    {"id":"d12","name":"Demo Shield",    "type":"defense","subtype":"blast_armor",  "defense":45,"durability":3,"maxDurability":3,"effectiveVs":["explosive","missile"],"healAmount":0},
    {"id":"d13","name":"Steel Plate",    "type":"defense","subtype":"plate_armor",  "defense":30,"durability":5,"maxDurability":5,"effectiveVs":["pistol","assault_rifle","shotgun","sniper","explosive","missile","melee"],"healAmount":0},
    {"id":"d14","name":"Titanium Plate", "type":"defense","subtype":"plate_armor",  "defense":40,"durability":4,"maxDurability":4,"effectiveVs":["pistol","assault_rifle","shotgun","sniper","explosive","missile","melee"],"healAmount":0},
    {"id":"d15","name":"Dragon Scale",   "type":"defense","subtype":"plate_armor",  "defense":50,"durability":3,"maxDurability":3,"effectiveVs":["pistol","assault_rifle","shotgun","sniper","explosive","missile","melee"],"healAmount":0},
    {"id":"d16","name":"Riot Shield",    "type":"defense","subtype":"plate_armor",  "defense":35,"durability":6,"maxDurability":6,"effectiveVs":["pistol","assault_rifle","shotgun","sniper","explosive","missile","melee"],"healAmount":0},
    {"id":"d17","name":"Chain Mail",     "type":"defense","subtype":"plate_armor",  "defense":55,"durability":4,"maxDurability":4,"effectiveVs":["melee"],"healAmount":0},
    {"id":"d18","name":"Spike Guard",    "type":"defense","subtype":"plate_armor",  "defense":45,"durability":3,"maxDurability":3,"effectiveVs":["melee"],"healAmount":0},
    {"id":"d19","name":"Med Kit",        "type":"defense","subtype":"medkit",       "defense":0, "durability":1,"maxDurability":1,"effectiveVs":[],"healAmount":50},
    {"id":"d20","name":"Med Kit II",     "type":"defense","subtype":"medkit",       "defense":0, "durability":1,"maxDurability":1,"effectiveVs":[],"healAmount":40},
    {"id":"d21","name":"Syringe",        "type":"defense","subtype":"syringe",      "defense":0, "durability":1,"maxDurability":1,"effectiveVs":[],"healAmount":30},
    {"id":"d22","name":"Adrenaline Shot","type":"defense","subtype":"syringe",      "defense":0, "durability":1,"maxDurability":1,"effectiveVs":[],"healAmount":25},
    {"id":"d23","name":"Bandages",       "type":"defense","subtype":"bandage",      "defense":0, "durability":1,"maxDurability":1,"effectiveVs":[],"healAmount":20},
    {"id":"d24","name":"Field Dressing", "type":"defense","subtype":"bandage",      "defense":0, "durability":1,"maxDurability":1,"effectiveVs":[],"healAmount":15},
    {"id":"d25","name":"Ointment",       "type":"defense","subtype":"ointment",     "defense":0, "durability":1,"maxDurability":1,"effectiveVs":[],"healAmount":12},
    {"id":"d26","name":"Combat Stim",    "type":"defense","subtype":"syringe",      "defense":0, "durability":1,"maxDurability":1,"effectiveVs":[],"healAmount":35},
    {"id":"d27","name":"Exo Suit",       "type":"defense","subtype":"plate_armor",  "defense":55,"durability":3,"maxDurability":3,"effectiveVs":["pistol","assault_rifle","shotgun","sniper","explosive","missile","melee"],"healAmount":0},
    {"id":"d28","name":"Energy Shield",  "type":"defense","subtype":"plate_armor",  "defense":45,"durability":3,"maxDurability":3,"effectiveVs":["pistol","assault_rifle","shotgun","sniper","explosive","missile","melee"],"healAmount":0},
    {"id":"d29","name":"Ceramic Plate",  "type":"defense","subtype":"vest",         "defense":38,"durability":4,"maxDurability":4,"effectiveVs":["pistol","assault_rifle","shotgun"],"healAmount":0},
    {"id":"d30","name":"Carbon Weave",   "type":"defense","subtype":"blast_armor",  "defense":42,"durability":3,"maxDurability":3,"effectiveVs":["explosive","missile"],"healAmount":0},
]

CHARACTER_POOL = [
    {"id":"c1", "name":"Ghost Ops",   "faction":"hero",   "hp":330,"maxHp":330,"speed":9, "attribute":"pistol_specialist"},
    {"id":"c2", "name":"Iron Rex",    "faction":"villain","hp":410,"maxHp":410,"speed":4, "attribute":"heavy_armor"},
    {"id":"c3", "name":"Viper",       "faction":"villain","hp":315,"maxHp":315,"speed":10,"attribute":"swift"},
    {"id":"c4", "name":"The Medic",   "faction":"hero",   "hp":360,"maxHp":360,"speed":6, "attribute":"healing"},
    {"id":"c5", "name":"Tank",        "faction":"villain","hp":450,"maxHp":450,"speed":2, "attribute":"sniper_resist"},
    {"id":"c6", "name":"Shadow",      "faction":"hero",   "hp":300,"maxHp":300,"speed":10,"attribute":"shotgun_specialist"},
    {"id":"c7", "name":"Commando",    "faction":"hero",   "hp":370,"maxHp":370,"speed":7, "attribute":"rifle_specialist"},
    {"id":"c8", "name":"Rogue",       "faction":"villain","hp":325,"maxHp":325,"speed":9, "attribute":"swift"},
    {"id":"c9", "name":"Berserker",   "faction":"villain","hp":380,"maxHp":380,"speed":5, "attribute":"melee_specialist"},
    {"id":"c10","name":"Sentinel",    "faction":"hero",   "hp":400,"maxHp":400,"speed":4, "attribute":"explosive_resist"},
    {"id":"c11","name":"Wraith",      "faction":"hero",   "hp":310,"maxHp":310,"speed":10,"attribute":"extra_carry"},
    {"id":"c12","name":"Warlord",     "faction":"villain","hp":350,"maxHp":350,"speed":6, "attribute":"sniper_specialist"},
    {"id":"c13","name":"Titan",       "faction":"villain","hp":430,"maxHp":430,"speed":3, "attribute":"heavy_armor"},
    {"id":"c14","name":"Doc Havoc",   "faction":"hero",   "hp":355,"maxHp":355,"speed":7, "attribute":"healing"},
    {"id":"c15","name":"Demolisher",  "faction":"villain","hp":340,"maxHp":340,"speed":6, "attribute":"explosive_specialist"},
]

LOCATION_POOL = [
    {"id":"l1","effect":"neutral"},
    {"id":"l2","effect":"hero_zone"},
    {"id":"l3","effect":"villain_zone"},
    {"id":"l4","effect":"draw_weapon"},
    {"id":"l5","effect":"draw_defense"},
    {"id":"l6","effect":"radiation"},
    {"id":"l7","effect":"poison"},
    {"id":"l8","effect":"discard"},
    {"id":"l9","effect":"neutral"},
]

# encode subtypes / attributes / effects as small integers for obs vector
SUBTYPES = ["pistol","assault_rifle","shotgun","sniper","explosive","missile","melee",
            "vest","helmet","blast_armor","plate_armor","medkit","syringe","bandage","ointment"]
SUBTYPE_IDX = {s: i for i, s in enumerate(SUBTYPES)}

ATTRIBUTES = ["pistol_specialist","shotgun_specialist","rifle_specialist","sniper_specialist",
              "explosive_specialist","melee_specialist","swift","healing","heavy_armor",
              "sniper_resist","explosive_resist","extra_carry"]
ATTR_IDX = {a: i for i, a in enumerate(ATTRIBUTES)}

LOC_EFFECTS = ["neutral","hero_zone","villain_zone","draw_weapon","draw_defense",
               "radiation","poison","discard"]
LOC_IDX = {e: i for i, e in enumerate(LOC_EFFECTS)}

SPEED_IDX = {"fast":0,"medium":1,"slow":2,"charged":3}

# observation dimension:
#   self  char  : hp_norm, maxhp_norm, faction(1), attr_idx_norm  = 4
#   opp   char  : same                                             = 4
#   phase       : one-hot 5                                        = 5
#   positions   : self_pos/8, opp_pos/8, distance/4               = 3
#   locations   : 9 × effect_idx/7                                 = 9
#   hand slots  : 4 × 6 (type, subtype, speed, range, dmg/def, ammo/dur) = 24
#   in-play     : 4 × 6                                            = 24
#   opp in-play : 4 × 6 (armor visible)                           = 24
#   turn_norm   : 1                                                = 1
#   TOTAL                                                          = 98
OBS_DIM = 101  # 98 base + 3 explicit helper features

# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────
def _shuffle(lst):
    lst = copy.deepcopy(lst)
    random.shuffle(lst)
    return lst

def _bfs_dist(a, b):
    if a == b:
        return 0
    visited = {a}
    frontier = [a]
    d = 0
    while frontier:
        d += 1
        nxt = []
        for p in frontier:
            for n in ADJACENCY.get(p, []):
                if n == b:
                    return d
                if n not in visited:
                    visited.add(n)
                    nxt.append(n)
        frontier = nxt
    return 99

def _reachable(pos, steps):
    visited = {pos}
    frontier = [pos]
    for _ in range(steps):
        nxt = []
        for p in frontier:
            for n in ADJACENCY.get(p, []):
                if n not in visited:
                    visited.add(n)
                    nxt.append(n)
        frontier = nxt
    visited.discard(pos)
    return visited

def _encode_card(card):
    """Return 6-float encoding of one card slot (zeros if None)."""
    if card is None:
        return [0.0] * 6
    is_weapon = float(card["type"] == "weapon")
    sub = SUBTYPE_IDX.get(card["subtype"], 0) / max(len(SUBTYPES) - 1, 1)
    if card["type"] == "weapon":
        spd = SPEED_IDX.get(card["speed"], 0) / 3.0
        rng = card["range"] / 3.0
        val = card["damage"] / 100.0
        ammo = card["ammo"] / 20.0
    else:  # defense
        spd = 0.0
        rng = 0.0
        val = card.get("defense", 0) / 70.0 if card.get("healAmount", 0) == 0 else card["healAmount"] / 50.0
        ammo = card.get("durability", 1) / 6.0
    return [is_weapon, sub, spd, rng, val, ammo]

def _pad_cards(cards, n=4):
    """Pad / truncate card list to exactly n slots."""
    out = list(cards)[:n]
    while len(out) < n:
        out.append(None)
    return out

# ─────────────────────────────────────────────────────────────
#  ENVIRONMENT
# ─────────────────────────────────────────────────────────────
class BlastBattlesEnv(gym.Env):
    """
    Headless Blast Battles — 1 agent (player) vs heuristic bot.
    The agent is always 'player'; the bot mirrors the simple
    heuristic from the browser game.
    For 'impossible' difficulty, pass bot_model= a trained MaskablePPO
    instance to enable self-play.
    """
    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(self, render_mode: Optional[str] = None, difficulty: str = "medium",
                 bot_model=None):
        super().__init__()
        self.render_mode = render_mode
        self.difficulty  = difficulty
        self.bot_model   = bot_model  # trained model for impossible self-play
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.G = {}

    # ── RESET ─────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        w_deck = _shuffle(WEAPON_POOL)
        d_deck = _shuffle(DEFENSE_POOL)
        chars   = _shuffle(CHARACTER_POOL)

        player_char = copy.deepcopy(chars[0])
        bot_char    = copy.deepcopy(chars[1])

        def draw_hand(wd, dd):
            return [copy.deepcopy(wd.pop(0)),
                    copy.deepcopy(wd.pop(0)),
                    copy.deepcopy(dd.pop(0)),
                    copy.deepcopy(dd.pop(0))]

        player_hand = draw_hand(w_deck, d_deck)
        bot_hand    = draw_hand(w_deck, d_deck)

        locs = _shuffle(LOCATION_POOL)[:9]
        locs[2] = copy.deepcopy({"id":"lCenter","effect":"neutral"})

        self.G = {
            "turn": 1,
            "phase": 0,
            "difficulty": self.difficulty,
            "player": player_char,
            "bot": bot_char,
            "player_hand": player_hand,
            "bot_hand": bot_hand,
            "player_in_play": [],
            "bot_in_play": [],
            "player_pos": 0,
            "bot_pos": 4,
            "locations": locs,
            "weapon_deck": w_deck,
            "defense_deck": d_deck,
            "game_over": False,
            "winner": None,
            "dmg_dealt_this_turn": 0,
            "total_hp_progress": 0.0,
            "total_move_reward": 0.0,
            "entered_best_range": False,
            "turn_dmg_dealt": 0.0,
            "turn_dmg_received": 0.0,
            "game_dmg_dealt": 0.0,
            "game_dmg_received": 0.0,
        }

        # Apply location effects at turn start before first action
        self._apply_location_effects()
        return self._obs(), self._info()

    # ── STEP ──────────────────────────────────────────────────
    def step(self, action: int):
        G = self.G
        assert not G["game_over"], "Episode already over — call reset()"

        phase = PHASES[G["phase"]]
        reward = 0.0

        # ── Determine initiative — faster character acts first ─
        # On a speed tie, randomise 50/50 each phase
        player_speed = G["player"]["speed"]
        bot_speed    = G["bot"]["speed"]
        if player_speed > bot_speed:
            player_first = True
        elif bot_speed > player_speed:
            player_first = False
        else:
            player_first = random.random() < 0.5

        def player_act():
            nonlocal reward
            if phase == "movement":
                reward += self._player_move(action)
            else:
                reward += self._player_combat(action, phase)

        def bot_act():
            if phase == "movement":
                if self.bot_model is not None and self.difficulty == "impossible":
                    self._bot_move_model()
                else:
                    self._bot_move()
            else:
                if self.bot_model is not None and self.difficulty == "impossible":
                    self._bot_combat_model(phase)
                else:
                    self._bot_combat(phase)

        if player_first:
            G["_hp_bot_before"]    = G["bot"]["hp"]
            G["_hp_player_before"] = G["player"]["hp"]
            player_act()
            self._check_win()
            if G["game_over"]:
                return self._obs(), reward + self._terminal_reward(), True, False, self._info()
            bot_act()
        else:
            G["_hp_bot_before"]    = G["bot"]["hp"]
            G["_hp_player_before"] = G["player"]["hp"]
            bot_act()
            self._check_win()
            if G["game_over"]:
                return self._obs(), reward + self._terminal_reward(), True, False, self._info()
            player_act()

        self._check_win()
        if G["game_over"]:
            return self._obs(), reward + self._terminal_reward(), True, False, self._info()

        # ── Accumulate HP deltas into turn bucket ─────────────
        bot_hp_after    = G["bot"]["hp"]
        player_hp_after = G["player"]["hp"]
        G["turn_dmg_dealt"]    = G.get("turn_dmg_dealt",    0.0) + max(0, G["_hp_bot_before"]    - bot_hp_after)
        G["turn_dmg_received"] = G.get("turn_dmg_received", 0.0) + max(0, G["_hp_player_before"] - player_hp_after)

        # ── Advance phase ─────────────────────────────────────
        G["phase"] += 1
        if G["phase"] >= len(PHASES):
            G["phase"] = 0
            G["turn"] += 1
            # Emit full turn's HP exchange as a single reward signal
            dealt    = G.pop("turn_dmg_dealt",    0.0)
            received = G.pop("turn_dmg_received", 0.0)
            reward += (dealt    / G["bot"]["maxHp"])    * 5.0
            reward -= (received / G["player"]["maxHp"]) * 5.0
            reward += self._apply_location_effects()
            G["dmg_dealt_this_turn"] = 0

        # Truncation — when turn limit reached, highest HP wins (no draws)
        max_turns = 50 if G.get("difficulty") == "training" else MAX_TURNS
        truncated = G["turn"] > max_turns
        if truncated:
            G["game_over"] = True
            # Decide winner by remaining HP — eliminates draws entirely
            player_hp = G["player"]["hp"]
            bot_hp    = G["bot"]["hp"]
            if player_hp > bot_hp:
                G["winner"] = "player"
            elif bot_hp > player_hp:
                G["winner"] = "bot"
            else:
                G["winner"] = "player"  # tie goes to player (agent advantage)
            reward += self._terminal_reward()

        return self._obs(), reward, G["game_over"] and not truncated, truncated, self._info()

    # ── OBSERVATION ───────────────────────────────────────────
    def _obs(self):
        G = self.G
        v = []

        def char_vec(c):
            return [
                c["hp"] / c["maxHp"],
                c["maxHp"] / 450.0,
                1.0 if c["faction"] == "hero" else 0.0,
                ATTR_IDX.get(c["attribute"], 0) / max(len(ATTRIBUTES) - 1, 1),
            ]

        v += char_vec(G["player"])
        v += char_vec(G["bot"])

        # Phase one-hot (5)
        ph = [0.0] * 5
        ph[G["phase"]] = 1.0
        v += ph

        # Positions
        v.append(G["player_pos"] / 8.0)
        v.append(G["bot_pos"] / 8.0)
        v.append(_bfs_dist(G["player_pos"], G["bot_pos"]) / 4.0)

        # Location effects (9 nodes)
        for loc in G["locations"]:
            v.append(LOC_IDX.get(loc["effect"], 0) / max(len(LOC_EFFECTS) - 1, 1))

        # Player hand (4 slots × 6)
        for card in _pad_cards(G["player_hand"]):
            v += _encode_card(card)

        # Player in-play (4 slots × 6)
        for card in _pad_cards(G["player_in_play"]):
            v += _encode_card(card)

        # Bot in-play (armor/weapons visible — 4 slots × 6)
        for card in _pad_cards(G["bot_in_play"]):
            v += _encode_card(card)

        # Turn norm
        v.append(min(G["turn"], MAX_TURNS) / MAX_TURNS)

        # ── Explicit helper features — reduce network computation burden ──
        phase = PHASES[G["phase"]]
        dist  = _bfs_dist(G["player_pos"], G["bot_pos"])
        usable = [c for c in G["player_hand"] + G["player_in_play"]
                  if c["type"] == "weapon" and c.get("ammo", 0) > 0]

        # Feature 99: has valid shot RIGHT NOW (network was missing 49% of shots without this)
        has_valid_shot = 1.0 if any(
            c["speed"] == phase and c["range"] == dist for c in usable
        ) else 0.0
        v.append(has_valid_shot)

        # Feature 100: signed distance from best weapon range (0 = at best range)
        if usable:
            best_range = max(usable, key=lambda c: c["damage"])["range"]
            v.append((dist - best_range) / 4.0)  # negative = too close, positive = too far
        else:
            v.append(0.0)

        # Feature 101: bot can fire back right now (danger awareness)
        bot_can_fire_now = 1.0 if any(
            c["type"] == "weapon" and c.get("ammo", 0) > 0
            and c["speed"] == phase and c["range"] == dist
            for c in G["bot_in_play"]
        ) else 0.0
        v.append(bot_can_fire_now)

        assert len(v) == OBS_DIM, f"Obs dim mismatch: {len(v)} ≠ {OBS_DIM}"
        return np.array(v, dtype=np.float32)

    def _bot_obs(self):
        """Construct observation from the bot's perspective for self-play.
        Mirrors _obs() by swapping player/bot roles so the model sees
        itself as the 'player' and the opponent as the 'bot'."""
        G = self.G
        v = []

        def char_vec(c):
            return [
                c["hp"] / c["maxHp"],
                c["maxHp"] / 450.0,
                1.0 if c["faction"] == "hero" else 0.0,
                ATTR_IDX.get(c["attribute"], 0) / max(len(ATTRIBUTES) - 1, 1),
            ]

        # Swap: bot is now "player", player is now "bot"
        v += char_vec(G["bot"])
        v += char_vec(G["player"])

        # Phase one-hot (same)
        ph = [0.0] * 5
        ph[G["phase"]] = 1.0
        v += ph

        # Positions swapped
        v.append(G["bot_pos"] / 8.0)
        v.append(G["player_pos"] / 8.0)
        v.append(_bfs_dist(G["bot_pos"], G["player_pos"]) / 4.0)

        # Location effects (same)
        for loc in G["locations"]:
            v.append(LOC_IDX.get(loc["effect"], 0) / max(len(LOC_EFFECTS) - 1, 1))

        # Bot hand as "player hand" (bot's cards are hidden from player, but bot knows them)
        for card in _pad_cards(G["bot_hand"]):
            v += _encode_card(card)

        # Bot in-play as "player in-play"
        for card in _pad_cards(G["bot_in_play"]):
            v += _encode_card(card)

        # Player in-play as "bot in-play" (visible to both sides)
        for card in _pad_cards(G["player_in_play"]):
            v += _encode_card(card)

        # Turn norm
        v.append(min(G["turn"], MAX_TURNS) / MAX_TURNS)

        # Explicit helper features from bot's perspective
        phase = PHASES[G["phase"]]
        dist  = _bfs_dist(G["bot_pos"], G["player_pos"])
        bot_usable = [c for c in G["bot_hand"] + G["bot_in_play"]
                      if c["type"] == "weapon" and c.get("ammo", 0) > 0]

        has_valid_shot = 1.0 if any(
            c["speed"] == phase and c["range"] == dist for c in bot_usable
        ) else 0.0
        v.append(has_valid_shot)

        if bot_usable:
            best_range = max(bot_usable, key=lambda c: c["damage"])["range"]
            v.append((dist - best_range) / 4.0)
        else:
            v.append(0.0)

        player_can_fire_now = 1.0 if any(
            c["type"] == "weapon" and c.get("ammo", 0) > 0
            and c["speed"] == phase and c["range"] == dist
            for c in G["player_in_play"]
        ) else 0.0
        v.append(player_can_fire_now)

        return np.array(v, dtype=np.float32)

    def _bot_obs_action_masks(self):
        """Action masks from the bot's perspective for self-play."""
        G = self.G
        mask = np.zeros(N_ACTIONS, dtype=bool)
        phase = PHASES[G["phase"]]
        mask[0] = True  # SKIP always valid

        if phase == "movement":
            is_swift = G["bot"]["attribute"] == "swift"
            steps = 2 if is_swift else 1
            reachable = _reachable(G["bot_pos"], steps)
            for node in range(9):
                if node in reachable and node != G["bot_pos"]:
                    mask[5 + node] = True
        else:
            dist = _bfs_dist(G["bot_pos"], G["player_pos"])
            hand = G["bot_hand"]
            for slot in range(4):
                if slot >= len(hand):
                    continue
                card = hand[slot]
                if card["type"] == "weapon":
                    if card["speed"] == phase and card["range"] == dist:
                        mask[1 + slot] = True
                elif card["type"] == "defense":
                    if card["healAmount"] > 0:
                        mask[1 + slot] = True
                    else:
                        already = any(c["id"] == card["id"] for c in G["bot_in_play"])
                        if not already:
                            mask[1 + slot] = True
            # In-play bot weapons
            for card in G["bot_in_play"]:
                if card["type"] == "weapon" and card["ammo"] > 0:
                    if card["speed"] == phase and card["range"] == dist:
                        mask[1] = True
                        break
        return mask

    def _info(self):
        G = self.G
        return {
            "turn":       G["turn"],
            "phase":      PHASES[G["phase"]],
            "player_hp":  G["player"]["hp"],
            "bot_hp":     G["bot"]["hp"],
            "player_pos": G["player_pos"],
            "bot_pos":    G["bot_pos"],
            "winner":     G.get("winner"),
        }

    # ── ACTION MASK ───────────────────────────────────────────
    def action_masks(self) -> np.ndarray:
        """
        Returns a boolean array of length N_ACTIONS indicating which
        actions are valid in the current state.

        MaskablePPO calls this automatically each step — invalid actions
        get -inf logit so the agent never wastes steps on illegal moves.

        Action layout:
            0        : SKIP — always valid
            1-4      : play hand slot 0-3
            5-13     : move to arena node 0-8 (only valid in movement phase)
        """
        G = self.G
        mask = np.zeros(N_ACTIONS, dtype=bool)
        phase = PHASES[G["phase"]]

        # Action 0 — SKIP always valid
        mask[0] = True

        if phase == "movement":
            # Movement phase — card actions invalid, movement actions valid
            # if the target node is reachable
            is_swift = G["player"]["attribute"] == "swift"
            steps = 2 if is_swift else 1
            reachable = _reachable(G["player_pos"], steps)
            for node in range(9):
                if node in reachable and node != G["player_pos"]:
                    mask[5 + node] = True

        else:
            # Combat phase — movement actions invalid
            dist = _bfs_dist(G["player_pos"], G["bot_pos"])
            hand = G["player_hand"]

            for slot in range(4):
                if slot >= len(hand):
                    continue   # empty slot
                card = hand[slot]
                if card["type"] == "weapon":
                    # Valid only if phase matches speed AND range matches dist
                    if card["speed"] == phase and card["range"] == dist:
                        mask[1 + slot] = True
                elif card["type"] == "defense":
                    if card["healAmount"] > 0:
                        # Heal cards — valid any combat phase
                        mask[1 + slot] = True
                    else:
                        # Armor — valid if not already equipped
                        already = any(c["id"] == card["id"] for c in G["player_in_play"])
                        if not already:
                            mask[1 + slot] = True

            # In-play weapons — valid if phase matches speed AND range matches dist
            # Mark slot 0 as valid if any in-play weapon can fire
            # (agent picks the in-play weapon via action 1 which maps to slot 0)
            for card in G["player_in_play"]:
                if card["type"] == "weapon" and card["ammo"] > 0:
                    if card["speed"] == phase and card["range"] == dist:
                        mask[1] = True  # always use slot 0 for in-play weapon firing
                        break

        # Ensure at least SKIP is always available (safety net)
        if not mask.any():
            mask[0] = True

        return mask

    # ── PLAYER COMBAT ACTION ──────────────────────────────────
    def _player_combat(self, action, phase):
        G = self.G
        dist = _bfs_dist(G["player_pos"], G["bot_pos"])
        hp_ratio = G["player"]["hp"] / G["player"]["maxHp"]

        if action == ACT_SKIP:
            return 0.0  # no signal for skipping — agent learns opportunity cost naturally

        if 1 <= action <= 4:
            idx = action - 1
            hand = G["player_hand"]

            if idx < len(hand):
                card = hand[idx]

                if card["type"] == "weapon":
                    if card["speed"] != phase or dist != card["range"]:
                        return self._fire_inplay_weapon(phase, dist)
                    G["player_hand"].pop(idx)
                    G["player_in_play"].append(card)
                    return self._resolve_weapon_fire(card)

                elif card["type"] == "defense":
                    if card["healAmount"] > 0:
                        heal = min(card["healAmount"], G["player"]["maxHp"] - G["player"]["hp"])
                        G["player"]["hp"] += heal
                        G["player_hand"].pop(idx)
                    else:
                        already = any(c["id"] == card["id"] for c in G["player_in_play"])
                        if already:
                            return 0.0
                        G["player_hand"].pop(idx)
                        G["player_in_play"].append(card)
                    return 0.0  # defense rewards come implicitly via reduced incoming damage

            else:
                return self._fire_inplay_weapon(phase, dist)

        return 0.0

    def _fire_inplay_weapon(self, phase, dist):
        """Fire the best available in-play weapon matching phase and dist."""
        G = self.G
        valid = [c for c in G["player_in_play"]
                 if c["type"] == "weapon" and c["ammo"] > 0
                 and c["speed"] == phase and c["range"] == dist]
        if not valid:
            return 0.0
        # Pick highest damage in-play weapon
        card = max(valid, key=lambda c: c["damage"])
        return self._resolve_weapon_fire(card)

    def _resolve_weapon_fire(self, card):
        """Apply weapon damage. HP delta is measured in step() and converted to reward."""
        G = self.G
        dist = _bfs_dist(G["player_pos"], G["bot_pos"])
        dmg = card["damage"]
        dmg = self._apply_weapon_buff(dmg, card, G["player"])
        dmg = self._apply_location_dmg_buff(dmg, G["player"], G["player_pos"])
        result = self._apply_armor(dmg, card, G["bot_in_play"], G["bot"])
        G["bot"]["hp"] = max(0, G["bot"]["hp"] - result["final_dmg"])
        G["dmg_dealt_this_turn"] = G.get("dmg_dealt_this_turn", 0) + result["final_dmg"]
        card["ammo"] -= 1
        if card["ammo"] <= 0 and card in G["player_in_play"]:
            G["player_in_play"].remove(card)
        return 0.0  # reward issued in step() via HP delta

    # ── PLAYER MOVEMENT ACTION ────────────────────────────────
    def _player_move(self, action):
        G = self.G

        if action == ACT_SKIP:
            return 0.0  # holding position — reward comes from firing next phase

        if 5 <= action <= 13:
            target = action - 5
            is_swift = G["player"]["attribute"] == "swift"
            steps = 2 if is_swift else 1
            reachable = _reachable(G["player_pos"], steps)

            if target == G["player_pos"]:
                return 0.0
            if target not in reachable:
                return -0.05  # tiny penalty for invalid move attempt

            G["player_pos"] = target
            return 0.0  # positioning reward is implicit — better range = more damage next phase

        return 0.0

    # ── ARMOR MITIGATION ESTIMATE ─────────────────────────────
    def _estimate_armor_mitigation(self, weapon_card):
        """
        Rough fraction (0–1) of how much our in-play armor reduces
        damage from weapon_card. Used for movement danger penalty.
        """
        G = self.G
        total_reduction = 0.0
        total_dmg = max(weapon_card["damage"], 1)

        for armor in G["player_in_play"]:
            if armor["type"] != "defense" or armor["healAmount"] > 0:
                continue
            is_effective = weapon_card["subtype"] in armor["effectiveVs"]
            reduction = armor["defense"] if is_effective else int(armor["defense"] * 0.4)
            attr = G["player"]["attribute"]
            if attr == "heavy_armor":
                reduction = int(reduction * 1.4)
            if attr == "sniper_resist" and weapon_card["subtype"] == "sniper":
                reduction = int(total_dmg * 0.5)
            if attr == "explosive_resist" and weapon_card["subtype"] in ("explosive", "missile"):
                reduction = int(total_dmg * 0.4)
            total_reduction += reduction

        return min(total_reduction / total_dmg, 1.0)

    def _bot_combat_model(self, phase):
        """Use the trained model to make bot combat decisions (self-play)."""
        obs = self._bot_obs()
        masks = self._bot_obs_action_masks()
        # Only allow combat actions (0-4), not movement (5-13)
        masks[5:] = False
        if not masks.any():
            return
        action, _ = self.bot_model.predict(obs, deterministic=True, action_masks=masks)
        action = int(action)
        if action == 0:
            return  # skip
        self._execute_bot_card(action, phase)

    def _bot_move_model(self):
        """Use the trained model to make bot movement decisions (self-play)."""
        obs = self._bot_obs()
        masks = self._bot_obs_action_masks()
        # Only allow movement actions (0, 5-13), not combat (1-4)
        masks[1:5] = False
        if not masks.any():
            return
        action, _ = self.bot_model.predict(obs, deterministic=True, action_masks=masks)
        action = int(action)
        if action == 0 or action < 5:
            return  # skip
        target = action - 5
        G = self.G
        is_swift = G["bot"]["attribute"] == "swift"
        steps = 2 if is_swift else 1
        reachable = _reachable(G["bot_pos"], steps)
        if target in reachable:
            G["bot_pos"] = target

    def _execute_bot_card(self, action, phase):
        """Execute a bot card action (used in self-play mode)."""
        G = self.G
        idx = action - 1
        if idx >= len(G["bot_hand"]):
            return
        card = G["bot_hand"][idx]
        dist = _bfs_dist(G["bot_pos"], G["player_pos"])

        if card["type"] == "weapon":
            if card["speed"] != phase or card["range"] != dist:
                return
            G["bot_hand"].pop(idx)
            G["bot_in_play"].append(card)
            dmg = card["damage"]
            result = self._apply_armor(dmg, card, G["player_in_play"], G["player"])
            G["player"]["hp"] = max(0, G["player"]["hp"] - result["final_dmg"])
            card["ammo"] -= 1
            if card["ammo"] <= 0:
                G["bot_in_play"].remove(card)
        elif card["type"] == "defense":
            if card["healAmount"] > 0:
                G["bot"]["hp"] = min(G["bot"]["maxHp"], G["bot"]["hp"] + card["healAmount"])
                G["bot_hand"].pop(idx)
            else:
                already = any(c["id"] == card["id"] for c in G["bot_in_play"])
                if not already:
                    G["bot_hand"].pop(idx)
                    G["bot_in_play"].append(card)

    # ── BOT HEURISTIC ─────────────────────────────────────────
    def _bot_combat(self, phase):
        """
        Difficulty-aware heuristic bot.
        skip_rate: training=0.90 (fire-only 10%), easy=0.75, medium=0.50, hard=0.0
        Training fires only (no heal/armor) so agent learns defense without losing kill signal.
        """
        G = self.G
        skip_rate = {"training": 0.95, "easy": 0.75, "medium": 0.50, "semi_hard": 0.20, "hard": 0.0, "impossible": 0.0}
        if random.random() < skip_rate.get(G.get("difficulty", "medium"), 0.50):
            return 0  # bot passes

        # Training bot: fire only (no heal/armor) — teaches agent that bots fire back
        # while keeping bot easy to kill so offensive reward signal stays strong
        if G.get("difficulty") == "training":
            dist  = _bfs_dist(G["bot_pos"], G["player_pos"])
            all_w = [c for c in G["bot_hand"] + G["bot_in_play"] if c["type"] == "weapon"]
            valid = [c for c in all_w if c["speed"] == phase and c["range"] == dist]
            if not valid:
                return 0
            weapon = random.choice(valid)
            if weapon in G["bot_hand"]:
                G["bot_hand"].remove(weapon)
                G["bot_in_play"].append(weapon)
            dmg    = weapon["damage"]
            dmg    = self._apply_weapon_buff(dmg, weapon, G["bot"])
            result = self._apply_armor(dmg, weapon, G["player_in_play"], G["player"])
            G["player"]["hp"] = max(0, G["player"]["hp"] - result["final_dmg"])
            weapon["ammo"] -= 1
            if weapon["ammo"] <= 0 and weapon in G["bot_in_play"]:
                G["bot_in_play"].remove(weapon)
            return result["final_dmg"]

        all_bot_weapons = [c for c in G["bot_hand"] + G["bot_in_play"] if c["type"] == "weapon"]
        has_weapons     = len(all_bot_weapons) > 0
        has_heal        = any(c["type"] == "defense" and c["healAmount"] > 0 for c in G["bot_hand"])
        has_armor       = any(c["type"] == "defense" and c["healAmount"] == 0 for c in G["bot_in_play"])
        bot_total       = len(G["bot_hand"]) + len(G["bot_in_play"])
        bot_max         = 5 if G["bot"]["attribute"] == "extra_carry" else 4
        hp_ratio        = G["bot"]["hp"] / G["bot"]["maxHp"]
        is_low_hp       = hp_ratio < 0.4
        is_hard         = G.get("difficulty") in ("semi_hard", "hard", "impossible")

        # Priority 1: heal if low HP and heal card available
        if is_low_hp and has_heal:
            heal_card = next((c for c in G["bot_hand"] if c["type"] == "defense" and c["healAmount"] > 0), None)
            if heal_card:
                G["bot"]["hp"] = min(G["bot"]["maxHp"], G["bot"]["hp"] + heal_card["healAmount"])
                G["bot_hand"].remove(heal_card)
                return 0

        # Priority 2: equip armor if none in play
        if not has_armor:
            unequipped = [c for c in G["bot_hand"] if c["type"] == "defense" and c["healAmount"] == 0]
            if unequipped:
                # Hard: pick highest defense; easy/medium: random
                armor = max(unequipped, key=lambda c: c["defense"]) if is_hard else random.choice(unequipped)
                G["bot_hand"].remove(armor)
                G["bot_in_play"].append(armor)
                return 0

        # Priority 3: fire weapon matching phase speed + current range
        dist  = _bfs_dist(G["bot_pos"], G["player_pos"])
        valid = [c for c in all_bot_weapons if c["speed"] == phase and c["range"] == dist]

        if valid:
            # Hard: pick highest damage; easy/medium: random
            weapon = max(valid, key=lambda c: c["damage"]) if is_hard else random.choice(valid)
            if weapon in G["bot_hand"]:
                G["bot_hand"].remove(weapon)
                G["bot_in_play"].append(weapon)

            dmg = weapon["damage"]
            dmg = self._apply_weapon_buff(dmg, weapon, G["bot"])
            dmg = self._apply_location_dmg_buff(dmg, G["bot"], G["bot_pos"])
            result = self._apply_armor(dmg, weapon, G["player_in_play"], G["player"])
            G["player"]["hp"] = max(0, G["player"]["hp"] - result["final_dmg"])

            weapon["ammo"] -= 1
            if weapon["ammo"] <= 0 and weapon in G["bot_in_play"]:
                G["bot_in_play"].remove(weapon)

            return result["final_dmg"]

        return 0

    def _bot_move(self):
        G = self.G

        # Training difficulty — bot approaches agent to teach hold-and-wait strategy
        # Agent must learn to hold optimal firing position as bot closes in
        if G.get("difficulty") == "training":
            if random.random() < 0.50:  # 50% chance bot stays still
                return
            # Move one step toward the agent
            is_swift = G["bot"]["attribute"] == "swift"
            steps = 2 if is_swift else 1
            pos = G["bot_pos"]
            for _ in range(steps):
                neighbors = ADJACENCY.get(pos, [])
                # Pick neighbor that minimizes distance to player
                closer = [n for n in neighbors
                          if _bfs_dist(n, G["player_pos"]) < _bfs_dist(pos, G["player_pos"])]
                if closer:
                    pos = min(closer, key=lambda n: _bfs_dist(n, G["player_pos"]))
                else:
                    break
            G["bot_pos"] = pos
            return
        is_swift = G["bot"]["attribute"] == "swift"
        steps    = 2 if is_swift else 1

        # Easy/medium bots move randomly 75%/50% of the time instead of using heuristic
        # Hard/impossible always use the full positioning heuristic
        move_skip = {"easy": 0.75, "medium": 0.50, "semi_hard": 0.20}
        if G.get("difficulty") in move_skip:
            if random.random() < move_skip[G["difficulty"]]:
                # Random movement — pick any adjacent node or stay
                neighbors = ADJACENCY.get(G["bot_pos"], [])
                if neighbors:
                    G["bot_pos"] = random.choice(neighbors + [G["bot_pos"]])
                return

        all_bot_weapons = [c for c in G["bot_hand"] + G["bot_in_play"] if c["type"] == "weapon"]
        has_weapons     = len(all_bot_weapons) > 0
        has_heal        = any(c["type"] == "defense" and c["healAmount"] > 0 for c in G["bot_hand"])
        has_armor       = any(c["type"] == "defense" and c["healAmount"] == 0 for c in G["bot_in_play"])
        bot_total       = len(G["bot_hand"]) + len(G["bot_in_play"])
        bot_max         = 5 if G["bot"]["attribute"] == "extra_carry" else 4
        has_room        = bot_total < bot_max
        hp_ratio        = G["bot"]["hp"] / G["bot"]["maxHp"]
        is_low_hp       = hp_ratio < 0.4
        needs_supplies  = is_low_hp and not has_heal and not has_armor and has_room

        # Preferred weapon range — most common range among bot's weapons
        if has_weapons:
            range_counts = {}
            for w in all_bot_weapons:
                range_counts[w["range"]] = range_counts.get(w["range"], 0) + 1
            preferred_range = max(range_counts, key=range_counts.get)
        else:
            preferred_range = 1

        def score_node(node):
            loc = G["locations"][node]
            eff = loc["effect"]
            # Seek Forge when desperate for supplies
            if needs_supplies and eff == "draw_defense":
                return 1000
            # Seek Armory when out of weapons
            if not has_weapons and has_room and eff == "draw_weapon":
                return 900
            # Position at ideal weapon range
            if has_weapons:
                dist_from_node = _bfs_dist(node, G["player_pos"])
                return 500 - abs(dist_from_node - preferred_range) * 100
            # Fallback: close distance
            return -_bfs_dist(node, G["player_pos"])

        pos = G["bot_pos"]
        for _ in range(steps):
            neighbors = ADJACENCY.get(pos, [])
            if not neighbors:
                break
            best = max(neighbors, key=score_node)
            # Only move if it actually improves the score
            if score_node(best) > score_node(pos):
                pos = best
            else:
                break

        G["bot_pos"] = pos

    # ── GAME MECHANICS ────────────────────────────────────────
    def _apply_weapon_buff(self, dmg, card, char):
        attr = char["attribute"]
        sub  = card["subtype"]
        if attr == "pistol_specialist"    and sub == "pistol":        return int(dmg * 1.3)
        if attr == "shotgun_specialist"   and sub == "shotgun":       return int(dmg * 1.3)
        if attr == "rifle_specialist"     and sub in ("assault_rifle","sniper"): return int(dmg * 1.25)
        if attr == "sniper_specialist"    and sub == "sniper":        return int(dmg * 1.35)
        if attr == "explosive_specialist" and sub in ("explosive","missile"):    return int(dmg * 1.35)
        if attr == "melee_specialist"     and sub == "melee":         return int(dmg * 1.4)
        return dmg

    def _apply_location_dmg_buff(self, dmg, char, pos):
        loc = self.G["locations"][pos]
        if loc["effect"] == "hero_zone"    and char["faction"] == "hero":    return int(dmg * 1.2)
        if loc["effect"] == "villain_zone" and char["faction"] == "villain": return int(dmg * 1.2)
        return dmg

    def _apply_armor(self, dmg, attacking_card, armor_cards, defender):
        final_dmg = dmg
        for armor in [a for a in armor_cards if a["type"] == "defense" and a["healAmount"] == 0]:
            is_effective = attacking_card["subtype"] in armor["effectiveVs"]
            reduction = armor["defense"] if is_effective else int(armor["defense"] * 0.4)
            attr = defender["attribute"]
            if attr == "heavy_armor":
                reduction = int(reduction * 1.4)
            if attr == "sniper_resist" and attacking_card["subtype"] == "sniper":
                reduction = int(dmg * 0.5)
            if attr == "explosive_resist" and attacking_card["subtype"] in ("explosive","missile"):
                reduction = int(dmg * 0.4)
            actual = min(final_dmg, reduction)
            final_dmg -= actual
            if actual > 0:
                dura_lost = 1 if is_effective else (2 if dmg > 50 else 1)
                armor["durability"] -= dura_lost
                if armor["durability"] <= 0:
                    armor_cards.remove(armor)
        return {"final_dmg": max(0, final_dmg)}

    def _apply_location_effects(self):
        """Called at start of each new turn (fast phase).
        Returns shaped reward for the player only (card draws)."""
        G = self.G
        draw_reward = 0.0

        def apply_for(char, hand, in_play, pos, is_player):
            nonlocal draw_reward
            loc = G["locations"][pos]
            eff = loc["effect"]

            if eff in ("radiation", "poison"):
                dmg = 5 if eff == "radiation" else 8
                char["hp"] = max(0, char["hp"] - dmg)
                if is_player:
                    # Small per-turn penalty — much less than one attack reward (+1.65)
                    # so agent still prefers attacking over fleeing
                    draw_reward -= dmg / max(char["maxHp"], 1) * 5.0

            if eff == "healing" and char["attribute"] == "healing":
                char["hp"] = min(char["maxHp"], char["hp"] + 10)

            max_total = 5 if char["attribute"] == "extra_carry" else 4
            total_cards = len(hand) + len(in_play)

            if eff == "draw_weapon" and total_cards < max_total and G["weapon_deck"]:
                card = copy.deepcopy(G["weapon_deck"].pop(0))
                hand.append(card)
                if is_player:
                    # Base reward + bonus scaled to weapon damage
                    draw_reward += 0.10 + (card["damage"] / 100.0) * 0.10

            if eff == "draw_defense" and total_cards < max_total and G["defense_deck"]:
                card = copy.deepcopy(G["defense_deck"].pop(0))
                hand.append(card)
                if is_player:
                    if card["healAmount"] > 0:
                        # Heal cards worth more when already hurt
                        hp_ratio = char["hp"] / char["maxHp"]
                        urgency = 1.0 + (1.0 - hp_ratio)   # 1.0-2.0x multiplier
                        draw_reward += (card["healAmount"] / 100.0) * 0.5 * urgency
                    else:
                        # Armor — base reward + bonus scaled to defense value
                        draw_reward += 0.10 + (card["defense"] / 70.0) * 0.05

            # healing attribute passive
            if char["attribute"] == "healing":
                char["hp"] = min(char["maxHp"], char["hp"] + 10)

        apply_for(G["player"], G["player_hand"], G["player_in_play"], G["player_pos"], True)
        apply_for(G["bot"],    G["bot_hand"],    G["bot_in_play"],    G["bot_pos"],    False)

        return draw_reward

    def _check_win(self):
        G = self.G
        if G["player"]["hp"] <= 0:
            G["game_over"] = True
            G["winner"] = "bot"
        elif G["bot"]["hp"] <= 0:
            G["game_over"] = True
            G["winner"] = "player"

    def _terminal_reward(self):
        G = self.G
        bot_hp_ratio    = G["bot"]["hp"]  / G["bot"]["maxHp"]
        player_hp_ratio = G["player"]["hp"] / G["player"]["maxHp"]

        if G["winner"] == "player":
            return 20.0 + (player_hp_ratio * 5.0)    # +20 to +25
        elif G["winner"] == "bot":
            return -20.0 * bot_hp_ratio               # 0 to -20
        else:
            return -5.0 - (bot_hp_ratio * 15.0)       # -5 to -20

    def _best_weapon_range(self, cards):
        """Best range among held weapons (for movement reward shaping)."""
        ranges = [c["range"] for c in cards if c["type"] == "weapon"]
        return ranges[0] if ranges else 1

    # ── RENDER ────────────────────────────────────────────────
    def render(self):
        if self.render_mode != "ansi":
            return
        G = self.G
        phase = PHASES[G["phase"]]
        print(f"Turn {G['turn']} | Phase: {phase.upper()} | "
              f"Player HP: {G['player']['hp']}/{G['player']['maxHp']} "
              f"({G['player']['name']}) pos={G['player_pos']}  |  "
              f"Bot HP: {G['bot']['hp']}/{G['bot']['maxHp']} "
              f"({G['bot']['name']}) pos={G['bot_pos']}")
        print(f"  Player hand: {[c['name'] for c in G['player_hand']]}")
        print(f"  Player in-play: {[c['name'] for c in G['player_in_play']]}")
        print(f"  Distance: {_bfs_dist(G['player_pos'], G['bot_pos'])}")
