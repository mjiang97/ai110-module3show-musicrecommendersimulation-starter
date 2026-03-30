import csv
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py

    Required fields (no defaults) must be listed first so that the dataclass
    stays compatible with the 4-argument test fixtures.

    Optional numeric targets are used with Gaussian proximity scoring.
    When None, that feature is skipped — the required target_energy is always scored.
    """
    favorite_genre:      str    # e.g. "lofi", "jazz", "edm"
    favorite_mood:       str    # e.g. "focused", "happy", "melancholic"
    target_energy:       float  # [0.0 – 1.0]  calm ←→ intense
    likes_acoustic:      bool   # convenience flag: True when target_acousticness > 0.5

    # Optional extended profile (safe to omit for backwards-compat with tests)
    target_valence:      Optional[float] = None  # [0.0 – 1.0]  dark ←→ uplifting
    target_acousticness: Optional[float] = None  # [0.0 – 1.0]  electronic ←→ acoustic
    target_danceability: Optional[float] = None  # [0.0 – 1.0]  ambient ←→ groove-forward
    target_tempo_bpm:    Optional[float] = None  # absolute BPM, normalised internally


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

# Maximum points each feature can contribute — controls relative importance.
#
#   Genre match  : +2.0   (hard structural preference, most stable over time)
#   Mood match   : +1.0   (context-dependent, worth less than genre)
#   Energy       : up to +2.5  (strongest predictor of listening context)
#   Acousticness : up to +2.0  (sharp acoustic/electronic axis)
#   Valence      : up to +1.8  (emotional tone)
#   Tempo        : up to +1.2  (physical pace)
#   Danceability : up to +0.8  (secondary signal, correlated with energy)
#
# Max possible score (all features present): 11.3

_WEIGHTS = {
    "energy":        2.5,
    "acousticness":  2.0,
    "valence":       1.8,
    "tempo":         1.2,
    "danceability":  0.8,
}

# Sigma values control tolerance — how quickly the score drops off from target.
# Tighter sigma = punish small deviations more severely.
_SIGMA = {
    "energy":        0.15,
    "acousticness":  0.15,
    "valence":       0.20,
    "tempo":         0.10,   # applied after normalising to [0, 1]
    "danceability":  0.20,
}

# Threshold: only mention a feature in the explanation when its score is
# at least this fraction of its maximum weight.
_EXPLAIN_THRESHOLD = 0.80


def _gauss(song_val: float, target: float, sigma: float) -> float:
    """Gaussian proximity — returns 1.0 at a perfect match, decays toward 0."""
    return math.exp(-((song_val - target) ** 2) / (2 * sigma ** 2))


def score_song(song: Dict, user_prefs: Dict) -> Tuple[float, str]:
    """
    Score a single song dict against a user preference dict.

    Algorithm recipe
    ----------------
    1. +2.0 for a genre match (binary)
    2. +1.0 for a mood match  (binary)
    3. Up to +2.5 for energy proximity   (Gaussian, σ=0.15)
    4. Up to +2.0 for acousticness proximity (Gaussian, σ=0.15)
    5. Up to +1.8 for valence proximity  (Gaussian, σ=0.20)
    6. Up to +1.2 for tempo proximity    (Gaussian, σ=0.10, after BPM normalisation)
    7. Up to +0.8 for danceability proximity (Gaussian, σ=0.20)

    Returns
    -------
    (score, explanation)  where score is a float and explanation is a plain-
    English string describing which features drove the recommendation.
    """
    score = 0.0
    reasons: List[str] = []

    # --- 1. Genre match (+2.0) ---
    if song.get("genre") == user_prefs.get("favorite_genre"):
        score += 2.0
        reasons.append(f"genre match ({song['genre']})")

    # --- 2. Mood match (+1.0) ---
    if song.get("mood") == user_prefs.get("favorite_mood"):
        score += 1.0
        reasons.append(f"mood match ({song['mood']})")

    # --- 3. Energy proximity (always scored) ---
    energy_pts = _gauss(song["energy"], user_prefs["target_energy"], _SIGMA["energy"]) * _WEIGHTS["energy"]
    score += energy_pts
    if energy_pts >= _WEIGHTS["energy"] * _EXPLAIN_THRESHOLD:
        reasons.append(
            f"energy close match ({song['energy']:.2f} vs target {user_prefs['target_energy']:.2f})"
        )

    # --- 4. Acousticness proximity (optional) ---
    if user_prefs.get("target_acousticness") is not None:
        ac_pts = _gauss(song["acousticness"], user_prefs["target_acousticness"], _SIGMA["acousticness"]) * _WEIGHTS["acousticness"]
        score += ac_pts
        if ac_pts >= _WEIGHTS["acousticness"] * _EXPLAIN_THRESHOLD:
            reasons.append(
                f"acousticness close match ({song['acousticness']:.2f} vs target {user_prefs['target_acousticness']:.2f})"
            )

    # --- 5. Valence proximity (optional) ---
    if user_prefs.get("target_valence") is not None:
        val_pts = _gauss(song["valence"], user_prefs["target_valence"], _SIGMA["valence"]) * _WEIGHTS["valence"]
        score += val_pts
        if val_pts >= _WEIGHTS["valence"] * _EXPLAIN_THRESHOLD:
            reasons.append(
                f"valence close match ({song['valence']:.2f} vs target {user_prefs['target_valence']:.2f})"
            )

    # --- 6. Tempo proximity (optional) — normalised to [0, 1] before Gaussian ---
    if user_prefs.get("target_tempo_bpm") is not None:
        song_tempo_norm = song["tempo_bpm"] / 200.0
        user_tempo_norm = user_prefs["target_tempo_bpm"] / 200.0
        tempo_pts = _gauss(song_tempo_norm, user_tempo_norm, _SIGMA["tempo"]) * _WEIGHTS["tempo"]
        score += tempo_pts
        if tempo_pts >= _WEIGHTS["tempo"] * _EXPLAIN_THRESHOLD:
            reasons.append(f"tempo close match ({song['tempo_bpm']:.0f} BPM)")

    # --- 7. Danceability proximity (optional) ---
    if user_prefs.get("target_danceability") is not None:
        dance_pts = _gauss(song["danceability"], user_prefs["target_danceability"], _SIGMA["danceability"]) * _WEIGHTS["danceability"]
        score += dance_pts
        if dance_pts >= _WEIGHTS["danceability"] * _EXPLAIN_THRESHOLD:
            reasons.append(
                f"danceability close match ({song['danceability']:.2f} vs target {user_prefs['target_danceability']:.2f})"
            )

    explanation = (
        "Recommended because: " + ", ".join(reasons)
        if reasons
        else "Partial feature match — no single feature scored above threshold"
    )
    return round(score, 4), explanation


# ---------------------------------------------------------------------------
# OOP interface (used by tests)
# ---------------------------------------------------------------------------

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """

    def __init__(self, songs: List[Song]):
        self.songs = songs

    def _song_to_dict(self, song: Song) -> Dict:
        return {
            "id":           song.id,
            "title":        song.title,
            "artist":       song.artist,
            "genre":        song.genre,
            "mood":         song.mood,
            "energy":       song.energy,
            "tempo_bpm":    song.tempo_bpm,
            "valence":      song.valence,
            "danceability": song.danceability,
            "acousticness": song.acousticness,
        }

    def _profile_to_dict(self, user: UserProfile) -> Dict:
        return {
            "favorite_genre":    user.favorite_genre,
            "favorite_mood":     user.favorite_mood,
            "target_energy":     user.target_energy,
            "target_valence":    user.target_valence,
            "target_acousticness": user.target_acousticness,
            "target_danceability": user.target_danceability,
            "target_tempo_bpm":  user.target_tempo_bpm,
        }

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        user_dict = self._profile_to_dict(user)
        scored = [
            (song, score_song(self._song_to_dict(song), user_dict)[0])
            for song in self.songs
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [song for song, _ in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        user_dict = self._profile_to_dict(user)
        _, explanation = score_song(self._song_to_dict(song), user_dict)
        return explanation


# ---------------------------------------------------------------------------
# Functional interface (used by main.py)
# ---------------------------------------------------------------------------

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    songs: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                "id":           int(row["id"]),
                "title":        row["title"],
                "artist":       row["artist"],
                "genre":        row["genre"],
                "mood":         row["mood"],
                "energy":       float(row["energy"]),
                "tempo_bpm":    float(row["tempo_bpm"]),
                "valence":      float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
            })
    return songs


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py

    Returns a list of (song_dict, score, explanation) tuples, sorted by score descending.
    """
    scored = [(song, *score_song(song, user_prefs)) for song in songs]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
