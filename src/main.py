"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from src.recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv") 

    # Taste profile: late-night study session listener
    # Prefers calm, slightly melancholic lo-fi with acoustic texture,
    # moderate tempo, and enough rhythm to stay focused but not distracted.
    user_prefs = {
        "favorite_genre":    "lofi",      # core genre identity
        "favorite_mood":     "focused",   # contextual intent: studying/working
        "target_energy":     0.40,        # low-to-mid — calm but not sleepy
        "target_valence":    0.55,        # slightly below neutral — bittersweet, not sad
        "target_acousticness": 0.75,      # strongly prefers organic, warm textures
        "target_danceability": 0.60,      # light groove — enough rhythm to keep focus
        "target_tempo_bpm":  80,          # slow-to-mid BPM, unhurried pace
        "likes_acoustic":    True,        # binary flag kept for OOP UserProfile compat
    }

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\nTop recommendations:\n")
    for rec in recommendations:
        # You decide the structure of each returned item.
        # A common pattern is: (song, score, explanation)
        song, score, explanation = rec
        print(f"{song['title']} - Score: {score:.2f}")
        print(f"Because: {explanation}")
        print()


if __name__ == "__main__":
    main()
