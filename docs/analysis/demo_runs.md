# CineMatch agent - worked examples (deterministic engine layer)

## Case 1: Crime fan, recent hits
answers={'genre': 'crime', 'era': 'recent', 'popularity': 'well_known'}
  - Scam 1992: The Harshad Mehta Story | Biography, Crime, Drama | 2020 | rating 9.2
  - Dexter: Resurrection | Crime, Drama, Thriller | 2025 | rating 9.0
  - The Penguin | Crime, Drama | 2024 | rating 8.6

## Case 2: Light comedy, short
answers={'genre': 'comedy', 'length': 'short'}
  - Goblin | Comedy, Drama, Sci-Fi & Fantasy | 2016 | rating 8.701
  - TONIKAWA: Over the Moon for You | Animation, Comedy | 2020 | rating 8.544
  - Ginny & Georgia | Comedy, Drama | 2021 | rating 8.139

## Case 3: Hidden-gem sci-fi
answers={'genre': 'scifi_fantasy', 'popularity': 'hidden_gem'}
  - Goblin | Comedy, Drama, Sci-Fi & Fantasy | 2016 | rating 8.701
  - High School D×D | Action & Adventure, Animation, Comedy, Sci-Fi & Fantasy | 2012 | rating 8.7
  - Love Alarm | Drama, Sci-Fi & Fantasy | 2019 | rating 8.4

## Case 4: Surprise me (no preferences)
answers={}
  - Planet Earth II | Documentary, Family | 2016 | rating 9.4
  - Chernobyl | Drama, History, Thriller | 2019 | rating 9.3
  - Game of Thrones | Action, Adventure, Drama | 2011 | rating 9.2

## Case 5 (failure): request the engine cannot satisfy
When no catalog title matches the constraints, the ranker returns an empty result and the API replies with the graceful 'no recommendations / try rephrasing' message instead of inventing a title.
  (no match)
