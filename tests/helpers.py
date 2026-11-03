"""
Test helpers: write cards as short strings so tests read like the home rules doc.

"3s 10h 2h'" means 3♠, 10♥ and the bad copy of 2♥. parse_cards itself lives in
stacker.cards, since scripts/play_human.py types cards in too; it is imported here so tests
keep importing it from helpers.
"""

from itertools import combinations

from stacker.cards import parse_cards
from stacker.patterns import beats, classify


def classify_text(text, num_decks=1, leading=False):
    """Parse cards written like '3s 4s 5h' and classify them."""
    return classify(parse_cards(text), num_decks, leading)


def brute_force_plays(hand, num_decks, previous, opening_card):
    """
    Slow reference for moves.legal_plays: tries every non-empty subset of the hand.
    Same inputs as legal_plays. Returns a list of card tuples, lowest card first.
    Too slow for the game loop (2^n subsets), but plainly correct, so tests compare against it.
    """
    leading = previous is None
    plays = []
    for size in range(1, len(hand) + 1):
        for cards in combinations(sorted(hand), size):
            pattern = classify(list(cards), num_decks, leading)
            if pattern is None:
                continue
            if opening_card is not None and opening_card not in cards:
                continue
            if previous is not None and not beats(pattern, previous):
                continue
            plays.append(cards)
    return plays
