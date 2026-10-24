"""
Tests for legal-move enumeration (stage plan engine_play, tracker step "Legal-move enumeration").

The fast generator (moves.legal_plays) must list exactly the plays the slow brute force finds
(every subset of the hand, filtered by classify and beats), with no duplicates. One small hand
is also counted by hand, and every opening move on real deals must contain the opening card.
Seeds start at 10: seeds 0-2 are kept for confirm runs.
"""

import random

import pytest

from helpers import brute_force_plays, classify_text, parse_cards
from stacker.cards import CONFIGS, deal
from stacker.moves import legal_plays

# A 1-deck hand with a quad, pairs, runs, pair multi-runs and a 6-7.
ONE_DECK_HAND = "3s 3c 3d 3h 4s 4c 5s 5d 6h 7s 7c 8s 9d"

# A 2-deck hand with five 3s (a bomb), bad copies, pairs and a run.
TWO_DECK_HAND = "3s 3s' 3c 3c' 3d 4s 4h 5s 5h' 6c 7d 7d' 9h 2h"


def card(text):
    return parse_cards(text)[0]


def assert_matches_brute_force(hand_text, num_decks, previous, opening_card):
    """The generator and the brute force agree exactly, and the generator has no duplicates."""
    hand = parse_cards(hand_text)
    generated = []
    for play in legal_plays(hand, num_decks, previous, opening_card):
        generated.append(play.cards)
    expected = brute_force_plays(hand, num_decks, previous, opening_card)
    assert len(generated) == len(set(generated))
    assert set(generated) == set(expected)


# Each case: (what is on the table as text, whether it was a lead), or None for a round lead.
ONE_DECK_TABLE = [
    None,
    ("4h", False),
    ("4d 4h", False),
    ("4h 5h 6s", False),
    ("4d 4h 5c 5h 6c 6d", False),
    ("6d 7d", True),
    ("Ks Kc Kd Kh", False),
]


@pytest.mark.parametrize("table", ONE_DECK_TABLE)
def test_generator_matches_brute_force_one_deck(table):
    if table is None:
        previous = None
    else:
        previous = classify_text(table[0], num_decks=1, leading=table[1])
    assert_matches_brute_force(ONE_DECK_HAND, 1, previous, None)


def test_generator_matches_brute_force_one_deck_opening():
    assert_matches_brute_force(ONE_DECK_HAND, 1, None, card("3s"))


TWO_DECK_TABLE = [
    None,
    ("4h", False),
    ("4d 4h 5c 5h 6c 6d", False),
    ("Ks Kc Kd Kh", False),
    ("Ks Kc Kd Kh Ks'", False),
]


@pytest.mark.parametrize("table", TWO_DECK_TABLE)
def test_generator_matches_brute_force_two_decks(table):
    if table is None:
        previous = None
    else:
        previous = classify_text(table[0], num_decks=2, leading=table[1])
    assert_matches_brute_force(TWO_DECK_HAND, 2, previous, None)


def test_generator_matches_brute_force_two_decks_opening():
    # With 2 decks the bad 3♠ is the lowest card, so it is the opening card.
    assert_matches_brute_force(TWO_DECK_HAND, 2, None, card("3s'"))


def test_small_opening_hand_counted_by_hand():
    hand = parse_cards("3s 3c 4s 5s 6s")
    generated = set()
    for play in legal_plays(hand, 1, None, card("3s")):
        generated.add(play.cards)
    expected = set()
    for text in ("3s", "3s 3c", "3s 4s 5s", "3s 4s 5s 6s"):
        expected.add(tuple(parse_cards(text)))
    assert generated == expected


def test_every_opening_move_contains_the_opening_card():
    for num_players, config in CONFIGS.items():
        for seed in range(10, 30):
            result = deal(num_players, random.Random(seed))
            leader_hand = result.hands[result.leader_seat]
            plays = legal_plays(leader_hand, config.num_decks, None, result.opening_card)
            assert len(plays) > 0
            for play in plays:
                assert result.opening_card in play.cards
