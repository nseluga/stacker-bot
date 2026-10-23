"""
Tests for card order (architecture plan §2; stage plan engine_rules, tracker step "Card and ordering").

Card order is rank first (3 lowest, 2 highest), then suit (♠ < ♣ < ♦ < ♥), then deck:
the bad copy sorts just below the good copy of the same card.
"""

from helpers import parse_cards


def test_three_of_spades_is_lowest():
    hand = parse_cards("2h 3c 3s Ah 4s 3h 3d")
    assert min(hand) == parse_cards("3s")[0]


def test_good_two_of_hearts_is_highest_with_two_decks():
    hand = parse_cards("2h' 2h Ah 2d 2d' Ah'")
    assert max(hand) == parse_cards("2h")[0]


def test_good_copy_beats_bad_copy():
    assert parse_cards("3s")[0] > parse_cards("3s'")[0]


def test_rank_outranks_suit():
    # A higher rank wins even against the highest suit.
    assert parse_cards("4s")[0] > parse_cards("3h")[0]


def test_suit_outranks_deck():
    # A higher suit wins even when it is the bad copy.
    assert parse_cards("3c'")[0] > parse_cards("3s")[0]
