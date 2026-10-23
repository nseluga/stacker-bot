"""
Tests for dealing (architecture plan §2; stage plan engine_rules, tracker step "Deal").

Each player count has a fixed table setup (CONFIGS). A deal must hand out every card of
the deck exactly once, replay exactly from the same seed, and give the lead to whoever
holds the lowest dealt card (the bad copy leads when both copies are dealt).
Seeds start at 10: seeds 0-2 are kept for confirm runs.
"""

import random

import pytest

from helpers import parse_cards
from stacker.cards import CONFIGS, build_deck, deal, find_opening


def test_hand_and_aside_sizes_match_config():
    for num_players, config in CONFIGS.items():
        result = deal(num_players, random.Random(10))
        assert len(result.hands) == num_players
        for hand in result.hands:
            assert len(hand) == config.hand_size
        deck_size = len(build_deck(config.num_decks))
        assert len(result.aside) == deck_size - num_players * config.hand_size


def test_every_card_dealt_or_aside_exactly_once():
    # 4 players use one deck, 7 players use two.
    for num_players in (4, 7):
        result = deal(num_players, random.Random(10))
        all_cards = list(result.aside)
        for hand in result.hands:
            all_cards.extend(hand)
        expected = build_deck(CONFIGS[num_players].num_decks)
        assert sorted(all_cards) == sorted(expected)


def test_same_seed_gives_same_deal():
    first = deal(5, random.Random(10))
    second = deal(5, random.Random(10))
    assert first == second


def test_leader_holds_lowest_dealt_card():
    # Many seeds, so some deals put the lowest card (3♠ or 3♠') aside and the lead
    # falls to the next card up.
    for seed in range(10, 60):
        for num_players in CONFIGS:
            result = deal(num_players, random.Random(seed))
            assert result.opening_card in result.hands[result.leader_seat]
            for hand in result.hands:
                for card in hand:
                    assert card >= result.opening_card


def test_lead_falls_to_next_card_when_three_of_spades_is_aside():
    hands = [parse_cards("4s 3c"), parse_cards("5h 3d")]
    assert find_opening(hands) == (0, parse_cards("3c")[0])


def test_bad_copy_leads_over_good_copy():
    hands = [parse_cards("3s"), parse_cards("3s'")]
    assert find_opening(hands) == (1, parse_cards("3s'")[0])


def test_unsupported_player_count_raises():
    for num_players in (2, 9):
        with pytest.raises(ValueError):
            deal(num_players, random.Random(10))
