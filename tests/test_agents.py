"""
Tests for the baseline agents (stage plan baselines, pieces 2 and 5).

Greedy: scripted positions check its exact answer where the rule is easy to see by hand (lowest
single when leading, lowest beating pair, a bomb only when nothing else beats, the tie-break, PASS
only when forced). One property test then checks the rule itself, not greedy's sort key, at every
turn of real hands: so a broken key fails it even in positions no scripted test covers.
Seeds start at 10: seeds 0-2 are kept for confirm runs.
"""

import random

import pytest
from helpers import parse_cards

from stacker.agents import greedy_agent
from stacker.cards import CONFIGS
from stacker.game import PASS, HandState, legal_actions, new_hand, observe, play_out
from stacker.moves import Play
from stacker.patterns import BOMB, classify

# Hands per config in the property test: thousands of greedy turns in total, about a second.
PROPERTY_HANDS = 50


def position(hand_text, table_text):
    """
    A 4-player, 1-deck HandState with seat 0 to act holding hand_text.
    table_text None: seat 0 leads a round. Otherwise seat 1 just played table_text.
    The other seats hold one card each; greedy never looks at them.
    """
    hands = [parse_cards(hand_text), parse_cards("Ad"), parse_cards("Ac"), parse_cards("As")]
    state = HandState(hands, 1, 0, None, [])
    if table_text is not None:
        table_cards = parse_cards(table_text)
        state.table = Play(tuple(table_cards), classify(table_cards, 1, False))
        state.table_seat = 1
    return state


def greedy_choice(state):
    """greedy_agent's action in this position, called the way play_out calls it."""
    return greedy_agent(observe(state, []), legal_actions(state), random.Random(10))


@pytest.mark.parametrize(
    "hand_text, table_text, expected_text",
    [
        # Leading: the lowest single beats every other play on top card, then card count.
        ("4s 5d 5h 9s 9h 9c 9d Kh", None, "4s"),
        # Following a pair: 5d 5h and 6c 6d both beat 5s 5c; the lower one is played.
        ("3s 5d 5h 6c 6d 9s 9h 9c 9d Kh", "5s 5c", "5d 5h"),
        # Only the bomb beats a pair of Ks, so greedy spends it.
        ("3s 4h 9s 9h 9c 9d", "Ks Kh", "9s 9h 9c 9d"),
        # A pair of 9s and the bomb both beat 7s: the normal play comes first, the bomb is kept.
        ("3s 4h 9s 9h 9c 9d", "7h 7d", "9s 9c"),
        # Two runs tie on top card 5h and 3 cards: the lower cards are spent, 3c is kept.
        ("3s 3c 4s 4c 5h", "3d 4d 5d", "3s 4s 5h"),
    ],
)
def test_greedy_scripted_positions(hand_text, table_text, expected_text):
    choice = greedy_choice(position(hand_text, table_text))
    assert choice != PASS
    assert sorted(choice.cards) == sorted(parse_cards(expected_text))


def test_greedy_passes_when_nothing_beats():
    state = position("3s 4s", "Ks Kh")
    assert legal_actions(state) == [PASS]
    assert greedy_choice(state) == PASS


@pytest.mark.parametrize("num_players", list(CONFIGS))
def test_greedy_follows_its_rule_in_real_hands(num_players):
    turns_checked = []

    def checking_agent(observation, actions, rng):
        choice = greedy_agent(observation, actions, rng)
        assert choice in actions

        normal_plays = []
        bombs = []
        for action in actions:
            if action == PASS:
                continue
            if action.pattern.kind == BOMB:
                bombs.append(action)
            else:
                normal_plays.append(action)

        if len(normal_plays) + len(bombs) == 0:
            assert choice == PASS
        elif choice == PASS:
            pytest.fail("greedy passed while holding a legal play")
        elif choice.pattern.kind == BOMB:
            # A bomb only when no normal play was legal.
            assert normal_plays == []
        else:
            # A normal play: none of the other normal plays has a lower top card.
            for play in normal_plays:
                assert play.pattern.top_card >= choice.pattern.top_card

        turns_checked.append(choice)
        return choice

    agents = []
    for _ in range(num_players):
        agents.append(checking_agent)
    deal_rng = random.Random(10)
    for _ in range(PROPERTY_HANDS):
        play_out(new_hand(num_players, deal_rng), agents, random.Random(11))

    assert len(turns_checked) > 0
