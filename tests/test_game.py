"""
Tests for the hand loop (stage plan engine_play, tracker step "Hand loop to 2nd place").

The random sweep plays random-vs-random hands in every config and checks the state after every
step: the leader is never offered a pass, the opening card is in every first play, cards are
neither lost nor made, and on small hands the move list matches the brute force on real game
positions. Each hand must end with exactly two players out and points 2 / 1 / 0.
Scripted tests then set up exact 1-deck positions for each turn-order rule in the stage plan.
Seeds start at 10: seeds 0-2 are kept for confirm runs.
"""

import random

import pytest

from helpers import brute_force_plays, choose_random, parse_cards
from stacker.cards import CONFIGS, build_deck
from stacker.game import PASS, HandState, apply_action, legal_actions, new_hand, points

# Seeds for the random sweep: 50 hands per config.
SWEEP_SEEDS = range(10, 60)

# Hands this small are cross-checked against the brute force (2^8 = 256 subsets per check).
BRUTE_FORCE_MAX_HAND = 8

SWEEP_CASES = []
for num_players in CONFIGS:
    for seed in SWEEP_SEEDS:
        SWEEP_CASES.append((num_players, seed))


def check_step(state, actions, played):
    """Every check that must hold before the seat to act chooses."""
    hand = state.hands[state.turn]

    # The seat asked to act still has cards and has not passed this round.
    assert len(hand) > 0
    assert not state.passed[state.turn]

    # A round leader is never offered a pass; a follower always is.
    if state.table is None:
        assert PASS not in actions
    else:
        assert PASS in actions

    plays = []
    for action in actions:
        if action != PASS:
            plays.append(action.cards)

    # Before the first play, every offered play contains the opening card.
    if state.opening_card is not None:
        for cards in plays:
            assert state.opening_card in cards

    # On small hands, the offered plays are exactly what the brute force finds here.
    if len(hand) <= BRUTE_FORCE_MAX_HAND:
        if state.table is None:
            previous = None
        else:
            previous = state.table.pattern
        expected = brute_force_plays(hand, state.num_decks, previous, state.opening_card)
        assert len(plays) == len(set(plays))
        assert set(plays) == set(expected)

    # No card is lost or made: hands + played + aside is the full deck.
    all_cards = list(state.aside) + list(played)
    for seat_hand in state.hands:
        all_cards.extend(seat_hand)
    assert sorted(all_cards) == build_deck(state.num_decks)


@pytest.mark.parametrize("num_players, seed", SWEEP_CASES)
def test_random_hand_follows_the_rules(num_players, seed):
    rng = random.Random(seed)
    state = new_hand(num_players, rng)
    played = []
    while state.turn is not None:
        actions = legal_actions(state)
        check_step(state, actions, played)
        action = choose_random(state, actions, rng)
        if action != PASS:
            played.extend(action.cards)
        apply_action(state, action)

    # Exactly two players are out, the rest still hold cards.
    assert len(state.finish_order) == 2
    for seat in range(num_players):
        if seat in state.finish_order:
            assert len(state.hands[seat]) == 0
        else:
            assert len(state.hands[seat]) > 0

    # One 2, one 1, everyone else 0, and the 2 goes to the first player out.
    scores = points(state)
    expected_scores = [2, 1]
    for _ in range(num_players - 2):
        expected_scores.append(0)
    assert sorted(scores, reverse=True) == expected_scores
    assert scores[state.finish_order[0]] == 2
    assert scores[state.finish_order[1]] == 1


def scripted_state(hand_texts, leader_seat, opening_text=None):
    """
    A 1-deck position with exact hands and nothing aside.
    opening_text: the card the first play must include, like '3s', or None for a mid-hand position.
    """
    hands = []
    for hand_text in hand_texts:
        hands.append(parse_cards(hand_text))
    if opening_text is None:
        opening_card = None
    else:
        opening_card = parse_cards(opening_text)[0]
    return HandState(hands, 1, leader_seat, opening_card, [])


def act(state, move):
    """
    Plays one move for the seat to act: PASS, or cards written like '6s 6h'.
    Fails unless the move is among legal_actions, so scripts can only take legal steps.
    """
    actions = legal_actions(state)
    if move == PASS:
        assert PASS in actions
        apply_action(state, PASS)
        return

    cards = tuple(sorted(parse_cards(move)))
    for action in actions:
        if action != PASS and action.cards == cards:
            apply_action(state, action)
            return
    raise AssertionError(f"{move} is not a legal action for seat {state.turn}")


def test_leader_cannot_pass():
    state = scripted_state(["5s 9h", "6s 9c", "7s 9d", "8s 9s"], leader_seat=0)
    assert PASS not in legal_actions(state)


def test_passed_seat_is_skipped_even_holding_a_bomb():
    # Seat 1 holds a quad of 6s (a bomb) but passes; it must not act again this round.
    state = scripted_state(["5s 9h", "6s 6c 6d 6h 9c", "7s 9d", "8s 9s"], leader_seat=0)
    act(state, "5s")
    act(state, PASS)
    assert state.turn == 2
    act(state, "7s")
    act(state, "8s")
    assert state.turn == 0
    act(state, PASS)
    # Around the table from seat 0, seat 1 is skipped: it passed, bomb or not.
    assert state.turn == 2
    assert state.passed[1]


def test_round_winner_leads_next():
    state = scripted_state(["5s 9h", "6s 9c", "7s 9d", "8s 9s"], leader_seat=0)
    act(state, "5s")
    act(state, "9c")
    act(state, PASS)
    act(state, PASS)
    act(state, PASS)
    # Seat 1's 9 was not beaten: a new round starts with seat 1 leading and passes cleared.
    assert state.turn == 1
    assert state.table is None
    assert state.passed == [False, False, False, False]


def test_unbeaten_last_card_passes_the_lead_to_the_next_seat_with_cards():
    # Seat 1 goes out on an unbeaten 2. The lead goes forward to seat 2, not back to seat 0.
    state = scripted_state(["4s 9h", "2h", "5s 9d", "6s 9s"], leader_seat=0)
    act(state, "4s")
    act(state, "2h")
    assert state.finish_order == [1]
    act(state, PASS)
    act(state, PASS)
    act(state, PASS)
    assert state.turn == 2
    assert state.table is None


def test_beaten_last_card_keeps_the_round_going():
    # Seat 1 goes out on a 9, but seat 2 beats it, so the round goes on to seat 3.
    state = scripted_state(["4s 8h", "9h", "2h 5s", "6s 7s"], leader_seat=0)
    act(state, "4s")
    act(state, "9h")
    act(state, "2h")
    assert state.turn == 3
    assert state.table_seat == 2


def test_hand_stops_at_second_player_out():
    state = scripted_state(["3s", "4s", "5s 9d", "6s 9s"], leader_seat=0, opening_text="3s")
    act(state, "3s")
    act(state, "4s")
    # Two players are out: nobody acts again, and seats 2 and 3 keep their cards.
    assert state.turn is None
    assert state.finish_order == [0, 1]
    assert points(state) == [2, 1, 0, 0]


def test_opening_play_must_include_opening_card():
    state = scripted_state(["3s 4s 5h 6h 7s", "8s 9c", "8c 9d", "8d 9s"], leader_seat=0, opening_text="3s")
    actions = legal_actions(state)
    for action in actions:
        assert parse_cards("3s")[0] in action.cards
    # The run 3-4-5 holds 3♠ and is offered; the lone 4♠ does not and is refused.
    offered_cards = []
    for action in actions:
        offered_cards.append(action.cards)
    assert tuple(parse_cards("3s 4s 5h")) in offered_cards
    with pytest.raises(AssertionError):
        act(state, "4s")


def test_bomb_can_lead_the_opening_and_only_a_higher_bomb_answers():
    state = scripted_state(
        ["3s 3c 3d 3h 9h", "4s 4c 4d 4h 5s 6s 7s 2h", "8c 9d", "8d 9s"],
        leader_seat=0,
        opening_text="3s",
    )
    act(state, "3s 3c 3d 3h")
    # Seat 1 holds a single 2 and a run, but only its bomb of 4s can beat a bomb of 3s.
    offered = []
    for action in legal_actions(state):
        if action == PASS:
            offered.append(PASS)
        else:
            offered.append(action.cards)
    assert offered == [tuple(parse_cards("4s 4c 4d 4h")), PASS]
