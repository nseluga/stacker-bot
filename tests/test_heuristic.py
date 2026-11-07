"""
Tests for the heuristic agent (stage plan baselines, piece 8; decision log 2026-10-31, amended
2026-11-01 and 2026-11-02).

Each test is a hand-built position, small enough to read at a glance. Positions are chosen so
every beat chance is far from the 0.2 control line (exactly 0, or close to 1), so the answer never
depends on the sampling's random seed. Rule tests assert both the play and the rule name, so a
position that lands on the right play through the wrong rule still fails.
Seat 0 is always the agent under test. Dev seed 10 throughout: seeds 0-2 are kept for confirm runs.
"""

import random

from helpers import classify_text
from stacker.cards import parse_cards
from stacker.game import PASS, Observation, new_hand, play_out
from stacker.heuristic import (
    Part,
    beat_chance,
    best_split,
    choose_follow,
    choose_lead,
    find_threat,
    heuristic_agent,
)
from stacker.moves import Play, legal_plays
from stacker.patterns import classify

# Seat that every position is built for.
OUR_SEAT = 0

# Opponent card counts for a quiet table: nobody near going out.
NO_THREAT = (8, 8, 8)


def single_play(card):
    """A one-card Play, used to record cards as already played in a position's history."""
    return Play((card,), classify([card], 1, True))


def position(hand, opponents, table=None, table_seat=None, passed=None, played=""):
    """
    Builds a 1-deck Observation for seat 0, plus its legal actions, from short card strings.
    hand: our cards, e.g. "3s 4s 5h". opponents: card counts for seats 1, 2, ... in order.
    table: the play to beat, as a card string, or None when we lead. table_seat: who played it.
    passed: one bool per seat (ours included); default nobody has passed.
    played: cards already out of the game, as a card string. They go into the history so the
    agent counts them as seen, like the table play itself.
    Returns (observation, actions): actions are legal_plays, plus PASS when following.
    """
    our_cards = parse_cards(hand)
    cards_left = (len(our_cards),) + tuple(opponents)
    if passed is None:
        passed = (False,) * len(cards_left)

    # Seen cards: each played card as a single by seat 1 (only its cards matter to the agent),
    # then the table play by the seat that made it.
    history = []
    for card in parse_cards(played):
        history.append((1, single_play(card)))
    table_play = None
    if table is not None:
        table_cards = parse_cards(table)
        table_play = Play(tuple(sorted(table_cards)), classify(table_cards, 1, True))
        history.append((table_seat, table_play))

    observation = Observation(
        seat=OUR_SEAT,
        hand=tuple(sorted(our_cards)),
        num_decks=1,
        table=table_play,
        table_seat=table_seat,
        passed=tuple(passed),
        cards_left=cards_left,
        finish_order=(),
        opening_card=None,
        history=tuple(history),
    )

    # The same actions the game loop would offer: legal plays, then PASS when a play is down.
    if table_play is None:
        actions = legal_plays(list(our_cards), 1, None, None)
    else:
        actions = legal_plays(list(our_cards), 1, table_play.pattern, None)
        actions.append(PASS)
    return observation, actions


def cards(text):
    """The sorted card tuple a Play holds, from a card string, to compare against a chosen play."""
    return tuple(sorted(parse_cards(text)))


def lead(hand, opponents, played=""):
    """choose_lead on a lead position; returns (cards played, rule name)."""
    observation, actions = position(hand, opponents, played=played)
    play, rule = choose_lead(observation, actions, random.Random(10))
    return play.cards, rule


def follow(hand, opponents, table, table_seat, passed=None):
    """choose_follow on a follow position; returns (cards played or PASS, rule name)."""
    observation, actions = position(hand, opponents, table, table_seat, passed)
    action, rule = choose_follow(observation, actions, random.Random(10))
    if action == PASS:
        return PASS, rule
    return action.cards, rule


# ---- Features ------------------------------------------------------------------------------------


def test_beat_chance_is_zero_when_no_unseen_card_beats():
    # 2♥ is the highest card, so no deal of the unseen cards can beat it.
    unseen = parse_cards("3s 9c Kd As 2s")
    assert beat_chance(classify_text("2h", leading=True), unseen, [3, 2], random.Random(10)) == 0.0


def test_beat_chance_is_one_when_every_hand_beats():
    # Every unseen card beats 3♠, so every sampled hand holds a beater.
    unseen = parse_cards("4s 4c 4d 4h 5s 5c 5d 5h 6s 6c")
    pattern = classify_text("3s", leading=True)
    assert beat_chance(pattern, unseen, [5, 5], random.Random(10)) == 1.0


def test_beat_chance_ignores_opponents_too_small_to_answer():
    # Plenty of higher pairs are unseen, but a 1-card hand cannot answer a pair.
    unseen = parse_cards("4s 4c 5s 5c 6s 6c")
    pattern = classify_text("3s 3c", leading=True)
    assert beat_chance(pattern, unseen, [1], random.Random(10)) == 0.0


def test_best_split_finds_fewest_plays():
    # A 5-card run 3-7 plus a pair of 9s: two plays, not seven singles.
    split = best_split(parse_cards("3s 4c 5d 6h 7s 9s 9c"))
    assert split == (Part(first_rank=0, num_ranks=5, group_size=1), Part(6, 1, 2))


def test_best_split_prefers_fewer_singles_on_a_tie():
    # 33 44 5 and 345 3 4 are both three plays; the first leaves one single, not two.
    split = best_split(parse_cards("3s 3c 4s 4c 5s"))
    assert split == (Part(0, 1, 2), Part(1, 1, 2), Part(2, 1, 1))


def test_threat_at_table_seat_makes_us_last_defender():
    # Seat 1 led and holds 2 cards; seats 2 and 3 passed. If we pass, seat 1 leads again.
    observation, _actions = position(
        "8s 9s", [2, 6, 6], table="5s", table_seat=1, passed=(False, False, True, True)
    )
    assert find_threat(observation) == (True, True)


def test_threat_behind_another_defender():
    # Seat 3 led; seat 1 (6 cards) acts before seat 2 (2 cards), so seat 1 can stop it too.
    observation, _actions = position("8s 9s", [6, 2, 6], table="5s", table_seat=3)
    assert find_threat(observation) == (True, False)


def test_passed_seat_is_not_a_threat():
    # Seat 1 holds 2 cards but passed earlier this round, so it cannot go out before the round ends.
    observation, _actions = position(
        "8s 9s", [2, 6, 6], table="5s", table_seat=2, passed=(False, True, False, True)
    )
    assert find_threat(observation) == (False, False)


# ---- Lead rules ----------------------------------------------------------------------------------


def test_lead_goes_out_with_whole_hand():
    # The whole hand is one pair: lead it and win.
    assert lead("9s 9c", NO_THREAT) == (cards("9s 9c"), "out")


def test_lead_finish_leads_control_and_keeps_the_loser():
    # 2♥ cannot be beaten, 3♠ is the one loser: lead 2♥, win the lead back, then go out with 3♠.
    assert lead("3s 2h", NO_THREAT) == (cards("2h"), "R7")


def test_lead_finish_comes_before_danger():
    # Seat 1 holds 1 card, but 2♥ wins the lead and the 44 left can't be answered by one card.
    # R7 (2♥) must win over R6b, which would lead the losing pair first.
    assert lead("4s 4c 2h", (1, 5, 5)) == (cards("2h"), "R7")


def test_lead_danger_leads_a_pair_past_a_one_card_opponent():
    # Seat 1 holds 1 card and three losers are left: a pair cannot be answered by one card.
    assert lead("4s 4c 6s 9d", (1, 5, 5)) == (cards("4s 4c"), "R6b")


def test_lead_danger_with_only_singles_leads_the_highest():
    # Only singles left: the highest is the least likely to let the 1-card seat out.
    assert lead("4s 6s 9d", (1, 5, 5)) == (cards("9d"), "R6b")


def test_lead_setup_leads_the_control_pair_first():
    # The other 2s are played, so AA (A♥ top) and 2♥ cannot be beaten. Losers 3♠ and 5♦ are two,
    # and 2♥ wins one lead back: a finish plan. Lead the pair, which only wins when we lead it.
    assert lead("3s 5d Ad Ah 2h", NO_THREAT, played="2s 2c 2d") == (cards("Ad Ah"), "R8")


def test_lead_lowest_leads_the_lowest_loser():
    # Four losers and no controls: no plan, so shed the lowest single.
    assert lead("3s 5d 9c Kh", NO_THREAT) == (cards("3s"), "R6")


# ---- Follow rules --------------------------------------------------------------------------------


def test_follow_goes_out_over_the_table():
    # Our pair beats the table's pair and is our whole hand.
    assert follow("9s 9c", NO_THREAT, "5s 5c", 1) == (cards("9s 9c"), "out")


def test_follow_danger_plays_the_lowest_control():
    # Seat 1 led and holds 2 cards; seats 2 and 3 passed. 2♥ is the only sure winner.
    passed = (False, False, True, True)
    assert follow("6s 9c 10d 2h", (2, 6, 6), "5s", 1, passed) == (cards("2h"), "danger")


def test_follow_danger_last_defender_throws_the_highest():
    # Same threat, no sure winner, nobody after us: try the strongest play.
    passed = (False, False, True, True)
    assert follow("6s 9c 10d", (2, 6, 6), "5s", 1, passed) == (cards("10d"), "danger")


def test_follow_danger_falls_through_when_another_seat_can_stop_it():
    # Seat 2 holds 2 cards, but seat 1 acts first and can stop it, so danger passes the decision
    # on, and R1 sheds the lowest free single.
    assert follow("6s 9c 10d", (6, 2, 6), "5s", 3) == (cards("6s"), "R1")


def test_follow_free_keeps_the_pair_whole():
    # 4♠ beats 3♥ but breaks the 44 pair (no fewer plays); 7♦ leaves 44 K, one play fewer.
    assert follow("4s 4c 7d Kh", NO_THREAT, "3h", 1) == (cards("7d"), "R1")


def test_follow_take_lead_with_a_control_that_sets_up_the_finish():
    # 2♥ wins the round; then 55 is the one loser left, so leading it puts us out.
    assert follow("5s 5c 2h", NO_THREAT, "9d", 1) == (cards("2h"), "R8")


def test_follow_keep_passes_rather_than_break_the_aces():
    # An ace beats K♦ but splits AA, isn't a control (2s are unseen) and frees nothing: pass.
    assert follow("3s 3c 5d 7h As Ac", NO_THREAT, "Kd", 1) == (PASS, "R3")


# ---- Agent ---------------------------------------------------------------------------------------


def test_heuristic_agent_replays_with_the_same_seed():
    # The agent samples hidden hands, so the same rng seed must give the same whole hand.
    traces = []
    for _ in range(2):
        state = new_hand(4, random.Random(10))
        trace, _points = play_out(state, [heuristic_agent] * 4, random.Random(11))
        traces.append(trace)
    assert traces[0] == traces[1]
