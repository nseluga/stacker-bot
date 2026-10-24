"""
Tests for pattern recognition (architecture plan §2; stage plan engine_rules, tracker step
"Pattern recognition"). Each rule has a case that must be accepted and one that must be
rejected (classify returns None).
"""

from helpers import classify_text, parse_cards
from stacker.patterns import BOMB, RUN, SET, Pattern


def card(text):
    return parse_cards(text)[0]


# Sets


def test_single_pair_triple_are_sets():
    assert classify_text("9d") == Pattern(SET, 1, 1, card("9d"))
    assert classify_text("9d 9s") == Pattern(SET, 2, 1, card("9d"))
    assert classify_text("9d 9s 9h") == Pattern(SET, 3, 1, card("9h"))


def test_two_different_ranks_are_not_a_pair():
    assert classify_text("3s 4s") is None


def test_empty_play_is_rejected():
    assert classify_text("") is None


# Bombs and quads


def test_quad_is_a_bomb_with_one_deck():
    assert classify_text("Js Jc Jd Jh", num_decks=1).kind == BOMB


def test_quad_is_a_set_with_two_decks():
    assert classify_text("Js Jc Jd Jh", num_decks=2) == Pattern(SET, 4, 1, card("Jh"))


def test_five_and_eight_of_a_kind_are_bombs_with_two_decks():
    assert classify_text("Js Js' Jc Jc' Jd", num_decks=2) == Pattern(BOMB, 5, 1, card("Jd"))
    eight = classify_text("Js Js' Jc Jc' Jd Jd' Jh Jh'", num_decks=2)
    assert eight == Pattern(BOMB, 8, 1, card("Jh"))


# Plain runs


def test_three_consecutive_ranks_are_a_run():
    assert classify_text("3s 4s 5h") == Pattern(RUN, 1, 3, card("5h"))


def test_full_thirteen_card_run_is_legal():
    run = classify_text("3s 4s 5s 6s 7s 8s 9s 10s Js Qs Ks As 2s")
    assert run == Pattern(RUN, 1, 13, card("2s"))


def test_run_may_end_king_ace_two():
    assert classify_text("Ks As 2s") == Pattern(RUN, 1, 3, card("2s"))


def test_wrap_past_two_is_rejected():
    assert classify_text("As 2s 3s") is None
    assert classify_text("2s 3s 4s") is None


def test_run_with_a_gap_is_rejected():
    assert classify_text("3s 4s 6s") is None


def test_two_ranks_other_than_six_seven_are_rejected_even_when_leading():
    assert classify_text("3s 4s", leading=True) is None
    assert classify_text("7s 8s", leading=True) is None


# Multi-runs


def test_pairs_of_consecutive_ranks_are_a_multi_run():
    assert classify_text("3s 3c 4s 4c 5s 5c") == Pattern(RUN, 2, 3, card("5c"))


def test_multi_run_may_end_king_ace_two():
    assert classify_text("Ks Kc As Ac 2s 2c") == Pattern(RUN, 2, 3, card("2c"))


def test_triples_of_consecutive_ranks_are_a_multi_run():
    assert classify_text("3s 3c 3d 4s 4c 4d 5s 5c 5d") == Pattern(RUN, 3, 3, card("5d"))


def test_full_house_is_rejected():
    assert classify_text("3s 3c 3d 4s 4c") is None


def test_uneven_groups_are_rejected():
    assert classify_text("3s 3c 4s 5s 5c") is None
    assert classify_text("3s 3c 4s 4c 4d 5s 5c") is None


def test_quads_in_a_multi_run_make_a_run_not_a_bomb_with_one_deck():
    play = classify_text("3s 3c 3d 3h 4s 4c 4d 4h 5s 5c 5d 5h", num_decks=1)
    assert play == Pattern(RUN, 4, 3, card("5h"))


def test_bomb_sized_groups_in_a_multi_run_with_two_decks():
    play = classify_text("5s 5s' 5c 5c' 5d 6s 6s' 6c 6c' 6d 7s 7s' 7c 7c' 7d", num_decks=2)
    assert play == Pattern(RUN, 5, 3, card("7d"))


# 6-7 on lead


def test_six_seven_on_lead_counts_as_three_rank_run_at_every_group_size():
    assert classify_text("6s 7h", leading=True) == Pattern(RUN, 1, 3, card("7h"))
    assert classify_text("6s 6c 7s 7h", leading=True) == Pattern(RUN, 2, 3, card("7h"))
    assert classify_text("6s 6c 6d 7s 7c 7h", leading=True) == Pattern(RUN, 3, 3, card("7h"))
    four_each = classify_text("6s 6c 6d 6h 7s 7c 7d 7h", leading=True)
    assert four_each == Pattern(RUN, 4, 3, card("7h"))


def test_six_seven_is_rejected_when_not_leading():
    assert classify_text("6s 7h", leading=False) is None
    assert classify_text("6s 6c 7s 7h", leading=False) is None


def test_uneven_six_seven_is_rejected():
    assert classify_text("6s 6c 7h", leading=True) is None


# Top card


def test_top_card_uses_suit():
    assert classify_text("3s 3h").top_card == card("3h")


def test_top_card_is_the_good_copy_with_two_decks():
    assert classify_text("8h' 8h", num_decks=2).top_card == card("8h")
