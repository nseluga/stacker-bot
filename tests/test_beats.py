"""
Tests for beat comparison (architecture plan §2; stage plan engine_rules, tracker step
"Beat comparison"). A bomb beats any non-bomb; otherwise the new play must have the same
shape as the previous one and a higher top card.
"""

from helpers import classify_text as play
from stacker.patterns import beats

# Bombs, one deck


def test_quad_beats_any_non_bomb_with_one_deck():
    quad = play("3s 3c 3d 3h")
    assert beats(quad, play("2h"))
    assert beats(quad, play("2d 2h"))
    assert beats(quad, play("Qs Kh Ah"))
    assert beats(quad, play("4s 4c 4d 4h 5s 5c 5d 5h 6s 6c 6d 6h"))


def test_ace_quad_loses_only_to_two_quad():
    aces = play("As Ac Ad Ah")
    twos = play("2s 2c 2d 2h")
    assert not beats(aces, twos)
    assert beats(twos, aces)
    assert beats(aces, play("Ks Kc Kd Kh"))


def test_no_non_bomb_beats_a_quad_with_one_deck():
    quad = play("3s 3c 3d 3h")
    assert not beats(play("2h"), quad)
    assert not beats(play("3s 4s 5s 6s 7s 8s 9s 10s Js Qs Ks As 2h"), quad)


# Quads and bombs, two decks


def test_quad_is_a_normal_pattern_with_two_decks():
    quad = play("3s 3c 3d 3h", num_decks=2)
    assert not beats(quad, play("2h", num_decks=2))
    assert beats(play("4s 4c 4d 4h", num_decks=2), quad)


def test_same_rank_quads_go_by_top_card_with_two_decks():
    lower = play("9s 9c 9d 9h'", num_decks=2)
    higher = play("9s' 9c' 9d' 9h", num_decks=2)
    assert beats(higher, lower)
    assert not beats(lower, higher)


def test_five_of_a_kind_beats_any_non_bomb_with_two_decks():
    bomb = play("3s 3s' 3c 3c' 3d", num_decks=2)
    assert beats(bomb, play("2s 2c 2d 2h", num_decks=2))
    assert beats(bomb, play("2h", num_decks=2))


def test_bigger_bomb_beats_smaller_bomb_with_two_decks():
    six_threes = play("3s 3s' 3c 3c' 3d 3d'", num_decks=2)
    five_twos = play("2s 2s' 2c 2c' 2d", num_decks=2)
    assert beats(six_threes, five_twos)
    assert not beats(five_twos, six_threes)


def test_same_size_bombs_go_by_rank_with_two_decks():
    five_twos = play("2s 2s' 2c 2c' 2d", num_decks=2)
    five_aces = play("As As' Ac Ac' Ad", num_decks=2)
    assert beats(five_twos, five_aces)
    assert not beats(five_aces, five_twos)


# Same shape required


def test_set_must_match_size():
    assert not beats(play("2s 2h"), play("3s"))
    assert not beats(play("2s 2h"), play("3s 3c 3d"))
    assert beats(play("4s 4c"), play("3d 3h"))


def test_run_must_match_length():
    assert not beats(play("4s 5s 6s 7s"), play("3s 4s 5s"))
    assert not beats(play("Js Qs Ks"), play("3s 4s 5s 6s"))


def test_multi_run_must_match_group_size():
    assert beats(play("4s 4c 5s 5c 6s 6c"), play("3s 3c 4s 4c 5s 5h"))
    assert not beats(play("4s 5s 6s"), play("3s 3c 4s 4c 5s 5h"))


def test_set_and_run_of_same_card_count_do_not_beat_each_other():
    assert not beats(play("2s 2c 2d"), play("3s 4s 5s"))
    assert not beats(play("Js Qs Ks"), play("3s 3c 3d"))


# Top card and ties


def test_run_decided_by_highest_card():
    # The Doc's example: 4♠ 5♠ 6♥ beats 4♥ 5♥ 6♠.
    assert beats(play("4s 5s 6h"), play("4h 5h 6s"))
    assert not beats(play("4h 5h 6s"), play("4s 5s 6h"))


def test_good_copy_beats_bad_copy():
    assert beats(play("9h", num_decks=2), play("9h'", num_decks=2))
    assert not beats(play("9h'", num_decks=2), play("9h", num_decks=2))


def test_play_does_not_beat_itself():
    same = play("7s 8s 9s")
    assert not beats(same, same)


# 6-7 on lead


def test_three_rank_run_topped_above_the_seven_beats_six_seven():
    six_seven = play("6s 7c", leading=True)
    assert beats(play("5s 6s 7h"), six_seven)
    assert beats(play("8s 9s 10s"), six_seven)
    assert not beats(play("5h 6h 7s"), six_seven)
    assert not beats(play("4h 5h 6h"), six_seven)


def test_pair_six_seven_needs_a_three_rank_pair_run():
    pair_six_seven = play("6s 6c 7s 7d", leading=True)
    assert beats(play("5s 5c 6s 6c 7s 7h"), pair_six_seven)
    assert beats(play("6d 6h 7c 7h 8s 8c"), pair_six_seven)
    assert not beats(play("5s 5c 6s 6c 7s 7c"), pair_six_seven)
    assert not beats(play("8s 9s 10s"), pair_six_seven)


def test_longer_run_does_not_beat_six_seven():
    assert not beats(play("4s 5s 6s 7h"), play("6s 7c", leading=True))
