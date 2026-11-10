"""
Tests for the M3 encoding (stage plan specialist_explore, stage 1 `encoding`).

Card grid: each card lands in its (deck channel, suit, rank) cell, good copy in channel 0.
Relative slot: seats count from the viewer, 0 = me.
Action vector: card grid, pass flag, then kind / group size / span one-hots.
History: last H moves, oldest first, padded at the front, mask True for real moves.
Card sets: played per seat, and unseen = deck minus own hand minus everything played.
State: every named block holds the right bits; slots past the table size stay 0.
Decision: the bundle equals the single encoders, one action row per legal action, in order.
"""

import random

from helpers import parse_cards
from stacker.cards import CONFIGS, build_deck
from stacker.encode import (
    ACTION_CARDS,
    ACTION_DIM,
    ACTION_GROUP,
    ACTION_KIND,
    ACTION_PASS,
    ACTION_SPAN,
    CARD_GRID_SHAPE,
    CARD_GRID_SIZE,
    CARDS_LEFT_BITS,
    HISTORY_WINDOW,
    LAST_MOVE_SIZE,
    MAX_PLAYERS,
    MOVE_CARDS,
    MOVE_DIM,
    MOVE_PASS,
    MOVE_SEAT,
    NUM_OPPONENT_SLOTS,
    PATTERN_KINDS,
    STATE_BOMBS,
    STATE_CARDS_LEFT,
    STATE_DECKS,
    STATE_DIM,
    STATE_FINISHED,
    STATE_HAND,
    STATE_LAST_MOVE,
    STATE_LEADING,
    STATE_PASSED,
    STATE_PLAYED,
    STATE_PLAYERS,
    STATE_TABLE,
    STATE_TABLE_SEAT,
    STATE_UNSEEN,
    card_grid,
    encode_action,
    encode_decision,
    encode_history,
    encode_state,
    played_by_seat,
    relative_slot,
    unseen_cards,
)
from stacker.game import PASS, Observation, legal_actions, new_hand, play_out
from stacker.moves import Play
from stacker.patterns import BOMB, RUN, SET, classify

# Dev seed for dealing real hands (rule 2: seeds >= 10 outside confirm runs).
DEV_SEED = 10


def make_play(text, num_decks=1, leading=True):
    """Builds a Play from typed cards, classified the way the engine would."""
    cards = tuple(sorted(parse_cards(text)))
    return Play(cards, classify(list(cards), num_decks, leading))


def observation_with_history(me, num_players, history, hand=()):
    """An Observation holding only what the encoders read: viewer, seat count, hand, history."""
    return Observation(
        seat=me,
        hand=tuple(hand),
        num_decks=1,
        table=None,
        table_seat=None,
        passed=(False,) * num_players,
        cards_left=(0,) * num_players,
        finish_order=(),
        opening_card=None,
        history=tuple(history),
    )


def pattern_bits(vector):
    """Returns (kind, group size, span) read back from an action vector's one-hot blocks."""
    kind = PATTERN_KINDS[int(vector[ACTION_KIND].argmax())]
    group_size = int(vector[ACTION_GROUP].argmax()) + 1
    span = int(vector[ACTION_SPAN].argmax()) + 1
    return kind, group_size, span


def test_three_of_spades_is_first_cell():
    grid = card_grid(parse_cards("3s"))
    assert grid.shape == CARD_GRID_SHAPE
    assert grid[0, 0, 0] == 1.0
    assert grid.sum() == 1.0


def test_good_two_of_hearts_is_in_good_channel():
    grid = card_grid(parse_cards("2h"))
    assert grid[0, 3, 12] == 1.0
    assert grid.sum() == 1.0


def test_bad_two_of_hearts_is_in_bad_channel():
    grid = card_grid(parse_cards("2h'"))
    assert grid[1, 3, 12] == 1.0
    assert grid.sum() == 1.0


def test_grid_sum_equals_card_count():
    cards = parse_cards("3s 3s' 10h jd 2h 2h'")
    assert card_grid(cards).sum() == len(cards)


def test_one_deck_fills_good_channel_only():
    grid = card_grid(build_deck(1))
    assert grid[0].sum() == 52
    assert grid[1].sum() == 0


def test_two_decks_fill_every_cell_once():
    grid = card_grid(build_deck(2))
    assert grid.sum() == 104
    assert grid.min() == 1.0


def test_empty_set_is_all_zeros():
    assert card_grid([]).sum() == 0


def test_relative_slots_in_four_players():
    # Viewer is seat 2: seat 3 acts next, then 0, then 1.
    me = 2
    assert relative_slot(2, me, 4) == 0
    assert relative_slot(3, me, 4) == 1
    assert relative_slot(0, me, 4) == 2
    assert relative_slot(1, me, 4) == 3


def test_pass_sets_only_the_pass_bit():
    vector = encode_action(PASS)
    assert vector.shape == (ACTION_DIM,)
    assert vector[ACTION_PASS] == 1.0
    assert vector.sum() == 1.0


def test_single_three_of_spades():
    vector = encode_action(make_play("3s"))
    assert vector[ACTION_CARDS].sum() == 1
    assert vector[ACTION_PASS] == 0.0
    assert pattern_bits(vector) == (SET, 1, 1)


def test_pair_run():
    vector = encode_action(make_play("3s 3c 4s 4c 5s 5c"))
    assert vector[ACTION_CARDS].sum() == 6
    assert pattern_bits(vector) == (RUN, 2, 3)


def test_two_deck_five_of_a_kind_is_bomb():
    vector = encode_action(make_play("9s 9s' 9c 9d 9h", num_decks=2))
    assert vector[ACTION_CARDS].sum() == 5
    assert pattern_bits(vector) == (BOMB, 5, 1)


def test_six_seven_lead_spans_three():
    vector = encode_action(make_play("6s 7s"))
    assert pattern_bits(vector) == (RUN, 1, 3)


def test_each_play_sets_one_bit_per_pattern_block():
    vector = encode_action(make_play("10h jh qh kh"))
    assert vector[ACTION_KIND].sum() == 1
    assert vector[ACTION_GROUP].sum() == 1
    assert vector[ACTION_SPAN].sum() == 1


def test_action_shape_is_the_same_in_every_config():
    # The first decision of a real deal in each config: every legal play encodes to ACTION_DIM.
    rng = random.Random(DEV_SEED)
    for num_players in CONFIGS:
        state = new_hand(num_players, rng)
        for action in legal_actions(state):
            vector = encode_action(action)
            assert vector.shape == (ACTION_DIM,)
            assert vector[ACTION_CARDS].sum() == len(action.cards)


def test_empty_history_is_all_padding():
    moves, mask = encode_history(observation_with_history(0, 4, []))
    assert moves.shape == (HISTORY_WINDOW, MOVE_DIM)
    assert moves.sum() == 0
    assert not mask.any()


def test_latest_move_is_last_row():
    # 4 players, viewer seat 2. Seat 3 plays 3s, seat 0 passes, seat 1 plays 5h.
    history = [(3, make_play("3s")), (0, PASS), (1, make_play("5h"))]
    moves, mask = encode_history(observation_with_history(2, 4, history))

    # Three real rows at the end, padding before them.
    assert mask.tolist() == [False] * (HISTORY_WINDOW - 3) + [True] * 3
    assert moves[: HISTORY_WINDOW - 3].sum() == 0

    # Last row: seat 1 is 3 seats after seat 2, playing the good 5h.
    last = moves[-1]
    assert last[MOVE_SEAT].tolist() == [0, 0, 0, 1, 0, 0, 0, 0]
    assert last[MOVE_CARDS].reshape(CARD_GRID_SHAPE)[0, 3, 2] == 1.0
    assert last[MOVE_CARDS].sum() == 1
    assert last[MOVE_PASS] == 0.0

    # First real row: seat 3 is the next seat after me.
    assert moves[-3][MOVE_SEAT].argmax() == 1


def test_pass_row_has_only_seat_and_pass_bits():
    moves, mask = encode_history(observation_with_history(2, 4, [(0, PASS)]))
    row = moves[-1]
    assert row[MOVE_PASS] == 1.0
    assert row[MOVE_SEAT].argmax() == 2
    assert row.sum() == 2


def test_long_history_keeps_last_window_moves():
    # 20 moves, each a pass by seat (move number % 4), so each row's seat tells which move it is.
    history = []
    for move_number in range(20):
        history.append((move_number % 4, PASS))
    moves, mask = encode_history(observation_with_history(0, 4, history))
    assert mask.all()
    # Row 0 is move 5 (the 6th move), by seat 1.
    assert moves[0][MOVE_SEAT].argmax() == 5 % 4
    assert moves[-1][MOVE_SEAT].argmax() == 19 % 4


def test_history_shape_is_the_same_in_every_config():
    # Play real dealt hands with a first-legal-action agent, encoding every observation on the way.
    seen_moves = []

    def encoding_agent(observation, actions, rng):
        moves, mask = encode_history(observation)
        assert moves.shape == (HISTORY_WINDOW, MOVE_DIM)
        assert mask.sum() == min(len(observation.history), HISTORY_WINDOW)
        seen_moves.append(len(observation.history))
        return actions[0]

    rng = random.Random(DEV_SEED)
    for num_players in CONFIGS:
        state = new_hand(num_players, rng)
        agents = [encoding_agent] * num_players
        play_out(state, agents, rng)

    # The hands ran past the window, so full windows were checked too.
    assert max(seen_moves) > HISTORY_WINDOW


def test_played_by_seat_reads_history():
    history = [(3, make_play("3s 3c")), (0, PASS), (1, make_play("5h")), (3, make_play("9d"))]
    played = played_by_seat(observation_with_history(2, 4, history))
    assert played[0] == []
    assert played[1] == parse_cards("5h")
    assert played[2] == []
    assert played[3] == parse_cards("3s 3c 9d")


def test_unseen_with_no_history_is_deck_minus_hand():
    hand = parse_cards("3s 4c 2h")
    observation = observation_with_history(0, 4, [], hand=hand)
    assert played_by_seat(observation) == [[], [], [], []]
    unseen = unseen_cards(observation)
    assert len(unseen) == 52 - 3
    for card in hand:
        assert card not in unseen


def test_hand_unseen_and_played_cover_the_deck_exactly_once():
    # At every decision of real dev hands, in every config, the three sets partition the deck.
    checked = []

    def partition_agent(observation, actions, rng):
        all_cards = list(observation.hand) + unseen_cards(observation)
        for seat_cards in played_by_seat(observation):
            all_cards.extend(seat_cards)
        assert sorted(all_cards) == build_deck(observation.num_decks)
        checked.append(observation.num_decks)
        return actions[0]

    rng = random.Random(DEV_SEED)
    for num_players in CONFIGS:
        state = new_hand(num_players, rng)
        agents = [partition_agent] * num_players
        play_out(state, agents, rng)

    # Both deck counts were exercised.
    assert 1 in checked
    assert 2 in checked


def grid_of(text):
    """Flattened card grid of typed cards, to compare against a state block."""
    return card_grid(parse_cards(text)).reshape(-1)


def test_hand_built_four_player_state():
    # 4 players, I am seat 1. Seat 0 plays 3s, I play 5d, seat 2 passes, seat 3 plays 9h,
    # seat 0 passes. My turn, 9h on the table.
    # Slots from seat 1: seat 2 → 1, seat 3 → 2, seat 0 → 3.
    nine_of_hearts = make_play("9h", leading=False)
    history = [
        (0, make_play("3s")),
        (1, make_play("5d", leading=False)),
        (2, PASS),
        (3, nine_of_hearts),
        (0, PASS),
    ]
    hand = parse_cards("4s 4c jd 2h")
    observation = Observation(
        seat=1,
        hand=tuple(hand),
        num_decks=1,
        table=nine_of_hearts,
        table_seat=3,
        passed=(True, False, True, False),
        cards_left=(12, 4, 13, 12),
        finish_order=(),
        opening_card=None,
        history=tuple(history),
    )
    state = encode_state(observation)
    assert state.shape == (STATE_DIM,)

    # My hand, and unseen = 52 − my 4 − the 3 played.
    assert (state[STATE_HAND] == grid_of("4s 4c jd 2h")).all()
    assert state[STATE_UNSEEN].sum() == 52 - 4 - 3

    # Played per opponent slot (row k = slot k + 1): slot 1 none, slot 2 9h, slot 3 3s.
    played = state[STATE_PLAYED].reshape(NUM_OPPONENT_SLOTS, CARD_GRID_SIZE)
    assert played[0].sum() == 0
    assert (played[1] == grid_of("9h")).all()
    assert (played[2] == grid_of("3s")).all()

    # Table: the 9h with its pattern, played by slot 2; I am not leading.
    assert (state[STATE_TABLE] == encode_action(nine_of_hearts)).all()
    assert state[STATE_TABLE_SEAT].tolist() == [0, 0, 1, 0, 0, 0, 0, 0]
    assert state[STATE_LEADING] == 0.0

    # Last move per opponent slot: slot 1 passed, slot 2 played 9h, slot 3's 3s was replaced by a pass.
    last = state[STATE_LAST_MOVE].reshape(NUM_OPPONENT_SLOTS, LAST_MOVE_SIZE)
    assert last[0][CARD_GRID_SIZE] == 1.0
    assert last[0].sum() == 1
    assert (last[1][:CARD_GRID_SIZE] == grid_of("9h")).all()
    assert last[1][CARD_GRID_SIZE] == 0.0
    assert last[2][CARD_GRID_SIZE] == 1.0
    assert last[2].sum() == 1

    # Cards left by slot: me 4, slot 1 (seat 2) 13, slot 2 (seat 3) 12, slot 3 (seat 0) 12.
    cards_left = state[STATE_CARDS_LEFT].reshape(MAX_PLAYERS, CARDS_LEFT_BITS)
    assert cards_left[0].argmax() == 4
    assert cards_left[1].argmax() == 13
    assert cards_left[2].argmax() == 12
    assert cards_left[3].argmax() == 12

    # Passed this round: seats 0 and 2, i.e. slots 3 and 1. No one is out.
    assert state[STATE_PASSED].tolist() == [0, 1, 0, 1, 0, 0, 0, 0]
    assert state[STATE_FINISHED].sum() == 0

    # 4 players is bit 1 of 3–8; 1 deck is bit 0; no bombs is bit 0.
    assert state[STATE_PLAYERS].tolist() == [0, 1, 0, 0, 0, 0]
    assert state[STATE_DECKS].tolist() == [1, 0]
    assert state[STATE_BOMBS].argmax() == 0


def test_leading_sets_flag_and_leaves_table_empty():
    observation = observation_with_history(0, 4, [], hand=parse_cards("3s"))
    state = encode_state(observation)
    assert state[STATE_LEADING] == 1.0
    assert state[STATE_TABLE].sum() == 0
    assert state[STATE_TABLE_SEAT].sum() == 0


def test_bombs_are_counted_and_capped():
    # One quad bomb in a 1-deck game sets bit 1.
    history = [(1, make_play("7s 7c 7d 7h"))]
    state = encode_state(observation_with_history(0, 4, history))
    assert state[STATE_BOMBS].argmax() == 1
    assert state[STATE_BOMBS].sum() == 1

    # Nine bombs (only plays are read, not who could hold them) land in the capped last bit.
    history = []
    for rank_name in ("3", "4", "5", "6", "7", "8", "9", "10", "j"):
        quad = rank_name + "s " + rank_name + "c " + rank_name + "d " + rank_name + "h"
        history.append((1, make_play(quad)))
    state = encode_state(observation_with_history(0, 4, history))
    assert state[STATE_BOMBS][-1] == 1.0
    assert state[STATE_BOMBS].sum() == 1


def test_state_shape_and_empty_slots_in_every_config():
    # At every decision of real dev hands: fixed shape, slots past the table size all 0, and one
    # cards-left bit per real seat.
    checked_configs = []

    def state_agent(observation, actions, rng):
        state = encode_state(observation)
        assert state.shape == (STATE_DIM,)
        num_players = len(observation.cards_left)

        played = state[STATE_PLAYED].reshape(NUM_OPPONENT_SLOTS, CARD_GRID_SIZE)
        last = state[STATE_LAST_MOVE].reshape(NUM_OPPONENT_SLOTS, LAST_MOVE_SIZE)
        cards_left = state[STATE_CARDS_LEFT].reshape(MAX_PLAYERS, CARDS_LEFT_BITS)
        for slot in range(MAX_PLAYERS):
            if slot >= num_players:
                assert cards_left[slot].sum() == 0
                assert state[STATE_PASSED.start + slot] == 0
                assert state[STATE_FINISHED.start + slot] == 0
                assert state[STATE_TABLE_SEAT.start + slot] == 0
                assert played[slot - 1].sum() == 0
                assert last[slot - 1].sum() == 0
            else:
                assert cards_left[slot].sum() == 1
        assert state[STATE_PLAYERS].sum() == 1
        assert state[STATE_DECKS].sum() == 1
        checked_configs.append(num_players)
        return actions[0]

    rng = random.Random(DEV_SEED)
    for num_players in CONFIGS:
        state = new_hand(num_players, rng)
        agents = [state_agent] * num_players
        play_out(state, agents, rng)

    assert sorted(set(checked_configs)) == sorted(CONFIGS)


# The stage 5 long-window arm: every hand seen in M2 fits in 190 moves.
LONG_WINDOW = 190


def test_decision_matches_single_encoders():
    history = [(3, make_play("3s")), (0, make_play("5h", leading=False))]
    observation = observation_with_history(2, 4, history, hand=parse_cards("4s 9d"))
    actions = [make_play("9d", leading=False), PASS]

    state, moves, mask, action_matrix = encode_decision(observation, actions)
    expected_moves, expected_mask = encode_history(observation)
    assert (state == encode_state(observation)).all()
    assert (moves == expected_moves).all()
    assert (mask == expected_mask).all()
    assert action_matrix.shape == (2, ACTION_DIM)
    assert (action_matrix[0] == encode_action(actions[0])).all()
    assert (action_matrix[1] == encode_action(PASS)).all()


def test_decision_shapes_in_every_config_and_long_window():
    # At every decision of real dev hands: K rows for K legal actions, fixed other shapes.
    checked_configs = []

    def decision_agent(observation, actions, rng):
        for window in (HISTORY_WINDOW, LONG_WINDOW):
            state, moves, mask, action_matrix = encode_decision(observation, actions, window)
            assert state.shape == (STATE_DIM,)
            assert moves.shape == (window, MOVE_DIM)
            assert mask.shape == (window,)
            assert action_matrix.shape == (len(actions), ACTION_DIM)
        checked_configs.append(len(observation.cards_left))
        return actions[0]

    rng = random.Random(DEV_SEED)
    for num_players in CONFIGS:
        state = new_hand(num_players, rng)
        agents = [decision_agent] * num_players
        play_out(state, agents, rng)

    assert sorted(set(checked_configs)) == sorted(CONFIGS)
