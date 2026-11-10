"""
Turns what a seat sees into fixed-shape numpy arrays for the M3 Q-net (architecture plan §6).

Every config (3–8 players, 1 or 2 decks) encodes to the same shapes, so one net can read any
table. The pieces, in build order:

- The card grid: any set of cards becomes a [2, 4, 13] array of 0s and 1s, one cell per
  (deck copy, suit, rank). Channel 0 is the good copy, channel 1 the bad copy, so 1-deck games
  fill channel 0 only and a 1-deck hand looks the same as the good half of a 2-deck hand.
- The relative slot: seats are numbered from the viewer, 0 = me, 1 = the next seat to act after
  me, and so on, so the net learns "the player after me" rather than "seat 2".
- The action vector: one candidate move as its card grid, a pass flag, and its pattern spelled
  out as three one-hots (kind, cards per rank, ranks spanned), so the net is told the play's
  shape instead of having to work it out from the cards.
- The history: the last H moves of the hand as an [H, MOVE_DIM] table, one row per move (who
  moved, as a relative slot; which cards; or a pass), oldest first. Short hands are padded with
  zero rows at the front, and a mask marks the real rows, so the last row is always the latest
  move. Any sequence encoder (the LSTM now, a transformer arm later) reads the same table.
- The card sets the state is built from: what each seat has played this hand, and the unseen
  cards (the deck minus my hand minus everything played: other hands plus the aside).
- The state vector: everything the seat can see packed into one [STATE_DIM] vector, every block
  named. Per-seat blocks are in relative slot order and padded to 8 slots (7 opponents), so a
  4-player game leaves slots 4–7 at 0 and the shape never changes.
- The decision: encode_decision bundles state, history, mask and one row per legal action. It is
  the one call actors and the learner use, so the encoding has a single entry point.

Numpy only: torch is added at stage 2 (stage plan specialist_explore), and actors stay torch-free.
"""

import numpy as np

from stacker.cards import BAD_DECK, CONFIGS, GOOD_DECK, RANK_NAMES, SUIT_SYMBOLS, build_deck
from stacker.game import PASS
from stacker.patterns import BOMB, RUN, SET

# Grid axis sizes: deck copy (good, bad) × suit × rank.
NUM_DECK_CHANNELS = 2
NUM_SUITS = len(SUIT_SYMBOLS)
NUM_RANKS = len(RANK_NAMES)
CARD_GRID_SHAPE = (NUM_DECK_CHANNELS, NUM_SUITS, NUM_RANKS)

# Card.deck value → grid channel. Card.deck stores GOOD_DECK = 1 and BAD_DECK = 0, the reverse of
# the channel order, so the channel is looked up here rather than indexed by the deck value.
GOOD_CHANNEL = 0
BAD_CHANNEL = 1
DECK_CHANNEL = {GOOD_DECK: GOOD_CHANNEL, BAD_DECK: BAD_CHANNEL}


def card_grid(cards):
    """
    Marks each card in its cell of the card grid.
    cards: iterable of Cards (a hand, a play, cards seen played); may be empty.
    Returns float32 [2, 4, 13]: axis 0 deck channel (0 good, 1 bad), axis 1 suit (♠ ♣ ♦ ♥),
    axis 2 rank (3 … 2). A cell is 1 when that card is in the set, else 0.
    Each physical card exists once per game, so no cell is ever counted twice.
    """
    grid = np.zeros(CARD_GRID_SHAPE, dtype=np.float32)
    for card in cards:
        grid[DECK_CHANNEL[card.deck], card.suit, card.rank] = 1.0
    return grid


def relative_slot(seat, me, num_players):
    """
    Returns where seat sits as seen from me: 0 = me, 1 = the next seat after me, up to
    num_players − 1 = the seat just before me. The modulo wraps seats that come before me
    around the table.
    """
    return (seat - me) % num_players


# Pattern kinds in one-hot order.
PATTERN_KINDS = (SET, RUN, BOMB)

# Largest group of one rank: 8 of a kind, possible only with 2 decks. Group size g sets bit g − 1.
MAX_GROUP_SIZE = 8

# Most ranks a play can span: a run from 3 to 2. Span s sets bit s − 1.
MAX_SPAN = NUM_RANKS

# Action vector layout, in order:
#   cards  104  card_grid of the play, flattened (deck channel, suit, rank)
#   pass     1  1 for PASS
#   kind     3  one-hot over PATTERN_KINDS
#   group    8  one-hot, cards per rank 1–8
#   span    13  one-hot, ranks spanned 1–13 (a 6-7 lead counts as 3, as beats compares it)
CARD_GRID_SIZE = NUM_DECK_CHANNELS * NUM_SUITS * NUM_RANKS
ACTION_CARDS = slice(0, CARD_GRID_SIZE)
ACTION_PASS = CARD_GRID_SIZE
ACTION_KIND = slice(ACTION_PASS + 1, ACTION_PASS + 1 + len(PATTERN_KINDS))
ACTION_GROUP = slice(ACTION_KIND.stop, ACTION_KIND.stop + MAX_GROUP_SIZE)
ACTION_SPAN = slice(ACTION_GROUP.stop, ACTION_GROUP.stop + MAX_SPAN)
ACTION_DIM = ACTION_SPAN.stop


def encode_action(action):
    """
    Encodes one candidate action for the Q-net to score.
    action: a Play from legal_actions, or PASS.
    Returns float32 [ACTION_DIM] laid out as the block list above. A PASS has only its pass
    bit set: no cards, no pattern.
    """
    vector = np.zeros(ACTION_DIM, dtype=np.float32)
    if action == PASS:
        vector[ACTION_PASS] = 1.0
        return vector

    # The cards go in as the flattened grid, so the same card always sets the same position.
    vector[ACTION_CARDS] = card_grid(action.cards).reshape(-1)

    # Each one-hot block is set by offsetting from its block's start.
    pattern = action.pattern
    vector[ACTION_KIND.start + PATTERN_KINDS.index(pattern.kind)] = 1.0
    vector[ACTION_GROUP.start + pattern.group_size - 1] = 1.0
    vector[ACTION_SPAN.start + pattern.num_ranks - 1] = 1.0
    return vector


# Most seats at any table. Seat one-hots are this wide in every config, so shapes never change.
MAX_PLAYERS = max(CONFIGS)

# Default history window: the M3 control reads the last 15 moves (decision log 2026-11-08;
# the 190-move arm passes a different window).
HISTORY_WINDOW = 15

# History row layout, in order:
#   seat    8  one-hot of the mover's relative slot (0 = me)
#   cards 104  card_grid of the play, flattened; all 0 for a pass
#   pass    1  1 for a pass
MOVE_SEAT = slice(0, MAX_PLAYERS)
MOVE_CARDS = slice(MOVE_SEAT.stop, MOVE_SEAT.stop + CARD_GRID_SIZE)
MOVE_PASS = MOVE_CARDS.stop
MOVE_DIM = MOVE_PASS + 1


def encode_history(observation, window=HISTORY_WINDOW):
    """
    Encodes the last `window` moves of the hand, as seen by the seat to act.
    observation: an Observation; its history is every (seat, action) so far, in play order.
    window: int, how many recent moves to keep.
    Returns (moves, mask):
      moves float32 [window, MOVE_DIM]: one row per move, oldest first, laid out as above.
      mask bool [window]: True for a real move, False for a padding row.
    Passes count as moves, so each one takes a row. When fewer than `window` moves exist, the
    real rows sit at the end and the padding rows at the front are all 0.
    """
    me = observation.seat
    num_players = len(observation.cards_left)
    moves = np.zeros((window, MOVE_DIM), dtype=np.float32)
    mask = np.zeros(window, dtype=bool)

    # Keep only the most recent moves, then place them so the latest lands in the last row.
    recent = observation.history[-window:]
    first_row = window - len(recent)
    for offset, (seat, action) in enumerate(recent):
        row = first_row + offset
        mask[row] = True
        moves[row, MOVE_SEAT.start + relative_slot(seat, me, num_players)] = 1.0
        if action == PASS:
            moves[row, MOVE_PASS] = 1.0
        else:
            moves[row, MOVE_CARDS] = card_grid(action.cards).reshape(-1)
    return moves, mask


def played_by_seat(observation):
    """
    Lists the cards each seat has played this hand, read from the public history.
    observation: an Observation.
    Returns a list with one list of Cards per seat, indexed by absolute seat (not relative slot;
    encode_state turns seats into slots). Passes add nothing.
    """
    num_players = len(observation.cards_left)
    played = []
    for seat in range(num_players):
        played.append([])

    for seat, action in observation.history:
        if action != PASS:
            played[seat].extend(action.cards)
    return played


def unseen_cards(observation):
    """
    Returns the cards the seat to act cannot see: the game's whole deck minus its own hand minus
    every card played so far. That is every other hand plus the aside.
    observation: an Observation.
    Returns a list of Cards, lowest first. Each physical card is a distinct (rank, suit, deck)
    tuple, so removing by set never confuses the good and bad copy.
    """
    # Everything the seat has seen: its own hand and every card on the history.
    seen = set(observation.hand)
    for seat_cards in played_by_seat(observation):
        seen.update(seat_cards)

    # build_deck lists cards lowest first, so keeping its order keeps the result sorted.
    unseen = []
    for card in build_deck(observation.num_decks):
        if card not in seen:
            unseen.append(card)
    return unseen


# Opponent slots: every seat but me (slot 0) at the largest table.
NUM_OPPONENT_SLOTS = MAX_PLAYERS - 1

# Smallest table. The player-count one-hot sets bit (num_players − MIN_PLAYERS).
MIN_PLAYERS = min(CONFIGS)
NUM_PLAYER_COUNTS = MAX_PLAYERS - MIN_PLAYERS + 1

# Deck counts are 1 or 2; the deck-count one-hot sets bit (num_decks − 1).
NUM_DECK_COUNTS = 2

# Largest hand dealt in any config (3p: 17). Cards left run 0 to this, one bit each.
MAX_HAND_SIZE = 0
for _config in CONFIGS.values():
    if _config.hand_size > MAX_HAND_SIZE:
        MAX_HAND_SIZE = _config.hand_size
CARDS_LEFT_BITS = MAX_HAND_SIZE + 1

# Bombs played: one-hot over 0 to BOMB_CAP, the last bit meaning BOMB_CAP or more.
# ponytail: cap is a guess (M2 did not count bombs); raise it if stage 4 logs show more.
BOMB_CAP = 6
BOMB_BITS = BOMB_CAP + 1

# Width of one opponent's last-move block: card grid + pass flag.
LAST_MOVE_SIZE = CARD_GRID_SIZE + 1

# State vector layout, in order (slot 0 = me, slot k = k seats after me):
#   hand         104  my cards
#   unseen       104  unseen_cards: other hands plus the aside
#   played     7×104  cards played by opponent slots 1–7
#   table        129  the play to beat, as encode_action; 0 when I lead
#   table_seat     8  slot one-hot of who played it
#   last_move  7×105  each opponent's latest action this hand: grid + pass flag; 0 if none yet
#   cards_left  8×18  per slot, one-hot of cards left 0–17
#   players        6  one-hot, 3–8 players
#   decks          2  one-hot, 1–2 decks
#   bombs          7  one-hot, bombs played 0–5, last bit 6 or more
#   leading        1  1 when I lead (no play on the table)
#   passed         8  per slot, 1 if that seat has passed this round
#   finished       8  per slot, 1 if that seat is out
STATE_HAND = slice(0, CARD_GRID_SIZE)
STATE_UNSEEN = slice(STATE_HAND.stop, STATE_HAND.stop + CARD_GRID_SIZE)
STATE_PLAYED = slice(STATE_UNSEEN.stop, STATE_UNSEEN.stop + NUM_OPPONENT_SLOTS * CARD_GRID_SIZE)
STATE_TABLE = slice(STATE_PLAYED.stop, STATE_PLAYED.stop + ACTION_DIM)
STATE_TABLE_SEAT = slice(STATE_TABLE.stop, STATE_TABLE.stop + MAX_PLAYERS)
STATE_LAST_MOVE = slice(
    STATE_TABLE_SEAT.stop, STATE_TABLE_SEAT.stop + NUM_OPPONENT_SLOTS * LAST_MOVE_SIZE
)
STATE_CARDS_LEFT = slice(STATE_LAST_MOVE.stop, STATE_LAST_MOVE.stop + MAX_PLAYERS * CARDS_LEFT_BITS)
STATE_PLAYERS = slice(STATE_CARDS_LEFT.stop, STATE_CARDS_LEFT.stop + NUM_PLAYER_COUNTS)
STATE_DECKS = slice(STATE_PLAYERS.stop, STATE_PLAYERS.stop + NUM_DECK_COUNTS)
STATE_BOMBS = slice(STATE_DECKS.stop, STATE_DECKS.stop + BOMB_BITS)
STATE_LEADING = STATE_BOMBS.stop
STATE_PASSED = slice(STATE_LEADING + 1, STATE_LEADING + 1 + MAX_PLAYERS)
STATE_FINISHED = slice(STATE_PASSED.stop, STATE_PASSED.stop + MAX_PLAYERS)
STATE_DIM = STATE_FINISHED.stop


def encode_state(observation):
    """
    Packs everything the seat to act can see into one vector.
    observation: an Observation.
    Returns float32 [STATE_DIM] laid out as the block list above. Per-slot blocks are written
    through reshaped views ([slots, width]), so row k of a view is slot k (or opponent slot k + 1).
    """
    me = observation.seat
    num_players = len(observation.cards_left)
    state = np.zeros(STATE_DIM, dtype=np.float32)

    # The two card sets from my point of view: what I hold and what I cannot see.
    state[STATE_HAND] = card_grid(observation.hand).reshape(-1)
    state[STATE_UNSEEN] = card_grid(unseen_cards(observation)).reshape(-1)

    # Cards each opponent has shown, row k = opponent slot k + 1. Slots past the table stay 0.
    played_view = state[STATE_PLAYED].reshape(NUM_OPPONENT_SLOTS, CARD_GRID_SIZE)
    played = played_by_seat(observation)
    for seat in range(num_players):
        slot = relative_slot(seat, me, num_players)
        if slot > 0:
            played_view[slot - 1] = card_grid(played[seat]).reshape(-1)

    # The play to beat, with its pattern, and who played it. Nothing on the table means I lead.
    if observation.table is None:
        state[STATE_LEADING] = 1.0
    else:
        state[STATE_TABLE] = encode_action(observation.table)
        table_slot = relative_slot(observation.table_seat, me, num_players)
        state[STATE_TABLE_SEAT.start + table_slot] = 1.0

    # Each opponent's latest action: walking the history in order, later moves overwrite earlier.
    last_move_view = state[STATE_LAST_MOVE].reshape(NUM_OPPONENT_SLOTS, LAST_MOVE_SIZE)
    for seat, action in observation.history:
        slot = relative_slot(seat, me, num_players)
        if slot > 0:
            row = last_move_view[slot - 1]
            row[:] = 0.0
            if action == PASS:
                row[CARD_GRID_SIZE] = 1.0
            else:
                row[:CARD_GRID_SIZE] = card_grid(action.cards).reshape(-1)

    # Per-slot public counts and flags: cards left, passed this round, out of the hand.
    cards_left_view = state[STATE_CARDS_LEFT].reshape(MAX_PLAYERS, CARDS_LEFT_BITS)
    finished_seats = set(observation.finish_order)
    for seat in range(num_players):
        slot = relative_slot(seat, me, num_players)
        cards_left_view[slot, observation.cards_left[seat]] = 1.0
        if observation.passed[seat]:
            state[STATE_PASSED.start + slot] = 1.0
        if seat in finished_seats:
            state[STATE_FINISHED.start + slot] = 1.0

    # Table size, so one net can tell configs apart.
    state[STATE_PLAYERS.start + num_players - MIN_PLAYERS] = 1.0
    state[STATE_DECKS.start + observation.num_decks - 1] = 1.0

    # Bombs played so far this hand, capped so the one-hot stays fixed width.
    bombs_played = 0
    for seat, action in observation.history:
        if action != PASS and action.pattern.kind == BOMB:
            bombs_played += 1
    state[STATE_BOMBS.start + min(bombs_played, BOMB_CAP)] = 1.0
    return state


def encode_decision(observation, actions, window=HISTORY_WINDOW):
    """
    Encodes one decision: what the seat sees, plus every action it may take.
    observation: an Observation. actions: the legal actions, as from legal_actions (K of them).
    window: int, history length (HISTORY_WINDOW for the control).
    Returns (state, history, mask, action_matrix):
      state float32 [STATE_DIM]; history float32 [window, MOVE_DIM]; mask bool [window];
      action_matrix float32 [K, ACTION_DIM], row i = actions[i], so an argmax over the K
      Q values indexes straight back into actions.
    K is not padded here: padding a batch to its longest K belongs to the learner.
    """
    state = encode_state(observation)
    history, mask = encode_history(observation, window)

    action_matrix = np.zeros((len(actions), ACTION_DIM), dtype=np.float32)
    for index, action in enumerate(actions):
        action_matrix[index] = encode_action(action)
    return state, history, mask, action_matrix
