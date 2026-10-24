"""
The hand loop for home-rules Big Two (architecture plan §2; stage plan engine_play).

A hand is a sequence of rounds. The leader plays anything legal; each later seat, in seat order,
must beat the play on the table or pass. A pass locks that seat out until the round ends. The
round ends when everyone else still holding cards has passed, and its winner leads the next one.
The hand stops the moment the 2nd player goes out, scored 2 / 1 / 0.

HandState holds everything the loop needs. It takes the hands directly, so tests can set up exact
positions; new_hand builds one from a fresh deal.
"""

from typing import NamedTuple

from stacker.cards import CONFIGS, deal
from stacker.moves import legal_plays

# The pass action. Its own marker, because None already means "nothing on the table".
PASS = "pass"

# Points for 1st and 2nd out; everyone else scores 0.
POINTS_BY_PLACE = (2, 1)

# The hand ends the moment every scoring place is filled, i.e. when the 2nd player goes out.
FINISHERS_TO_END_HAND = len(POINTS_BY_PLACE)


class HandState:
    """
    The full state of one hand in progress.
    hands: list of card lists, one per seat. Each is copied, so playing never edits the caller's lists.
    num_decks: 1 or 2, passed on to classify.
    aside: cards no one was dealt, kept so tests can check every card is accounted for.
    turn: the seat to act next, or None once the hand is over.
    table: the Play on the table, or None when the seat to act is leading a round.
    table_seat: the seat that made the table play, or None.
    passed: one bool per seat, True once that seat passes this round. Reset at each new round.
    finish_order: seats that have played their last card, first out first.
    opening_card: the card the hand's first play must include; None once that play is made.
    """

    def __init__(self, hands, num_decks, leader_seat, opening_card, aside):
        self.hands = []
        for hand in hands:
            self.hands.append(list(hand))
        self.num_decks = num_decks
        self.aside = list(aside)
        self.turn = leader_seat
        self.table = None
        self.table_seat = None
        self.passed = []
        for _ in hands:
            self.passed.append(False)
        self.finish_order = []
        self.opening_card = opening_card


def new_hand(num_players, rng):
    """
    Deals a fresh hand and returns its starting HandState.
    num_players: 3 to 8. rng: the caller's random.Random, so one seed replays the hand.
    """
    dealt = deal(num_players, rng)
    num_decks = CONFIGS[num_players].num_decks
    return HandState(dealt.hands, num_decks, dealt.leader_seat, dealt.opening_card, dealt.aside)


def legal_actions(state):
    """
    Lists every action the seat to act may take: each legal Play, then PASS if allowed.
    state: a HandState.
    PASS is offered only when a play is on the table, so a round's leader can never pass.
    The order is fixed (generator order, PASS last), so a seeded agent replays exactly.
    """
    if state.table is None:
        previous = None
    else:
        previous = state.table.pattern

    actions = legal_plays(state.hands[state.turn], state.num_decks, previous, state.opening_card)
    if state.table is not None:
        actions.append(PASS)
    return actions


def next_seat(state, after_seat):
    """
    Finds the next seat, going around the table from after_seat, that still holds cards and
    has not passed this round. state: a HandState; after_seat: int.
    Returns that seat, or None if there is none.
    The walk is a full circle, so after_seat itself is checked last. The caller ends the round
    when the result is None or the seat whose play is on the table: no one else can answer it.
    """
    num_seats = len(state.hands)
    for step in range(1, num_seats + 1):
        seat = (after_seat + step) % num_seats
        if len(state.hands[seat]) > 0 and not state.passed[seat]:
            return seat
    return None


def apply_action(state, action):
    """
    Applies one action by the seat to act, changing state in place.
    state: a HandState with turn set; action: a Play or PASS from legal_actions(state).
    The action is trusted, not re-checked: the tests check every action against legal_actions.
    """
    acting_seat = state.turn

    if action == PASS:
        # The pass lock: this seat sits out the rest of the round, even holding a bomb.
        state.passed[acting_seat] = True
    else:
        for card in action.cards:
            state.hands[acting_seat].remove(card)
        state.table = action
        state.table_seat = acting_seat
        # Only the hand's very first play must include the opening card.
        state.opening_card = None

        if len(state.hands[acting_seat]) == 0:
            state.finish_order.append(acting_seat)
            if len(state.finish_order) == FINISHERS_TO_END_HAND:
                state.turn = None
                return

    # If the next seat able to act is the table seat itself, or no one, nobody can answer the
    # table play: the round is over.
    following_seat = next_seat(state, acting_seat)
    if following_seat is None or following_seat == state.table_seat:
        # New round: clear passes and the table. The winner leads if they still hold cards;
        # otherwise the next seat after them that does. (next_seat checks its start seat last,
        # so the winner has to be checked here first.)
        for seat in range(len(state.passed)):
            state.passed[seat] = False
        state.table = None
        winner = state.table_seat
        if len(state.hands[winner]) > 0:
            state.turn = winner
        else:
            state.turn = next_seat(state, winner)
        state.table_seat = None
    else:
        state.turn = following_seat


def points(state):
    """
    Scores a hand. state: a HandState.
    Returns a list with one score per seat: 2 for 1st out, 1 for 2nd out, 0 for everyone else.
    """
    scores = []
    for _ in state.hands:
        scores.append(0)
    for place, seat in enumerate(state.finish_order):
        scores[seat] = POINTS_BY_PLACE[place]
    return scores


class Observation(NamedTuple):
    """
    What one seat may see at its turn: everything public at a real table, plus its own hand.
    Other seats' hands and the aside are never in it, so an agent that gets only this cannot cheat.
    Every field is a tuple or an immutable value, so an agent cannot edit the game through it.
    seat: the seat to act. hand: that seat's cards.
    num_decks, table, table_seat, opening_card: as in HandState.
    passed: one bool per seat. cards_left: one count per seat (public: everyone can count cards).
    finish_order: seats already out, first out first.
    history: every (seat, action) so far this hand, in play order.
    """
    seat: int
    hand: tuple
    num_decks: int
    table: object
    table_seat: object
    passed: tuple
    cards_left: tuple
    finish_order: tuple
    opening_card: object
    history: tuple


def observe(state, history):
    """
    Builds the Observation for the seat to act.
    state: a HandState with turn set. history: the list of (seat, action) played so far; the
    caller keeps it, because HandState doesn't record past plays.
    Lists are copied into tuples, so later moves never change an Observation already handed out.
    """
    cards_left = []
    for hand in state.hands:
        cards_left.append(len(hand))

    return Observation(
        seat=state.turn,
        hand=tuple(state.hands[state.turn]),
        num_decks=state.num_decks,
        table=state.table,
        table_seat=state.table_seat,
        passed=tuple(state.passed),
        cards_left=tuple(cards_left),
        finish_order=tuple(state.finish_order),
        opening_card=state.opening_card,
        history=tuple(history),
    )


def play_out(state, agents, rng):
    """
    Plays an already-dealt hand to the 2nd player out, with one agent per seat.
    state: a HandState (changed in place). agents: list, agents[seat] acts for that seat.
    rng: handed to every agent call.
    Each agent is called as agent(observation, actions, rng) and returns one of actions. It gets
    only the Observation, never the HandState, so it cannot see other hands.
    Returns (trace, points): trace is the list of (seat, action) in play order, points as above.
    The trace is also the public history each observation carries, so there is one list, not two.
    The loop always ends: a leader cannot pass, so every round removes at least one card.
    """
    # A wrong-length list would silently seat the wrong agent, so refuse it here.
    if len(agents) != len(state.hands):
        raise ValueError(f"{len(agents)} agents for {len(state.hands)} seats")

    trace = []
    while state.turn is not None:
        actions = legal_actions(state)
        observation = observe(state, trace)
        action = agents[state.turn](observation, actions, rng)
        trace.append((state.turn, action))
        apply_action(state, action)
    return trace, points(state)


def play_hand(num_players, rng, choose_action):
    """
    Deals and plays one full hand, to the 2nd player out, with the same agent in every seat.
    num_players: 3 to 8. rng: the caller's random.Random; it drives the deal and then the
    agent, so one seed replays the whole hand.
    choose_action(observation, actions, rng) -> one of actions.
    Returns (trace, points), as play_out.
    """
    state = new_hand(num_players, rng)
    agents = []
    for _ in range(num_players):
        agents.append(choose_action)
    return play_out(state, agents, rng)
