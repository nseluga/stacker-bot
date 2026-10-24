"""
Cards and card order for home-rules Big Two (architecture plan §2).

A card is a (rank, suit, deck) tuple. Python compares tuples field by field, left to
right, so sorting cards by this tuple gives the game's full card order: rank first
(3 lowest, 2 highest), then suit (♠ < ♣ < ♦ < ♥), then deck (bad copy just below the
good copy of the same card). With one deck, every card is a good copy.
"""

from typing import NamedTuple

# Rank names, lowest to highest. A card's rank field is its index here (3 → 0, 2 → 12).
RANK_NAMES = ("3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2")

# Suit symbols, lowest to highest. A card's suit field is its index here (♠ → 0, ♥ → 3).
SUIT_SYMBOLS = ("♠", "♣", "♦", "♥")

# Deck field values. The bad copy is the smaller value, so it sorts just below its good copy.
BAD_DECK = 0
GOOD_DECK = 1

# Printed after a bad copy so the two copies are told apart: 3♠' is bad, 3♠ is good.
BAD_COPY_MARK = "'"


class Card(NamedTuple):
    """
    One playing card.
    rank: int, index into RANK_NAMES (0 = 3, 12 = 2).
    suit: int, index into SUIT_SYMBOLS (0 = ♠, 3 = ♥).
    deck: int, GOOD_DECK or BAD_DECK; defaults to GOOD_DECK, the only deck in 1-deck games.
    The field order IS the game's card order, so comparisons and sorted() need no extra code.
    Do not reorder the fields.
    """

    rank: int
    suit: int
    deck: int = GOOD_DECK

    def __str__(self):
        name = RANK_NAMES[self.rank] + SUIT_SYMBOLS[self.suit]
        if self.deck == BAD_DECK:
            return name + BAD_COPY_MARK
        else:
            return name


# Deck field values present for each deck count. One deck has only good copies; two decks
# add a bad copy of every card. Listed low to high so build_deck returns cards in game order.
DECK_VALUES_BY_COUNT = {1: (GOOD_DECK,), 2: (BAD_DECK, GOOD_DECK)}


def build_deck(num_decks):
    """
    Returns every card in play for a game, lowest to highest.
    num_decks: int, 1 or 2 (1 deck for 3–5 players, 2 for 6–8).
    Returns a list of 52 Cards for 1 deck, or 104 for 2.
    """
    if num_decks not in DECK_VALUES_BY_COUNT:
        raise ValueError(f"num_decks must be 1 or 2, got {num_decks}")

    # Looping rank, then suit, then deck follows the card order, so the list comes out sorted.
    cards = []
    for rank in range(len(RANK_NAMES)):
        for suit in range(len(SUIT_SYMBOLS)):
            for deck in DECK_VALUES_BY_COUNT[num_decks]:
                cards.append(Card(rank, suit, deck))
    return cards


def find_opening(hands):
    """
    Finds who leads the first round and the card their opening play must include.
    hands: list of lists of Cards, one list per seat, as dealt.
    Returns (leader_seat, opening_card): the lowest dealt card and the seat holding it.
    Set-aside cards are not in any hand, so if 3♠ is aside the next lowest dealt card leads.
    With 2 decks the bad copy sorts below the good copy, so the bad copy's holder leads.
    """
    leader_seat = None
    opening_card = None
    for seat, hand in enumerate(hands):
        for card in hand:
            if opening_card is None or card < opening_card:
                leader_seat = seat
                opening_card = card

    if opening_card is None:
        raise ValueError("no cards were dealt")
    return leader_seat, opening_card


class GameConfig(NamedTuple):
    """How many decks a table uses and how many cards each seat is dealt."""

    num_decks: int
    hand_size: int


# Player count → table setup (architecture plan §2). Cards left after dealing go aside.
CONFIGS = {
    3: GameConfig(num_decks=1, hand_size=17),
    4: GameConfig(num_decks=1, hand_size=13),
    5: GameConfig(num_decks=1, hand_size=10),
    6: GameConfig(num_decks=2, hand_size=17),
    7: GameConfig(num_decks=2, hand_size=14),
    8: GameConfig(num_decks=2, hand_size=13),
}


class Deal(NamedTuple):
    """
    The result of one deal.
    hands: list of sorted card lists, one per seat (seat 0 first).
    aside: cards no one was dealt, unseen for the whole game.
    leader_seat, opening_card: who leads the first round, and the card that decided it.
    """

    hands: list
    aside: list
    leader_seat: int
    opening_card: Card


def deal(num_players, rng):
    """
    Shuffle and deal one game for num_players seats.
    rng is a random.Random the caller owns, so one seed replays the whole game.
    Seats take consecutive blocks of the shuffled deck. After a uniform shuffle this
    gives every hand the same odds as dealing one card at a time around the table.
    """
    if num_players not in CONFIGS:
        raise ValueError(f"num_players must be 3 to 8, got {num_players}")
    deck = build_deck(CONFIGS[num_players].num_decks)
    rng.shuffle(deck)
    return deal_from_order(deck, num_players)


def deal_from_order(deck_order, num_players):
    """
    Deals a deck that is already in a fixed order: the second half of deal, split out so a
    sealed deal (stored as its deck order) replays through exactly the same code as a live one.
    deck_order: list of Cards, the whole deck for this config. Not changed: slicing copies.
    Returns a Deal, as deal does.
    """
    if num_players not in CONFIGS:
        raise ValueError(f"num_players must be 3 to 8, got {num_players}")
    config = CONFIGS[num_players]

    # A short or long order would silently deal wrong-sized hands or aside, so refuse it.
    deck_size = len(build_deck(config.num_decks))
    if len(deck_order) != deck_size:
        raise ValueError(f"{num_players} players need {deck_size} cards, got {len(deck_order)}")
    deck = deck_order

    hands = []
    for seat in range(num_players):
        start = seat * config.hand_size
        hand = deck[start : start + config.hand_size]
        hand.sort()
        hands.append(hand)

    dealt_count = num_players * config.hand_size
    aside = deck[dealt_count:]

    leader_seat, opening_card = find_opening(hands)
    return Deal(hands, aside, leader_seat, opening_card)
