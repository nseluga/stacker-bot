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
