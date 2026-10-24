"""
Test helpers: write cards as short strings so tests read like the home rules doc.

"3s 10h 2h'" means 3♠, 10♥ and the bad copy of 2♥. Each token is a rank name, a suit
letter, and an optional mark for a bad copy. Only tests type cards in, so the suit letters
live here and not in the engine.
"""

from itertools import combinations

from stacker.cards import BAD_COPY_MARK, BAD_DECK, GOOD_DECK, RANK_NAMES, Card
from stacker.patterns import beats, classify

# Suit letters typed in tests, in the same order as SUIT_SYMBOLS (♠ ♣ ♦ ♥).
SUIT_LETTERS = ("s", "c", "d", "h")


def parse_cards(text):
    """
    Turns a space-separated string of cards into a list of Cards, in the order written.
    text: str, e.g. "3s 10h 2h'". Each token is a rank name from RANK_NAMES, then a suit
    letter from SUIT_LETTERS, then an optional BAD_COPY_MARK for a bad copy.
    Raises ValueError naming the token when a token cannot be read.
    """
    cards = []
    for token in text.split():
        # A trailing mark means the bad copy; without it, the card is the good copy.
        if token.endswith(BAD_COPY_MARK):
            deck = BAD_DECK
            rank_and_suit = token[:-1]
        else:
            deck = GOOD_DECK
            rank_and_suit = token

        # The suit letter is the last character left; everything before it is the rank name.
        rank_name = rank_and_suit[:-1]
        suit_letter = rank_and_suit[-1:]
        if rank_name not in RANK_NAMES or suit_letter not in SUIT_LETTERS:
            raise ValueError(f"cannot read card {token!r}")

        card = Card(RANK_NAMES.index(rank_name), SUIT_LETTERS.index(suit_letter), deck)
        cards.append(card)
    return cards


def classify_text(text, num_decks=1, leading=False):
    """Parse cards written like '3s 4s 5h' and classify them."""
    return classify(parse_cards(text), num_decks, leading)


def brute_force_plays(hand, num_decks, previous, opening_card):
    """
    Slow reference for moves.legal_plays: tries every non-empty subset of the hand.
    Same inputs as legal_plays. Returns a list of card tuples, lowest card first.
    Too slow for the game loop (2^n subsets), but plainly correct, so tests compare against it.
    """
    leading = previous is None
    plays = []
    for size in range(1, len(hand) + 1):
        for cards in combinations(sorted(hand), size):
            pattern = classify(list(cards), num_decks, leading)
            if pattern is None:
                continue
            if opening_card is not None and opening_card not in cards:
                continue
            if previous is not None and not beats(pattern, previous):
                continue
            plays.append(cards)
    return plays
