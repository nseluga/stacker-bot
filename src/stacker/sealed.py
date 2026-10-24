"""
Sealed deal sets (architecture plan §3 rule 1; stage plan harness).

A sealed deal is stored as its deck order, not as a seed: deal() uses rng.shuffle, and Python
guarantees only random() across versions, not shuffle (decision log 2026-10-24). A stored order
replays through cards.deal_from_order on any Python.

Line format: one deal per line. Each card is its index in build_deck(num_decks), written as two
lowercase hex digits. Indices, not printed cards, so renaming a Card field or symbol can't break an
old file. 52 cards → 104 characters; 104 cards → 208.
"""

from stacker.cards import build_deck

# Characters per card in a line: two hex digits cover indices 0–255, enough for 104 cards.
CHARS_PER_CARD = 2

# The only characters the encoder writes. Checked by hand, because int(pair, 16) also accepts
# "+1" or " a", and a line the encoder could never write must not decode.
HEX_DIGITS = "0123456789abcdef"


def encode_deck(deck_order, num_decks):
    """
    Writes a deck order as one line of text.
    deck_order: list of Cards, the whole deck. num_decks: 1 or 2.
    Returns a string of CHARS_PER_CARD hex digits per card.
    """
    index_of = {}
    for index, card in enumerate(build_deck(num_decks)):
        index_of[card] = index

    line = ""
    for card in deck_order:
        line += format(index_of[card], "02x")
    return line


def decode_deck(line, num_decks):
    """
    Reads a line written by encode_deck back into a deck order.
    line: the text, without its newline. num_decks: 1 or 2.
    Returns a list of Cards. Raises ValueError unless the line holds every card exactly once,
    so a malformed file fails here with a clear message instead of dealing a wrong game.
    """
    deck = build_deck(num_decks)
    if len(line) != len(deck) * CHARS_PER_CARD:
        raise ValueError(f"expected {len(deck) * CHARS_PER_CARD} characters, got {len(line)}")

    deck_order = []
    seen = set()
    for start in range(0, len(line), CHARS_PER_CARD):
        pair = line[start : start + CHARS_PER_CARD]
        for character in pair:
            if character not in HEX_DIGITS:
                raise ValueError(f"not lowercase hex at position {start}: {pair!r}")
        index = int(pair, 16)
        if index >= len(deck):
            raise ValueError(f"index {index} at position {start} is past the {len(deck)}-card deck")
        if index in seen:
            raise ValueError(f"card index {index} repeated at position {start}")
        seen.add(index)
        deck_order.append(deck[index])
    return deck_order
