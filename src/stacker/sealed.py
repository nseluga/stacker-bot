"""
Sealed deal sets (architecture plan §3 rule 1; stage plan harness).

A sealed deal is stored as its deck order, not as a seed: deal() uses rng.shuffle, and Python
guarantees only random() across versions, not shuffle (decision log 2026-10-24). A stored order
replays through cards.deal_from_order on any Python.

Line format: one deal per line. Each card is its index in build_deck(num_decks), written as two
lowercase hex digits. Indices, not printed cards, so renaming a Card field or symbol can't break an
old file. 52 cards → 104 characters; 104 cards → 208.

On disk: data/sealed-deals/<config>/deals.txt holds the lines; deals.sha256 beside it holds the
file's hash in shasum format, so a set can be checked by hand without this code:
    cd data/sealed-deals/4p1d && shasum -a 256 -c deals.sha256
"""

import hashlib
from pathlib import Path

from stacker.cards import CONFIGS, build_deck

# The tracked home of every sealed set: <repo>/data/sealed-deals, found from this file
# (<repo>/src/stacker/sealed.py), so it doesn't depend on the working directory.
SEALED_ROOT = Path(__file__).resolve().parents[2] / "data" / "sealed-deals"

DEALS_FILE = "deals.txt"
HASH_FILE = "deals.sha256"

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


def config_name(num_players):
    """The folder name for a config: players then decks, e.g. 4 players → "4p1d"."""
    if num_players not in CONFIGS:
        raise ValueError(f"num_players must be 3 to 8, got {num_players}")
    return f"{num_players}p{CONFIGS[num_players].num_decks}d"


def write_sealed(num_players, deck_orders, root=SEALED_ROOT):
    """
    Seals a deal set: writes deals.txt and its sha256 into root/<config>/.
    num_players: 3 to 8. deck_orders: list of deck orders (lists of Cards), one per deal.
    root: only tests change it, to seal into a temporary folder.
    Returns the hex digest. Raises FileExistsError if a set is already there: a set is sealed once
    (rule 1), so the code refuses to overwrite one instead of trusting care.
    """
    num_decks = CONFIGS[num_players].num_decks
    directory = Path(root) / config_name(num_players)
    deals_path = directory / DEALS_FILE
    hash_path = directory / HASH_FILE
    if deals_path.exists() or hash_path.exists():
        raise FileExistsError(f"a sealed set already exists in {directory}")
    if len(deck_orders) == 0:
        raise ValueError("no deals to seal")

    lines = []
    for deck_order in deck_orders:
        line = encode_deck(deck_order, num_decks)
        # Decode it back before anything is written: a wrong deck would otherwise be sealed for good.
        decode_deck(line, num_decks)
        lines.append(line)

    # Bytes, not text mode, so the line ending is "\n" on any OS and the hash doesn't depend on it.
    data = ("\n".join(lines) + "\n").encode("ascii")
    digest = hashlib.sha256(data).hexdigest()

    directory.mkdir(parents=True, exist_ok=True)
    deals_path.write_bytes(data)
    hash_path.write_bytes(f"{digest}  {DEALS_FILE}\n".encode("ascii"))
    return digest


def load_sealed(num_players, root=SEALED_ROOT):
    """
    Loads a sealed deal set after checking its hash.
    num_players: 3 to 8. root: as in write_sealed.
    Returns the list of deck orders, in file order (a confirm run takes the first N).
    Raises ValueError if deals.txt no longer matches deals.sha256, before decoding anything.
    """
    num_decks = CONFIGS[num_players].num_decks
    directory = Path(root) / config_name(num_players)
    data = (directory / DEALS_FILE).read_bytes()
    stored = (directory / HASH_FILE).read_text().split()[0]
    actual = hashlib.sha256(data).hexdigest()
    if actual != stored:
        raise ValueError(f"{directory / DEALS_FILE} changed since sealing: sha256 {actual}, sealed {stored}")

    deck_orders = []
    for line in data.decode("ascii").splitlines():
        deck_orders.append(decode_deck(line, num_decks))
    return deck_orders
