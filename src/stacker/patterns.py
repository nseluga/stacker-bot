"""
Pattern recognition and beat comparison for home-rules Big Two (architecture plan §2;
stage plan engine_rules, "Rules confirmed with Nate").

A play is classified into a Pattern: its kind (set, run, bomb), how many cards share each
rank (group size), how many ranks it spans, and its highest card. Only classify needs the
deck count, because it decides whether a group of one rank is a set or a bomb.
"""

from typing import NamedTuple

from stacker.cards import RANK_NAMES, Card

SET = "set"  # one rank: single, pair, triple, or (2 decks) quad
RUN = "run"  # 3+ consecutive ranks, same count of each; group size 1 is a plain run
BOMB = "bomb"  # 1 deck: a quad. 2 decks: 5 to 8 of a kind.

# Smallest one-rank group that counts as a bomb, by number of decks.
BOMB_MIN_SIZE = {1: 4, 2: 5}

MIN_RUN_RANKS = 3

# Rank indices for the 6-7 lead, which counts as a 3-rank run (5-6-7) at its group size.
SIX = RANK_NAMES.index("6")
SEVEN = RANK_NAMES.index("7")


class Pattern(NamedTuple):
    """
    What a play is, for comparing it to another play.
    kind: SET, RUN or BOMB.
    group_size: cards per rank (1 for a single or plain run, 2 for a pair or pair run, ...).
    num_ranks: ranks spanned. 1 for a set or bomb; 3 for a 6-7 lead.
    top_card: highest card in the play; its suit and deck break ties.
    """

    kind: str
    group_size: int
    num_ranks: int
    top_card: Card


def classify(cards, num_decks, leading):
    """
    Return the Pattern for this play, or None if it is not a legal play.
    leading is True when the play opens a round; only then is 6-7 allowed.
    Cards are assumed to come from one hand, so no card appears twice.
    """
    if len(cards) == 0:
        return None
    top_card = max(cards)

    count_by_rank = {}
    for card in cards:
        if card.rank in count_by_rank:
            count_by_rank[card.rank] += 1
        else:
            count_by_rank[card.rank] = 1
    ranks = sorted(count_by_rank)

    # One rank: a set, or a bomb if the group is big enough for this deck count.
    if len(ranks) == 1:
        group_size = len(cards)
        if group_size >= BOMB_MIN_SIZE[num_decks]:
            return Pattern(BOMB, group_size, 1, top_card)
        else:
            return Pattern(SET, group_size, 1, top_card)

    # Several ranks: must be a run. Every rank needs the same number of cards.
    # Bomb-sized groups are fine here (3333 4444 5555 is a run of group size 4).
    group_size = count_by_rank[ranks[0]]
    for rank in ranks:
        if count_by_rank[rank] != group_size:
            return None

    # Ranks must be consecutive. A wrap like A-2-3 is ranks 11, 12, 0, which fails here.
    for i in range(1, len(ranks)):
        if ranks[i] != ranks[i - 1] + 1:
            return None

    if len(ranks) >= MIN_RUN_RANKS:
        return Pattern(RUN, group_size, len(ranks), top_card)
    if leading and ranks == [SIX, SEVEN]:
        return Pattern(RUN, group_size, MIN_RUN_RANKS, top_card)
    return None


def beats(new, previous):
    """
    True if the play `new` beats the play `previous` (both Patterns from classify).
    A bomb beats any non-bomb. Bomb vs bomb: the bigger group wins, then the higher card.
    Otherwise the plays must have the same shape (kind, group size, ranks spanned, which
    together fix the card count) and the higher top card wins. A 6-7 lead is stored as a
    3-rank run topped by its 7, so it needs no special case here.
    """
    if new.kind == BOMB and previous.kind != BOMB:
        return True
    if previous.kind == BOMB and new.kind != BOMB:
        return False

    if new.kind == BOMB and previous.kind == BOMB:
        if new.group_size != previous.group_size:
            return new.group_size > previous.group_size
        return new.top_card > previous.top_card

    if new.kind != previous.kind:
        return False
    if new.group_size != previous.group_size:
        return False
    if new.num_ranks != previous.num_ranks:
        return False
    return new.top_card > previous.top_card
