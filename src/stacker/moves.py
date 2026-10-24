"""
Legal-move enumeration for home-rules Big Two (architecture plan §2; stage plan engine_play).

Moves are built rank by rank instead of testing every subset of the hand: a 17-card hand has
131,072 subsets, too many to check on every turn. First every play that uses one rank is listed,
then every run; classify and beats from patterns.py then keep only the legal ones. Every distinct
set of cards is its own move, because suits and deck copies change the top card and what is left
in the hand.
"""

from itertools import combinations, product
from typing import NamedTuple

from stacker.patterns import MIN_RUN_RANKS, SEVEN, SIX, Pattern, beats, classify


def cards_by_rank(hand):
    """
    Groups a hand's cards by rank.
    hand: list of Cards.
    Returns a dict from rank index to that rank's cards, lowest card first. The hand is sorted
    first, so the result does not depend on the order the cards came in (needed for replay).
    """
    groups = {}
    for card in sorted(hand):
        if card.rank in groups:
            groups[card.rank].append(card)
        else:
            groups[card.rank] = [card]
    return groups


def one_rank_plays(hand):
    """
    Lists every play that uses a single rank: singles, pairs, triples, quads and bigger groups.
    hand: list of Cards.
    Returns a list of card lists. A rank with n copies gives 2^n - 1 plays, one per non-empty
    pick of its cards. Whether a group is a set or a bomb is left to classify.
    """
    plays = []
    for rank_cards in cards_by_rank(hand).values():
        # Every pick of `size` cards from this rank, e.g. both 3♠3♥ and 3♣3♦ as pairs.
        for size in range(1, len(rank_cards) + 1):
            for pick in combinations(rank_cards, size):
                plays.append(list(pick))
    return plays


def picks_across_ranks(groups, ranks, group_size):
    """
    Lists every way to take group_size cards from each of the given ranks.
    groups: dict from cards_by_rank; ranks: list of rank indices, each holding >= group_size cards.
    Returns a list of card lists, one per combination. product crosses one pick per rank, so
    the count is the product of C(copies, group_size) over the ranks.
    """
    picks_per_rank = []
    for rank in ranks:
        picks_per_rank.append(list(combinations(groups[rank], group_size)))

    plays = []
    for one_pick_each in product(*picks_per_rank):
        cards = []
        for pick in one_pick_each:
            cards.extend(pick)
        plays.append(cards)
    return plays


def run_plays(hand, leading):
    """
    Lists every run and multi-run in a hand, at every group size.
    hand: list of Cards; leading: True when the play opens a round (only then is 6-7 allowed).
    Returns a list of card lists. Group size 1 gives plain runs, 2+ gives multi-runs; bomb-sized
    groups are allowed inside a multi-run, as ruled in engine_rules.
    """
    groups = cards_by_rank(hand)

    # The biggest group size any run could use is the most copies of one rank in the hand.
    max_group_size = 0
    for rank_cards in groups.values():
        if len(rank_cards) > max_group_size:
            max_group_size = len(rank_cards)

    plays = []
    for group_size in range(1, max_group_size + 1):
        # From each start rank, extend upward while the next rank has enough copies. Every
        # stretch of 3+ ranks is a run. Rank 2 (the top index) has nothing above it, so no wrap.
        for start in sorted(groups):
            stretch = []
            rank = start
            while rank in groups and len(groups[rank]) >= group_size:
                stretch.append(rank)
                if len(stretch) >= MIN_RUN_RANKS:
                    plays.extend(picks_across_ranks(groups, stretch, group_size))
                rank += 1

        # The 6-7 lead counts as a 3-rank run (5-6-7) at the same group size.
        if leading and SIX in groups and SEVEN in groups:
            if len(groups[SIX]) >= group_size and len(groups[SEVEN]) >= group_size:
                plays.extend(picks_across_ranks(groups, [SIX, SEVEN], group_size))
    return plays


class Play(NamedTuple):
    """
    One legal play.
    cards: tuple of Cards, lowest first. A tuple, so two Plays of the same cards compare equal.
    pattern: its Pattern from classify, kept so the game loop need not classify it again.
    """

    cards: tuple
    pattern: Pattern


def legal_plays(hand, num_decks, previous, opening_card):
    """
    Lists every legal play from a hand. Passing is not included; the game loop adds it.
    hand: list of Cards; num_decks: 1 or 2.
    previous: Pattern of the play on the table, or None when this play leads the round.
    opening_card: the Card the hand's first play must include, or None after the first play.
    Returns a list of Plays.
    """
    leading = previous is None

    # One-rank plays and runs never overlap (runs span 2+ ranks), so there are no duplicates.
    candidates = one_rank_plays(hand) + run_plays(hand, leading)

    plays = []
    for cards in candidates:
        pattern = classify(cards, num_decks, leading)
        if pattern is None:
            continue
        # The hand's first play must include the lowest dealt card.
        if opening_card is not None and opening_card not in cards:
            continue
        # A follower must beat what is on the table.
        if previous is not None and not beats(pattern, previous):
            continue
        plays.append(Play(tuple(cards), pattern))
    return plays
