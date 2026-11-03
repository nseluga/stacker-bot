"""
The heuristic agent: an ordered list of rules copied from Nate's own play (stage plan baselines;
decision log 2026-10-31, "Heuristic is an ordered rule list from Nate's play, not weighted
scores", amended 2026-11-01).

Mechanism. Each turn the agent first computes features once, then walks a rule list; the first
rule that returns a play decides, and a rule that does not apply returns None. Leading (empty
table) uses LEAD_RULES, following (a play to beat) uses FOLLOW_RULES. The features share one
idea: split the hand into the fewest plays (best_split), then ask of each play whether it is a
"control", meaning it is expected to win its round. A play is a control when the estimated chance
that some opponent can beat it is at most CONTROL_MAX_CHANCE. That chance comes from guessing the
opponents' hidden cards by random deals of the unseen cards (beat_chance). A "loser" is a part of
the split that is not a control. A "finish plan" (path out) exists when losers <= single
controls + 1: every loser gives up the lead, a single control can win it back, and the last
loser puts us out (count_plan).

Every feature reads only the Observation: the agent's own hand, the table, card counts and the
public history. Nothing here looks at another seat's hidden cards.
"""

from functools import lru_cache
from typing import NamedTuple

from stacker.agents import greedy_agent, greedy_key
from stacker.cards import RANK_NAMES, build_deck
from stacker.game import PASS
from stacker.moves import legal_plays
from stacker.patterns import BOMB, MIN_RUN_RANKS, SEVEN, SIX

# Distinct ranks in the deck (low to high), used to size per-rank count arrays.
NUM_RANKS = len(RANK_NAMES)

# Remainders whose best split is remembered. One entry is a few hundred bytes, so this caps the
# cache near 50 MB over a long eval.
# ponytail: plain LRU cap; raise it if the piece-9 probe shows split time dominating.
MAX_CACHED_SPLITS = 2**18

# A play (or split part) is a control, expected to win its round, when at most this share of
# sampled deals give some opponent a beater. Set from Nate's leads: he treated 0.08-0.14 as
# safe, 0.42-0.60 as not (decision log 2026-10-31, amended 2026-11-01).
CONTROL_MAX_CHANCE = 0.2

# Sampled deals per beat-chance estimate. The estimate's standard error is at most
# sqrt(0.25 / 200) = 0.035; near 0.2 it is about 0.028.
CONTROL_SAMPLES = 200

# An opponent with this many cards or fewer, still in the round, is a threat to go out, so the
# follow rules try to stop them (decision log 2026-10-31, R4/R5 merged 2026-11-01).
DANGER_CARDS = 2


def unseen_cards(observation):
    """
    Every card this seat has not seen: the full deck, minus its own hand, minus every card played
    so far this hand. These are the cards the opponents' hands are drawn from. observation: a
    game.Observation.
    Returns a list of Cards, lowest first.
    Cards set aside in the deal stay in the list: the seat cannot tell which cards they are, so it
    must treat them as possibly held by an opponent. Cards carry their deck index, so with two decks
    the two copies of a card are removed one at a time, never both by one play.
    """
    # Start from every card in play, then strike our own hand.
    unseen = build_deck(observation.num_decks)
    for card in observation.hand:
        unseen.remove(card)
    # Strike every card any seat has already played this hand (passes play no cards).
    for _seat, action in observation.history:
        if action != PASS:
            for card in action.cards:
                unseen.remove(card)
    return unseen


class Part(NamedTuple):
    """
    One play in a hand's split, by shape only: suits do not change how cards group. Rank indexes
    run low to high (RANK_NAMES order).
    first_rank: lowest rank index. num_ranks: 1 for a set or bomb, 3+ for a run (2 for the 6-7
    lead). group_size: cards of each rank (1 single or plain run, 2 pair or pair-run, ...).
    """

    first_rank: int
    num_ranks: int
    group_size: int


def best_split(hand):
    """
    Splits a hand into the fewest legal plays; the length of the result is the hand's
    plays-to-finish, the number of turns needed to empty it if nothing is ever beaten.
    hand: a list of Cards. Returns a tuple of Parts, lowest first rank first.
    Among splits with equally few plays, the one with the fewest singles wins: a lone card is the
    hardest part to get rid of. Remaining ties go to the first split found.
    """
    # Suits never change how cards group into plays, so reduce the hand to a count per rank. The
    # tuple form lets split_counts cache the answer for every hand with the same counts.
    counts = [0] * NUM_RANKS
    for card in hand:
        counts[card.rank] += 1
    return split_counts(tuple(counts))


@lru_cache(maxsize=MAX_CACHED_SPLITS)
def split_counts(counts):
    """
    best_split on rank counts: counts[rank] is how many cards of that rank remain.
    Returns a tuple of Parts. Remembered per counts, because the same remainder recurs across the
    candidate moves of one decision and across decisions.
    Mechanism (exhaustive search with memoization): the lowest remaining rank must go in some
    play, and any play holding it is either a set of that rank or a run starting at it. Try each,
    split the rest the same way, keep the best.
    Placing the lowest rank first means each split is found once, not once per ordering of its
    parts.
    """
    # Find the lowest rank still held. None left means the hand is empty: nothing to split.
    lowest = None
    for rank in range(NUM_RANKS):
        if counts[rank] > 0:
            lowest = rank
            break
    if lowest is None:
        return ()

    best = None
    best_key = None
    for part in parts_starting_at(counts, lowest):
        # Take this part's cards out, then split whatever is left the same way (recursion).
        rest = list(counts)
        for rank in range(part.first_rank, part.first_rank + part.num_ranks):
            rest[rank] -= part.group_size
        split = (part,) + split_counts(tuple(rest))

        # Score the split: fewer plays first, then fewer lone cards.
        singles = 0
        for each_part in split:
            if each_part.num_ranks == 1 and each_part.group_size == 1:
                singles += 1
        key = (len(split), singles)

        # Strictly fewer only, so on a full tie the first split found is kept.
        if best_key is None or key < best_key:
            best = split
            best_key = key
    return best


def parts_starting_at(counts, lowest):
    """
    Every play shape that holds a card of rank lowest, given rank counts where lowest is the lowest
    rank present. Returns a list of Parts: sets of each size first, then runs, shortest first.
    Runs follow the engine (patterns.classify): 3+ consecutive ranks with the same count of each,
    any group size, ending at the highest rank (the 2) at most, with no wrap past it. 6-7 also
    counts as a run, since split parts are played as leads and the home rules allow it there.
    """
    # Sets: one card of this rank, a pair, a triple, ... up to every copy held.
    parts = []
    for group_size in range(1, counts[lowest] + 1):
        parts.append(Part(lowest, 1, group_size))

    # Runs starting at this rank, at every group size (plain runs, pair-runs, ...).
    for group_size in range(1, counts[lowest] + 1):
        num_ranks = 1
        # Extend the run one rank at a time while the next rank still has group_size cards.
        while lowest + num_ranks < NUM_RANKS and counts[lowest + num_ranks] >= group_size:
            num_ranks += 1
            if num_ranks >= MIN_RUN_RANKS:
                parts.append(Part(lowest, num_ranks, group_size))
            # num_ranks is 2 here: the one legal 2-rank run is 6-7, allowed only as a lead.
            elif lowest == SIX and lowest + 1 == SEVEN:
                parts.append(Part(lowest, num_ranks, group_size))
    return parts


def beatable(pattern, unseen, max_opponent_cards):
    """
    True if some play made only of unseen cards could beat pattern, and fits in an opponent's hand.
    Used by beat_chance, which calls it on one sampled hand at a time, and which defines "control".
    pattern: a patterns.Pattern. unseen: list of Cards (from unseen_cards).
    max_opponent_cards: the most cards any opponent that could answer holds; the caller decides who
    counts (everyone holding cards for a lead, seats that have not passed for a follow).
    Mechanism: works on how many unseen cards of each rank there are, never listing plays (a long
    run over 3 unseen cards per rank has millions of suit combinations). It follows patterns.beats
    for same-shape plays: a beater has the same shape and a higher top card.
    Bombs are ignored as beaters, and a bomb of ours counts as unbeatable: about 3.4% of 4p/1d
    hands hold a quad and 0.9% of 8p/2d hands hold 5+ of a kind (scripts/bomb_rate.py; decision
    log 2026-10-31). Counting them would call almost every play beatable early in the hand, since
    any rank nobody has shown has all its copies unseen.
    ponytail: bomb-blind; a bomb-aware version needs evidence, e.g. a seat silent while a full
    rank stays unseen.
    Known limit: it assumes one opponent could hold every card a beater needs, so it can call a play
    beatable when the cards are really split across seats. That errs toward caution for us.
    """
    # Our bomb: treated as unbeatable, since opposing bombs are ignored (see above).
    if pattern.kind == BOMB:
        return False

    # Tally the unseen cards by rank, and remember the highest unseen card of each rank (among
    # beaters with the same top rank, only a higher suit wins).
    counts = [0] * NUM_RANKS
    highest = [None] * NUM_RANKS
    for card in unseen:
        counts[card.rank] += 1
        if highest[card.rank] is None or card > highest[card.rank]:
            highest[card.rank] = card

    # A beater has as many cards as the play; an opponent holding fewer cards cannot answer.
    if pattern.group_size * pattern.num_ranks > max_opponent_cards:
        return False
    # Same shape: num_ranks consecutive ranks with group_size unseen cards each, topped at or above
    # the play's top rank (patterns.Pattern stores a 6-7 lead as a 3-rank run, so it is checked
    # like one). Slide a window of that shape up the ranks, ending at each possible top rank, and
    # check whether the unseen cards could fill it.
    top_rank = pattern.top_card.rank
    for rank in range(top_rank, NUM_RANKS):
        first_rank = rank - pattern.num_ranks + 1
        if first_rank < 0:
            continue
        # Every rank in the window needs group_size unseen cards.
        window_filled = True
        for window_rank in range(first_rank, rank + 1):
            if counts[window_rank] < pattern.group_size:
                window_filled = False
        if not window_filled:
            continue
        # A higher top rank always wins. At the same top rank, only a higher card of it does.
        if rank > top_rank:
            return True
        if highest[rank] > pattern.top_card:
            return True
    return False


def part_candidates(part, actions):
    """
    The legal plays that match a split Part's shape: exactly its ranks, group_size cards of each.
    This turns the shape-only Part back into real plays with suits.
    part: a Part. actions: legal actions (PASS is skipped). Returns a list of Plays.
    Several can match when suits differ; the caller picks among them.
    """
    wanted_ranks = list(range(part.first_rank, part.first_rank + part.num_ranks))
    matches = []
    for action in actions:
        # Quick rejects: a pass, or the wrong number of cards for this shape.
        if action == PASS:
            continue
        if len(action.cards) != part.num_ranks * part.group_size:
            continue
        # The play must cover exactly the part's ranks, with group_size cards of each.
        count_by_rank = {}
        for card in action.cards:
            count_by_rank[card.rank] = count_by_rank.get(card.rank, 0) + 1
        if sorted(count_by_rank) != wanted_ranks:
            continue
        same_group_size = True
        for count in count_by_rank.values():
            if count != part.group_size:
                same_group_size = False
        if same_group_size:
            matches.append(action)
    return matches


def beat_chance(pattern, unseen, opponent_sizes, rng):
    """
    Estimated chance that some opponent holds a play that beats pattern (bombs ignored, as in
    beatable).
    pattern: a patterns.Pattern. unseen: list of Cards (from unseen_cards).
    opponent_sizes: card counts of the opponents who could answer. rng: the agent's random.Random.
    Mechanism (the sampling step of PIMC, perfect-information Monte Carlo: guess the hidden hands
    by random deals, then judge each guess as if the hands were known; Solinas 2019 on the shelf):
    shuffle the unseen cards and deal them out in the opponents' real hand sizes; cards left over
    were set aside in the deal.
    Check each opponent's sampled hand with beatable, and return the share of deals where anyone
    could beat it. An exact count is only practical for singles (decision log 2026-10-31,
    amended 2026-11-01).
    ponytail: uniform deals ignore what the play shows (a seat that passed on singles likely has no
    high ones); a card-location model is the upgrade.
    """
    # Nobody left to answer: nothing can beat it.
    if len(opponent_sizes) == 0:
        return 0.0
    # Shortcut: if even all unseen cards in one hand cannot beat it, no deal can.
    if not beatable(pattern, unseen, max(opponent_sizes)):
        return 0.0

    # Each sample: shuffle the unseen cards and deal consecutive slices, one per opponent, sized
    # to their real hands. One opponent able to beat it is enough to count the deal. The share of
    # such deals estimates the chance (CONTROL_SAMPLES sets its accuracy).
    pool = list(unseen)
    deals_with_beater = 0
    for _ in range(CONTROL_SAMPLES):
        rng.shuffle(pool)
        start = 0
        for size in opponent_sizes:
            if beatable(pattern, pool[start : start + size], size):
                deals_with_beater += 1
                break
            start += size
    return deals_with_beater / CONTROL_SAMPLES


def is_single(part):
    """True if the Part is one card (a lone single, not a pair or run)."""
    return part.num_ranks == 1 and part.group_size == 1


def split_controls(split, actions, unseen, opponent_sizes, rng):
    """
    For each part of a split, the play to lead for it and whether that play is a control.
    split: tuple of Parts. actions: legal lead plays for the hand the split came from.
    unseen, opponent_sizes, rng: as beat_chance.
    Returns (plays, controls), tuples with one entry per part: the Play to lead for it, and a bool
    that is True if that play is a control. A part's legal plays are tried lowest first (greedy's
    key): the first whose beat chance is <= CONTROL_MAX_CHANCE is its play, so a high suit is
    spent only when a low one would lose. If none is a control, the lowest play is kept and the
    part is a loser. A part with no legal play (the opening card rule) gets None and is a loser.
    """
    plays = []
    controls = []
    for part in split:
        # The legal plays that make up this part, lowest suit first.
        candidates = sorted(part_candidates(part, actions), key=greedy_key)
        # Take the cheapest one that is a control.
        chosen = None
        is_control = False
        for candidate in candidates:
            if beat_chance(candidate.pattern, unseen, opponent_sizes, rng) <= CONTROL_MAX_CHANCE:
                chosen = candidate
                is_control = True
                break
        # No control among them: the part is a loser, and its lowest play is the one to lead.
        if chosen is None and len(candidates) > 0:
            chosen = candidates[0]
        plays.append(chosen)
        controls.append(is_control)
    return tuple(plays), tuple(controls)


def count_plan(split, controls):
    """
    Returns (losers, single_controls) for a split: parts that are not controls, and single-card
    parts that are. split: tuple of Parts. controls: the matching bools from split_controls.
    The finish plan (R8) works when losers <= single_controls + 1: each loser led gives up the
    lead, each single control can win it back, and the last loser puts us out.
    """
    losers = 0
    single_controls = 0
    for index in range(len(split)):
        if not controls[index]:
            losers += 1
        elif is_single(split[index]):
            single_controls += 1
    return losers, single_controls


def opponents_holding_cards(observation):
    """Card counts of every opponent still holding cards, in seat order."""
    sizes = []
    for seat in range(len(observation.cards_left)):
        if seat != observation.seat and observation.cards_left[seat] > 0:
            sizes.append(observation.cards_left[seat])
    return sizes


def without(hand, cards):
    """A copy of hand with cards removed, one at a time (two decks can repeat rank and suit)."""
    remaining = list(hand)
    for card in cards:
        remaining.remove(card)
    return remaining


def go_out(observation, actions, features):
    """
    out: returns the play that uses every card in hand, so we go out and win; None if there is
    none. Shared by LEAD_RULES and FOLLOW_RULES, so it takes the common rule signature
    (observation, actions, features) and ignores features.
    """
    for action in actions:
        if action != PASS and len(action.cards) == len(observation.hand):
            return action
    return None


# ---- Leading -------------------------------------------------------------------------------------


class LeadFeatures(NamedTuple):
    """
    What every lead rule reads, worked out once per lead by lead_features.
    split: the best split, a tuple of Parts, lowest first rank first.
    plays, controls: one entry per part, from split_controls (the play to lead for it, or None;
    whether that play is a control).
    one_card_opponent: True if an opponent holds exactly 1 card.
    """
    split: tuple
    plays: tuple
    controls: tuple
    one_card_opponent: bool


def lead_features(observation, actions, rng):
    """
    Builds the LeadFeatures for a lead. observation: a game.Observation with no table play.
    actions: its legal actions. rng: the agent's random.Random, used by beat_chance.
    On a lead nobody has passed yet, so every opponent still holding cards could answer.
    """
    # Who could answer a lead, and whether any of them is one card from going out.
    opponent_sizes = opponents_holding_cards(observation)
    one_card_opponent = 1 in opponent_sizes
    # The plan for the hand: its best split, and which parts would win their round if led.
    split = best_split(observation.hand)
    plays, controls = split_controls(split, actions, unseen_cards(observation), opponent_sizes, rng)
    return LeadFeatures(split, plays, controls, one_card_opponent)


def lead_finish(observation, actions, features):
    """
    R7 finish: at most one loser left, so nothing can stop us. Lead a control, lowest first rank
    first, and keep the loser for last. Returns None (rule off) with 2+ losers or no control.
    """
    # Two or more losers: some opponent can still take the lead from us, so this rule is off.
    losers, _single_controls = count_plan(features.split, features.controls)
    if losers > 1:
        return None
    # Lead the lowest control; the split lists parts lowest first.
    for index in range(len(features.split)):
        if features.controls[index]:
            return features.plays[index]
    return None


def lead_danger(observation, actions, features):
    """
    R6b danger: an opponent holds 1 card, so a single could let them go out. Lead the lowest part
    that is not a single; if there is none (or none has a legal play), lead the highest single
    held.
    """
    if not features.one_card_opponent:
        return None
    # A pair, triple or run can't be answered by a 1-card hand: lead the lowest such part.
    for index in range(len(features.split)):
        if not is_single(features.split[index]) and features.plays[index] is not None:
            return features.plays[index]
    # Only singles left: lead the highest, the one least likely to be beaten.
    highest_single = None
    for action in actions:
        if len(action.cards) == 1:
            if highest_single is None or action.cards[0] > highest_single.cards[0]:
                highest_single = action
    return highest_single


def lead_setup(observation, actions, features):
    """
    R8 set up the finish: when the finish plan works (count_plan), lead the lowest multi-card
    control first. A control pair or run can win only when we lead it (opponents rarely lead that
    shape for us to answer), while single controls can still win the lead back when following.
    Known limit: it assumes opponents lead singles we can win back with.
    """
    # Too many losers for the single controls to cover: no finish plan, so this rule is off.
    losers, single_controls = count_plan(features.split, features.controls)
    if losers > single_controls + 1:
        return None
    # Lead the lowest multi-card control; it wins the round and keeps the lead with us.
    for index in range(len(features.split)):
        if features.controls[index] and not is_single(features.split[index]):
            return features.plays[index]
    return None


def lead_lowest(observation, actions, features):
    """
    R6 lowest: lead the lowest single that is not a control (an orphan, a loser single); controls
    are kept to win the lead back. No such single: lead the part holding the lowest rank. Returns
    None only if no part has a legal play.
    """
    # First choice: the lowest loser single, the hardest card to get rid of later.
    for index in range(len(features.split)):
        if is_single(features.split[index]) and not features.controls[index]:
            if features.plays[index] is not None:
                return features.plays[index]
    # Otherwise the lowest part with a legal play.
    for index in range(len(features.split)):
        if features.plays[index] is not None:
            return features.plays[index]
    return None


# The lead rules in the order they are tried; the first that returns a play decides. Each is
# rule(observation, actions, features) -> Play or None, so one rule can be tuned or moved alone.
LEAD_RULES = (
    ("out", go_out),
    ("R7", lead_finish),
    ("R6b", lead_danger),
    ("R8", lead_setup),
    ("R6", lead_lowest),
)


def choose_lead(observation, actions, rng):
    """
    Picks the lead when the table is empty (decision log 2026-10-31).
    observation: a game.Observation with no table play. actions: its legal actions.
    rng: the agent's random.Random, for beat_chance.
    Returns (action, rule): rule is the name from LEAD_RULES that chose it, for the replay script.
    Falls back to greedy's choice if no rule returns a play (no part matches a legal play).
    """
    # Work out the features once, then ask each rule in order; the first answer wins.
    features = lead_features(observation, actions, rng)
    for name, rule in LEAD_RULES:
        play = rule(observation, actions, features)
        if play is not None:
            return play, name
    return greedy_agent(observation, actions, None), "fallback"


# ---- Following -----------------------------------------------------------------------------------


class FollowFeatures(NamedTuple):
    """
    What every follow rule reads, worked out once per decision by follow_features.
    plays: the legal plays (PASS left out), lowest first by greedy's key; every other field has one
    entry per play, in the same order.
    frees: one bool per play, True if it leaves fewer plays-to-finish than the hand has now.
    controls: one bool per play, True if its beat chance, against opponents still in this round, is
    <= CONTROL_MAX_CHANCE: it should win the round.
    paths_out: one bool per play, True if it is a control and the hand left after it still has a
    finish plan (count_plan, against every opponent holding cards). False for non-controls.
    threat: True if an opponent with <= DANGER_CARDS cards can still act this round.
    last_defender: True if a threat exists and no other seat acts between us and it.
    """
    plays: tuple
    frees: tuple
    controls: tuple
    paths_out: tuple
    threat: bool
    last_defender: bool


def find_threat(observation):
    """
    Returns (threat, last_defender) for a follow decision.
    Walks the seats after ours in turn order. The round ends when play gets back to the table seat,
    so only seats before it (and the table seat itself, which leads next if everyone passes) matter.
    A threat is the first of those, still holding 1..DANGER_CARDS cards and not passed. Any other
    seat still in the round that we pass on the way could stop it, so we are not the last defender.
    observation: a game.Observation with a table play.
    ponytail: a 2-card threat counts even when the table's shape means it cannot go out this round.
    """
    num_seats = len(observation.cards_left)
    other_defenders = 0
    for step in range(1, num_seats):
        seat = (observation.seat + step) % num_seats
        cards = observation.cards_left[seat]
        # A seat still acts this round only if it holds cards and has not passed.
        in_round = cards > 0 and not observation.passed[seat]
        # First seat close to going out: that is the threat. We are its last defender only if
        # nobody else got a turn between us and it.
        if in_round and cards <= DANGER_CARDS:
            return True, other_defenders == 0
        # Past the table seat the round is over, so later seats don't matter.
        if seat == observation.table_seat:
            break
        # A seat that acts before the threat could stop it in our place.
        if in_round:
            other_defenders += 1
    return False, False


def follow_features(observation, actions, rng):
    """
    Builds the FollowFeatures for a decision with a play on the table.
    observation: a game.Observation. actions: its legal actions (PASS last). rng: for beat_chance.
    Plays-to-finish is the length of best_split; a play is free if the hand left after it splits
    into fewer plays (always true for a whole part of the split; also true for a play that rebuilds
    the rest into fewer parts).
    Controls here are judged against the opponents still in this round (answer_sizes); the
    finish plan after a win is judged against everyone holding cards (all_sizes), since our next
    lead can be answered by any of them.
    """
    # The yardstick for "free": how many plays the hand needs now.
    unseen = unseen_cards(observation)
    plays_to_finish = len(best_split(observation.hand))
    # This round, only opponents who have not passed can answer; later leads face everyone.
    answer_sizes = []
    for seat in range(len(observation.cards_left)):
        if seat == observation.seat or observation.cards_left[seat] == 0:
            continue
        if not observation.passed[seat]:
            answer_sizes.append(observation.cards_left[seat])
    all_sizes = opponents_holding_cards(observation)

    # The plays that beat the table, lowest first, so every rule below can take the first match.
    plays = []
    for action in actions:
        if action != PASS:
            plays.append(action)
    plays.sort(key=greedy_key)

    # One entry per play for each feature, filled in the same order as plays.
    frees = []
    controls = []
    paths_out = []
    for play in plays:
        # free: does the hand left after this play split into fewer plays than now?
        remaining = without(observation.hand, play.cards)
        remaining_split = best_split(remaining)
        frees.append(len(remaining_split) < plays_to_finish)
        # control: will this play likely win the round against the seats still in it?
        is_control = beat_chance(play.pattern, unseen, answer_sizes, rng) <= CONTROL_MAX_CHANCE
        controls.append(is_control)
        # path out: if it wins, we lead next. Score the leftover hand as a lead (no table, every
        # opponent can answer) and check it has a finish plan. Skipped for non-controls: the
        # sampling is the slow part, and a play that won't win gives us no lead to use.
        path_out = False
        if is_control:
            lead_actions = legal_plays(remaining, observation.num_decks, None, None)
            _plays, remaining_controls = split_controls(
                remaining_split, lead_actions, unseen, all_sizes, rng
            )
            losers, single_controls = count_plan(remaining_split, remaining_controls)
            path_out = losers <= single_controls + 1
        paths_out.append(path_out)

    # Is someone about to go out this round, and are we the last seat who can stop them?
    threat, last_defender = find_threat(observation)
    return FollowFeatures(
        tuple(plays), tuple(frees), tuple(controls), tuple(paths_out), threat, last_defender
    )


def follow_danger(observation, actions, features):
    """
    danger (R4+R5 merged, 2026-11-01): a threat can still act this round. Play the lowest control
    that beats the table, since winning the round stops them. With no control: the last defender
    must try, with the highest play; otherwise leave it to the seats still to act (fall through).
    """
    if not features.threat:
        return None
    # Lowest play that should win the round; plays are sorted lowest first.
    for index in range(len(features.plays)):
        if features.controls[index]:
            return features.plays[index]
    # No sure winner. Nobody after us can stop the threat, so throw the strongest play we have.
    if features.last_defender and len(features.plays) > 0:
        return features.plays[-1]
    return None


def follow_free(observation, actions, features):
    """
    R1 free: the lowest play that leaves fewer plays-to-finish and is not a control. Controls are
    left to R8, so they are spent only when they set up a finish.
    """
    for index in range(len(features.plays)):
        if features.frees[index] and not features.controls[index]:
            return features.plays[index]
    return None


def follow_take_lead(observation, actions, features):
    """
    R8 take the lead (replaces R2, 2026-11-01): the lowest control that wins the round and leaves a
    hand with a finish plan. Winning the round gives us the lead to start that plan.
    """
    for index in range(len(features.plays)):
        if features.paths_out[index]:
            return features.plays[index]
    return None


def follow_keep(observation, actions, features):
    """R3 keep: nothing above applies, so pass and keep combos and controls whole. Always fires."""
    return PASS


# The follow rules in the order they are tried, same contract as LEAD_RULES. R3 (pass) is last
# and always answers, so choose_follow never runs out of rules.
FOLLOW_RULES = (
    ("out", go_out),
    ("danger", follow_danger),
    ("R1", follow_free),
    ("R8", follow_take_lead),
    ("R3", follow_keep),
)


def choose_follow(observation, actions, rng):
    """
    Picks the action when a play is on the table (decision log 2026-10-31, amended 2026-11-01).
    observation: a game.Observation with a table play. actions: its legal actions, PASS last.
    rng: the agent's random.Random, for beat_chance.
    Returns (action, rule), as choose_lead. R3 always returns PASS, so a rule always decides.
    """
    # Work out the features once, then ask each rule in order; the first answer wins.
    features = follow_features(observation, actions, rng)
    for name, rule in FOLLOW_RULES:
        action = rule(observation, actions, features)
        if action is not None:
            return action, name


def heuristic_agent(observation, actions, rng):
    """
    The heuristic rung's agent, with the same contract as greedy_agent and random_agent.
    observation: a game.Observation. actions: its legal actions. rng: the agent's random.Random,
    passed through to the control estimates, so the same seed replays the same choices.
    Returns one action from actions. The rule that chose it is dropped here; scratch scripts call
    choose_lead / choose_follow directly when they need it.
    """
    # An empty table means we lead (any play); otherwise we follow the play on it (or pass).
    if observation.table is None:
        action, _rule = choose_lead(observation, actions, rng)
    else:
        action, _rule = choose_follow(observation, actions, rng)
    return action
