"""
Paired evaluation on duplicate deals (architecture plan §3 rule 4; stage plan harness).

One stored deal is played once per seat rotation, num_players rotations in all. In rotation r the
test agent holds its format's seats shifted by r, and the field agent holds the rest. Over all
rotations the test agent sits in every seat equally often, so it plays every hand of the deal.
Luck from the deal (who got the strong hand, who leads) then cancels out of the comparison instead
of adding noise to it.
"""

import random
import statistics

from stacker.cards import CONFIGS, deal_from_order
from stacker.game import POINTS_BY_PLACE, HandState, play_out

# Every hand pays out the same total (2 for 1st out + 1 for 2nd), whoever wins.
POINTS_PER_HAND = sum(POINTS_BY_PLACE)

# Bootstrap resamples per CI (stage plan harness). More only tightens the CI's own Monte Carlo noise.
BOOTSTRAP_RESAMPLES = 10_000

# statistics.quantiles with n=40 cuts the sorted means every 2.5%: the first cut point is the
# 2.5th percentile and the last is the 97.5th, the two ends of a 95% CI.
QUANTILE_STEPS = 40


def rotated_seats(test_seats, rotation, num_players):
    """
    The seats the test agent holds in one rotation: each seat of test_seats moved rotation seats on,
    wrapping past the last seat back to 0.
    test_seats: the format at rotation 0. (0,) is 1v3, (0, 1) 2v2 adjacent, (0, 2) 2v2 opposite,
    (0, 1, 2) 3v1. Returns a tuple of seats.
    Raises ValueError on a format that would give wrong numbers silently: no test seat, a repeated
    seat, a seat outside 0..num_players-1, or the test agent in every seat (no field to compare to).
    """
    if len(test_seats) == 0:
        raise ValueError("test_seats is empty")
    if len(test_seats) >= num_players:
        raise ValueError(f"test agent holds all {num_players} seats, so there is no field agent")
    if len(set(test_seats)) != len(test_seats):
        raise ValueError(f"test_seats repeats a seat: {test_seats}")
    for seat in test_seats:
        if seat < 0 or seat >= num_players:
            raise ValueError(f"seat {seat} is outside 0..{num_players - 1}")

    seats = []
    for seat in test_seats:
        seats.append((seat + rotation) % num_players)
    return tuple(seats)


def rotation_scores(deck_order, num_players, test_agent, field_agent, test_seats, seed, deal_index):
    """
    Plays one stored deal once per rotation (0 to num_players - 1) and returns the test agent's
    points in each: a list of num_players numbers. Each is the sum of the points (2 / 1 / 0) of
    the seats the test agent held in that rotation.
    deck_order: list of Cards, the deal (a decoded sealed line, or a dev shuffle). Not changed.
    test_agent, field_agent: agent(observation, actions, rng) -> one of actions.
    test_seats: the format, as in rotated_seats. seed, deal_index: name the agent rng.
    Each rotation's agents share random.Random(f"{seed}:{deal_index}:{rotation}"). That rng is
    separate from the deal, so the same deal replays under any agents. A string seed is hashed
    inside random, so it does not depend on PYTHONHASHSEED.
    """
    num_decks = CONFIGS[num_players].num_decks
    scores = []
    for rotation in range(num_players):
        held = rotated_seats(test_seats, rotation, num_players)

        # The test agent sits in this rotation's seats; the field agent in every other seat.
        agents = []
        for seat in range(num_players):
            if seat in held:
                agents.append(test_agent)
            else:
                agents.append(field_agent)

        # A fresh state per rotation, because play_out changes the state in place.
        dealt = deal_from_order(deck_order, num_players)
        state = HandState(dealt.hands, num_decks, dealt.leader_seat, dealt.opening_card, dealt.aside)
        agent_rng = random.Random(f"{seed}:{deal_index}:{rotation}")
        _trace, seat_points = play_out(state, agents, agent_rng)

        score = 0
        for seat in held:
            score += seat_points[seat]
        scores.append(score)
    return scores


def paired_differences(deck_orders, num_players, test_agent, field_agent, test_seats, seed):
    """
    The paired per-deal difference for each deal: points per hand the test agent scored, minus
    what the field agent would have scored in the same seats. Returns a list of floats in deal
    order. Positive means the test agent is stronger.
    deck_orders: list of deals, each a list of Cards. Its position in the list is its deal_index.
    Other arguments as in rotation_scores.

    The field's share needs no second table. Picture the field agent alone at the table: the hand
    pays out POINTS_PER_HAND in total. Over all rotations the test agent's seats cover every seat
    exactly len(test_seats) times, so whatever that table scored, its points in those seats
    average to len(test_seats) * POINTS_PER_HAND / num_players. Example, 4p 1v3: the field share
    is 0.75; a test agent averaging 1.25 points per hand gives +0.50.
    """
    # Both terms are a whole number divided by num_players, so equal agents give exactly 0.0.
    field_share = len(test_seats) * POINTS_PER_HAND / num_players

    differences = []
    for deal_index, deck_order in enumerate(deck_orders):
        scores = rotation_scores(
            deck_order, num_players, test_agent, field_agent, test_seats, seed, deal_index
        )
        test_mean = sum(scores) / num_players
        differences.append(test_mean - field_share)
    return differences


def bootstrap_ci(values, rng):
    """
    Mean and percentile-bootstrap 95% CI of values (the per-deal differences).
    values: list of floats, one per deal. rng: a random.Random, so a seed replays the CI.
    Returns (mean, low, high). The ladder gate (plan §3 rule 3) passes when low > 0.

    How: draw len(values) values from values with replacement and take the mean; repeat
    BOOTSTRAP_RESAMPLES times. The spread of those means shows how much the mean would move on a
    different draw of deals. The 2.5th and 97.5th percentiles of them are the CI.
    Percentile, not BCa: for a mean over thousands of deals the two give the same coverage, and
    percentile is the most robust (Agarwal et al. 2021 Fig. 6; André et al. 2026 §4.1, §5).
    """
    if len(values) == 0:
        raise ValueError("no values to bootstrap")

    mean = sum(values) / len(values)
    # pure-Python resampling, ~27 s per CI at 10,000 deals; numpy when it arrives at M3.
    resample_means = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        resample = rng.choices(values, k=len(values))
        resample_means.append(sum(resample) / len(resample))

    cut_points = statistics.quantiles(resample_means, n=QUANTILE_STEPS, method="inclusive")
    low = cut_points[0]
    high = cut_points[-1]
    return mean, low, high
