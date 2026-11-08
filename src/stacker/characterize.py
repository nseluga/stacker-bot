"""
Measurements of the game itself (architecture plan §5 M2; stage plan characterization).

The engine is left alone. Instead, an agent is wrapped so that each decision it makes also writes
one record: how many actions were legal, whether a pass was among them, and whether it passed.
Every game-shape number (legal moves per turn, moves per hand, pass rate) comes from these records.
"""

import math
import random
import statistics
from typing import NamedTuple

from stacker.game import PASS, new_hand, play_out


class Decision(NamedTuple):
    """
    One agent decision, as counted for M2.
    legal_count: the number of legal actions, PASS included when it was allowed. This is what a
    net scores each turn at M3 (one forward pass per legal action).
    pass_allowed: True when PASS was offered, i.e. a play was on the table to beat. False on a
    lead, where passing is impossible.
    passed: True when the agent chose PASS.
    """

    legal_count: int
    pass_allowed: bool
    passed: bool


def counting_agent(agent, log):
    """
    Wraps agent so that it plays exactly as before and also appends one Decision per call to log.
    agent: agent(observation, actions, rng) -> one of actions. log: a list, shared by every seat
    that uses the wrapper, so one list can collect a hand or a whole run.
    Returns the wrapped agent, with the same call signature.
    """

    def counted(observation, actions, rng):
        # The inner agent decides; the wrapper only watches, so play is unchanged.
        action = agent(observation, actions, rng)

        # The engine lists PASS (last) only when a play is on the table, so membership is the test.
        pass_allowed = PASS in actions
        log.append(Decision(len(actions), pass_allowed, action == PASS))
        return action

    return counted


# statistics.quantiles with n=10 cuts the sorted values every 10%: the first cut point is the
# 10th percentile and the last is the 90th.
DECILE_STEPS = 10


def game_shape(num_players, agent, num_hands, seed):
    """
    Plays num_hands fresh dev hands with agent in every seat and measures the shape of the game.
    num_players: 3 to 8. agent: agent(observation, actions, rng). seed: a dev seed (≥ 10, rule 2).
    Hand i is dealt and played from random.Random(f"{seed}:{i}"), as in evaluate.py, so any one
    hand replays on its own.

    Returns (hand_rows, summary).
    hand_rows: one dict per hand: hand_index, moves, passes, pass_chances (decisions where a pass
    was allowed), max_legal (the largest legal_count in the hand).
    summary: dict with
    - moves per hand: moves_mean, moves_median, moves_p10, moves_p90;
    - legal moves per turn, pooled over every decision: legal_mean, legal_median, legal_p90,
      legal_max (PASS counts as one legal action when allowed; see Decision);
    - pass_rate: all passes / all pass chances. Pooled, not an average of per-hand rates, so a
      hand with 30 chances weighs 10 times a hand with 3: the honest per-decision rate.
    """
    if num_hands < 2:
        raise ValueError("percentiles need at least 2 hands")

    hand_rows = []
    hand_moves = []
    all_legal_counts = []
    total_passes = 0
    total_pass_chances = 0

    for hand_index in range(num_hands):
        # A fresh log per hand, so each hand's decisions can be counted on their own.
        log = []
        counted = counting_agent(agent, log)
        agents = []
        for _ in range(num_players):
            agents.append(counted)

        rng = random.Random(f"{seed}:{hand_index}")
        state = new_hand(num_players, rng)
        play_out(state, agents, rng)

        # Tally this hand's records.
        passes = 0
        pass_chances = 0
        max_legal = 0
        for decision in log:
            all_legal_counts.append(decision.legal_count)
            if decision.pass_allowed:
                pass_chances += 1
            if decision.passed:
                passes += 1
            if decision.legal_count > max_legal:
                max_legal = decision.legal_count

        hand_rows.append(
            {
                "hand_index": hand_index,
                "moves": len(log),
                "passes": passes,
                "pass_chances": pass_chances,
                "max_legal": max_legal,
            }
        )
        hand_moves.append(len(log))
        total_passes += passes
        total_pass_chances += pass_chances

    move_deciles = statistics.quantiles(hand_moves, n=DECILE_STEPS, method="inclusive")
    legal_deciles = statistics.quantiles(all_legal_counts, n=DECILE_STEPS, method="inclusive")

    # A table where nobody ever had a pass chance would make the rate 0/0; report None then.
    if total_pass_chances > 0:
        pass_rate = total_passes / total_pass_chances
    else:
        pass_rate = None

    summary = {
        "moves_mean": statistics.mean(hand_moves),
        "moves_median": statistics.median(hand_moves),
        "moves_p10": move_deciles[0],
        "moves_p90": move_deciles[-1],
        "legal_mean": statistics.mean(all_legal_counts),
        "legal_median": statistics.median(all_legal_counts),
        "legal_p90": legal_deciles[-1],
        "legal_max": max(all_legal_counts),
        "pass_rate": pass_rate,
    }
    return hand_rows, summary


def deal_luck_share(rotation_rows):
    """
    How much of one hand's score spread duplicate pairing removes (decision log 2026-11-07,
    "M2 deal-luck share"). rotation_rows: one list per deal, the test agent's points in each
    rotation, as stacker.evaluate.rotation_scores returns them. Every list has num_players entries.

    Mechanism: a single hand's score varies by the variance of all rotation scores pooled.
    Averaging num_players hands from separate deals would cut that by num_players. Pairing instead
    averages the num_players rotations of one deal, and its spread is the variance of the
    per-deal means. share = 1 - paired / (pooled / num_players).
    1: pairing cancels all of it (the deal only moves luck between seats).
    0: no better than separate deals. Negative: rotations move together (the deal helps every
    seat alike), so pairing gains nothing; identical rotations give 1 - num_players.
    Population variance in both places, so those edge cases come out exact.

    Returns (share, sigma). sigma: sample stdev of the per-deal rotation means, equal to the σ of
    M1's paired differences (subtracting the constant field share does not change the spread).
    share is None when every score is equal (0 / 0).
    """
    if len(rotation_rows) < 2:
        raise ValueError("a variance needs at least 2 deals")

    num_players = len(rotation_rows[0])
    all_scores = []
    deal_means = []
    for scores in rotation_rows:
        all_scores.extend(scores)
        deal_means.append(sum(scores) / num_players)

    # The spread num_players unpaired hands would have, against the spread pairing leaves.
    unpaired_variance = statistics.pvariance(all_scores) / num_players
    paired_variance = statistics.pvariance(deal_means)
    sigma = statistics.stdev(deal_means)

    if unpaired_variance == 0:
        return None, sigma
    return 1 - paired_variance / unpaired_variance, sigma


# Power-analysis multiplier: z for a two-sided 5% test (1.96) plus z for 80% power (0.84).
# Same value as scripts/plot_pilot.py and the M1 confirm's N (decision log "M1 confirm uses δ").
POWER_MULTIPLIER = 2.8


def deals_needed(sigma, delta):
    """
    Deals a paired evaluation needs to detect a true edge of delta points per hand.
    sigma: σ of the per-deal paired difference, from a dev pilot. delta: the edge, > 0.

    Mechanism (Colas et al. 2018 §4.2): the mean of N deals has standard error σ/√N. The CI's low
    end clears 0 at a 5% two-sided level once the mean is 1.96 standard errors above 0, and a
    true edge of δ lands there 80% of the time when δ = (1.96 + 0.84)·σ/√N. Solved for N and
    rounded up: N = ceil((2.8·σ/δ)²).
    """
    if sigma < 0:
        raise ValueError(f"sigma must be ≥ 0, got {sigma}")
    if delta <= 0:
        raise ValueError(f"delta must be > 0, got {delta}")
    return math.ceil((POWER_MULTIPLIER * sigma / delta) ** 2)
