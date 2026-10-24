"""
Tests for the evaluation harness (stage plan harness, piece 11; check "Check that fails if the code
is wrong").

Information: an agent's Observation must not change when other seats' hands change, and an agent
is only ever handed an Observation, never the HandState with every hand in it.
Sealing: a set sealed into a temporary folder loads back as the same deck orders, and a deal swapped
inside the file is caught by the hash. The temporary folder runs the same code as the real sets
without decoding all 50,000 deals per config. One more test hashes the real tracked files (no
decode) so damage to data/sealed-deals/ is caught too.
Evaluation: with one deterministic agent as both test and field, every paired difference is exactly
0 in every format; the CI behaves at its two known answers; the same seed replays the same numbers.
Seeds start at 10: seeds 0-2 are kept for confirm runs.
"""

import hashlib
import random

import pytest

from stacker.agents import random_agent
from stacker.cards import CONFIGS, build_deck, deal_from_order
from stacker.evaluate import bootstrap_ci, paired_differences
from stacker.game import Observation, new_hand, observe, play_out
from stacker.sealed import (
    CHARS_PER_CARD,
    DEALS_FILE,
    HASH_FILE,
    SEALED_ROOT,
    config_name,
    load_sealed,
    write_sealed,
)

# Deals per config in the temporary seal: enough to cover many deck orders, fast to write and load.
SEAL_TEST_DEALS = 20

# The four formats (stage plan harness): 1v3, 2v2 adjacent, 2v2 opposite, 3v1 at 4 players. At other
# table sizes the same seat sets are used where they leave at least one field seat.
FORMATS = ((0,), (0, 1), (0, 2), (0, 1, 2))

# Deals per (config, format) in the exact-zero test: every rotation of every deal must give 0.
EXACT_ZERO_DEALS = 5

EXACT_ZERO_CASES = []
for num_players in CONFIGS:
    for test_seats in FORMATS:
        if len(test_seats) < num_players:
            EXACT_ZERO_CASES.append((num_players, test_seats))


def dev_deck_orders(num_players, count, seed):
    """count shuffled deck orders for num_players, from one random.Random(seed)."""
    rng = random.Random(seed)
    deck_orders = []
    for _ in range(count):
        deck_order = build_deck(CONFIGS[num_players].num_decks)
        rng.shuffle(deck_order)
        deck_orders.append(deck_order)
    return deck_orders


@pytest.mark.parametrize("num_players", list(CONFIGS))
def test_observation_hides_other_hands(num_players):
    state = new_hand(num_players, random.Random(10))
    seat = state.turn
    opponents = []
    for other in range(num_players):
        if other != seat:
            opponents.append(other)
    first, second = opponents[0], opponents[1]
    before = observe(state, [])

    # Swap one card between two opponents: same card counts, different hidden hands.
    state.hands[first][0], state.hands[second][0] = state.hands[second][0], state.hands[first][0]
    assert observe(state, []) == before

    # Swap one card between the seat to act and an opponent: the seat's own hand changes.
    state.hands[seat][0], state.hands[first][0] = state.hands[first][0], state.hands[seat][0]
    assert observe(state, []) != before


@pytest.mark.parametrize("num_players", list(CONFIGS))
def test_agents_get_only_the_observation(num_players):
    seen_types = []

    def spy_agent(observation, actions, rng):
        seen_types.append(type(observation))
        return random_agent(observation, actions, rng)

    agents = []
    for _ in range(num_players):
        agents.append(spy_agent)
    play_out(new_hand(num_players, random.Random(10)), agents, random.Random(11))

    assert len(seen_types) > 0
    for seen_type in seen_types:
        assert seen_type is Observation


@pytest.mark.parametrize("num_players", list(CONFIGS))
def test_seal_round_trip(num_players, tmp_path):
    deck_orders = dev_deck_orders(num_players, SEAL_TEST_DEALS, seed=10)
    write_sealed(num_players, deck_orders, root=tmp_path)
    loaded = load_sealed(num_players, root=tmp_path)

    # The same deck orders in the same order, so a confirm's deal_index means the same deal.
    assert loaded == deck_orders

    # Each loaded deal hands out every card exactly once: hands plus aside are the whole deck.
    full_deck = sorted(build_deck(CONFIGS[num_players].num_decks))
    for deck_order in loaded:
        dealt = deal_from_order(deck_order, num_players)
        every_card = list(dealt.aside)
        for hand in dealt.hands:
            every_card.extend(hand)
        assert sorted(every_card) == full_deck


def test_seal_tamper_raises(tmp_path):
    num_players = 4
    write_sealed(num_players, dev_deck_orders(num_players, SEAL_TEST_DEALS, seed=10), root=tmp_path)
    deals_path = tmp_path / config_name(num_players) / DEALS_FILE

    # Swap the first two cards of the first deal. The line is still a valid deck, so only the hash
    # can tell that the deal changed (a broken line would also fail to decode, hiding a skipped check).
    lines = deals_path.read_text().splitlines()
    first_card = lines[0][:CHARS_PER_CARD]
    second_card = lines[0][CHARS_PER_CARD : 2 * CHARS_PER_CARD]
    lines[0] = second_card + first_card + lines[0][2 * CHARS_PER_CARD :]
    deals_path.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="changed since sealing"):
        load_sealed(num_players, root=tmp_path)


@pytest.mark.parametrize("num_players", list(CONFIGS))
def test_real_sealed_sets_match_their_hash(num_players):
    # Hash only, no decode: decoding 50,000 deals per config would blow the suite's time budget.
    directory = SEALED_ROOT / config_name(num_players)
    actual = hashlib.sha256((directory / DEALS_FILE).read_bytes()).hexdigest()
    stored = (directory / HASH_FILE).read_text().split()[0]
    assert actual == stored


def first_legal_agent(observation, actions, rng):
    """Always takes the first legal action, so it plays the same deal the same way every time."""
    return actions[0]


@pytest.mark.parametrize("num_players, test_seats", EXACT_ZERO_CASES)
def test_same_agent_gives_exactly_zero(num_players, test_seats):
    # Test and field are the same deterministic agent, so every rotation is the same game. The test
    # seats then cover each seat len(test_seats) times and collect exactly the field's share.
    # Any mismatch in rotations or in the field share shows up as a nonzero difference.
    deck_orders = dev_deck_orders(num_players, EXACT_ZERO_DEALS, seed=10)
    differences = paired_differences(
        deck_orders, num_players, first_legal_agent, first_legal_agent, test_seats, seed=10
    )
    assert differences == [0.0] * EXACT_ZERO_DEALS


def test_bootstrap_of_constant_has_zero_width():
    # Every resample of a constant list has the same mean, so the CI collapses onto it.
    assert bootstrap_ci([0.5] * 50, random.Random(10)) == (0.5, 0.5, 0.5)


def test_bootstrap_random_vs_random_covers_zero():
    # Random against itself has a true difference of 0, so its 95% CI should cover 0. Seeded, so the
    # outcome is fixed: these are the same 200 deals the script draws with --seed 10.
    deck_orders = dev_deck_orders(4, 200, seed=10)
    differences = paired_differences(deck_orders, 4, random_agent, random_agent, (0,), seed=10)
    _mean, low, high = bootstrap_ci(differences, random.Random("10:bootstrap"))
    assert low <= 0 <= high


def test_paired_differences_replay():
    deck_orders = dev_deck_orders(4, 20, seed=10)
    first = paired_differences(deck_orders, 4, random_agent, random_agent, (0, 2), seed=10)
    again = paired_differences(deck_orders, 4, random_agent, random_agent, (0, 2), seed=10)
    assert first == again
