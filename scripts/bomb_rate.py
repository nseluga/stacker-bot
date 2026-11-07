"""
How often a dealt hand holds a bomb, for every table size (3 to 8 players).

Backs the decision log entry of 2026-10-31: the heuristic's "can this beat the table" check
ignores opponent bombs, which is only safe if bombs are rare. A bomb here is one rank held
in bomb-size copies: 4 with one deck, 5 with two (stacker.patterns.BOMB_MIN_SIZE).

Mechanism: for each table size, deal DEALS shuffled tables, count how many copies of each
rank every seat holds, and flag the seat if its biggest group reaches the bomb size. Reports
the share of hands with a bomb and the share of tables where at least one seat has one.
This is a dev simulation on freshly shuffled decks (seed 10), not the sealed deals.
Writes results/baselines/bomb-rate.md.
"""

import random
from pathlib import Path

from stacker.cards import CONFIGS, build_deck
from stacker.patterns import BOMB_MIN_SIZE

# Tables dealt per table size; more tables tighten the estimated rates.
DEALS = 20_000
SEED = 10
RESULT_PATH = Path("results/baselines/bomb-rate.md")


def main():
    """
    Deals DEALS tables per table size and measures how often a bomb-sized rank group appears.
    Takes no arguments. Prints the markdown table and writes it to RESULT_PATH; returns nothing.
    """
    rng = random.Random(SEED)

    # Output starts with the title and table header; one row is added per table size below.
    lines = [
        f"# Bomb rate per dealt hand ({DEALS:,} deals per config, seed {SEED}, "
        "scripts/bomb_rate.py)",
        "",
        "| players | decks | bomb size | per hand | any seat at the table |",
        "|---|---|---|---|---|",
    ]
    for num_players, config in CONFIGS.items():
        # A group of this many equal ranks is a bomb for this table's deck count.
        bomb_size = BOMB_MIN_SIZE[config.num_decks]
        hands_with_bomb = 0
        tables_with_bomb = 0
        for _ in range(DEALS):
            # Shuffle a fresh deck and cut it into consecutive hands, one per seat.
            deck = build_deck(config.num_decks)
            rng.shuffle(deck)
            table_has_bomb = False
            for seat in range(num_players):
                hand = deck[seat * config.hand_size : (seat + 1) * config.hand_size]

                # Count copies of each rank in this hand; the largest count decides bomb or not.
                count_by_rank = {}
                for card in hand:
                    count_by_rank[card.rank] = count_by_rank.get(card.rank, 0) + 1
                if max(count_by_rank.values()) >= bomb_size:
                    hands_with_bomb += 1
                    table_has_bomb = True
            if table_has_bomb:
                tables_with_bomb += 1

        # Per-hand rate counts every seat; per-table asks whether anyone at the table has one.
        per_hand = hands_with_bomb / (DEALS * num_players)
        per_table = tables_with_bomb / DEALS
        lines.append(
            f"| {num_players} | {config.num_decks} | {bomb_size}+ "
            f"| {per_hand:.3f} | {per_table:.3f} |"
        )

    # Save the table for review and echo it to the terminal.
    RESULT_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
