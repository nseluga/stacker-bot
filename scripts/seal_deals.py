"""
Seals the v2 test deal sets: 50,000 deals for each of the six configs (architecture plan §3
rule 1; decision log 2026-10-24, "Sealed cap is 50,000 deals for every config").

Run once, before any training:
    PYTHONPATH=src .venv/bin/python scripts/seal_deals.py
Output: data/sealed-deals/<config>/deals.txt + deals.sha256, tracked in git. write_sealed refuses
to overwrite, so a second run raises instead of resealing. These deals are never used for tuning
or exploration; only a confirm run loads them.

The stored card order is the test set; the seeds below only record how it was drawn.
"""

import random

from stacker.cards import CONFIGS, build_deck
from stacker.sealed import config_name, write_sealed

DEALS_PER_CONFIG = 50_000

# A string seed per config. Dev deals use integer seeds ≥ 10, and random.Random hashes a str seed
# through sha512 instead, so no dev seed can replay a sealed set. One seed per config keeps the six
# sets independent of each other.
SEED_PREFIX = "stacker-sealed-v2:"


def main():
    for num_players in sorted(CONFIGS):
        name = config_name(num_players)
        rng = random.Random(SEED_PREFIX + name)
        deck_orders = []
        for _ in range(DEALS_PER_CONFIG):
            deck_order = build_deck(CONFIGS[num_players].num_decks)
            rng.shuffle(deck_order)
            deck_orders.append(deck_order)
        digest = write_sealed(num_players, deck_orders)
        print(f"{name}  {len(deck_orders)} deals  sha256 {digest}")


if __name__ == "__main__":
    main()
