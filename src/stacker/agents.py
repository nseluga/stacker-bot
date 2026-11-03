"""
Scripted agents for the ladder: random → greedy → heuristic (architecture plan §3 rule 3).

The agent contract, used by game.play_out: an agent is a function
    agent(observation, actions, rng) -> one of actions
observation: a game.Observation, only what this seat may see; agents never get the HandState.
actions: the legal actions from game.legal_actions, in their fixed order (PASS last when allowed).
rng: a random.Random. An agent draws randomness only from it, so a seed replays the hand.
"""

from stacker.game import PASS
from stacker.patterns import BOMB


def random_agent(observation, actions, rng):
    """The bottom rung: picks uniformly among the legal actions. Ignores the observation."""
    return rng.choice(actions)


def greedy_key(play):
    """
    greedy's ordering of plays, lowest first: (bomb or not, top card, card count, the cards).
    play: a moves.Play, never PASS. Used by greedy_agent and by the heuristic's tie-breaks.
    """
    return (play.pattern.kind == BOMB, play.pattern.top_card, len(play.cards), play.cards)


def greedy_agent(observation, actions, rng):
    """
    The second rung, "lowest legal" (plan §3 rule 3; decision log 2026-10-24 and 2026-10-31):
    plays the lowest play it has and passes only when PASS is its only action. Ignores rng and the
    rest of the observation, so it plays a given position the same way every time.

    Lowest means the smallest key, compared in order:
    1. bomb or not: every normal play comes before any bomb, so it never spends a bomb when a
       normal play beats the table;
    2. top card: Card compares by rank, then suit, then deck, the same order beats uses;
    3. card count: fewer cards first;
    4. the cards themselves, lowest first: a tie-break, so of 3s-4s-5h and 3c-4s-5h it plays the
       lower cards and keeps 3c. It also makes the choice independent of the order of actions.
    Leading, the lowest single always wins on keys 2 and 3, so greedy leads its lowest single.
    """
    best_play = None
    best_key = None
    for action in actions:
        if action == PASS:
            continue
        key = greedy_key(action)
        if best_key is None or key < best_key:
            best_play = action
            best_key = key

    if best_play is None:
        return PASS
    return best_play
