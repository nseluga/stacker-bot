"""
Scripted agents for the ladder: random → greedy → heuristic (architecture plan §3 rule 3).

The agent contract, used by game.play_out: an agent is a function
    agent(observation, actions, rng) -> one of actions
observation: a game.Observation, only what this seat may see; agents never get the HandState.
actions: the legal actions from game.legal_actions, in their fixed order (PASS last when allowed).
rng: a random.Random. An agent draws randomness only from it, so a seed replays the hand.
"""


def random_agent(observation, actions, rng):
    """The bottom rung: picks uniformly among the legal actions. Ignores the observation."""
    return rng.choice(actions)
