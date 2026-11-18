"""
Learning curve of one training run: each saved checkpoint played against fixed field agents on
dev deals (stage plan specialist_explore, stage 4 piece 8; spec part 14, equation 15).

Exploration only (rule 2): dev seeds ≥ 10. There is no path to the sealed deals here.
"""

import torch

from stacker.dmc import choose_action
from stacker.encode import encode_decision
from stacker.net import QNet

# Evaluation plays the net's best action on every turn: no random moves (equation 15).
EVAL_EPSILON = 0.0


def load_checkpoint_net(path):
    """
    Builds the net a checkpoint file holds, ready to play.
    path: a checkpoint_<updates done>.pt written by stacker.dmc.train (a QNet state_dict).
    Returns a QNet on the CPU, in eval mode.
    """
    net = QNet()
    # map_location: a checkpoint is saved from the CPU copy, but loading must not depend on it.
    # weights_only: the file holds tensors only, so torch refuses to run any code stored in it.
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    # Eval mode turns off anything that behaves differently in training (dropout, batch
    # statistics). QNet has none today; this keeps evaluation right if one is added.
    net.eval()
    return net


def checkpoint_agent(net, observation, actions, rng):
    """
    Plays the action the net scores highest (equation 15).
    net: a QNet from load_checkpoint_net. The caller binds it with functools.partial, since
    play_out calls an agent as agent(observation, actions, rng).
    observation, actions, rng: as play_out passes them. Returns one of actions.
    A forced turn (one legal action) returns it without calling the net; equal scores go to the
    lowest index; no gradients are recorded. All three are choose_action's own behaviour.
    """
    decision = encode_decision(observation, actions)
    # With epsilon 0 choose_action still draws one number from rng on a turn with a choice, and
    # then never takes the random branch. The draw is the same on every replay of a deal.
    index = choose_action(net, decision, EVAL_EPSILON, rng)
    return actions[index]
