"""
The M3 control Q-net (architecture plan §6; spec in docs/stage-plans/specialist_explore.md).

The net scores every legal action of a decision: it reads what the seat sees (state), the recent
moves (history) and one candidate action, and predicts the hand's final score if that action is
played. The pieces, in build order:

- The batch: collate_decisions stacks several decisions into rectangular tensors. Decisions have
  different numbers of legal actions, so the action lists are padded with zero rows at the back
  and a mask marks the real rows.
- The net's parts: QNet.__init__ creates an LSTM that will summarize the history and an MLP that
  will score one (history summary, state, action) vector. Sizes follow DouZero's control net
  (decision log, 2026-11-09).
- The scores: QNet.summarize_history turns a decision's recent moves into one vector, and
  QNet.forward joins it with the state and each action and returns one Q per action.
- The loss: q_loss compares the Q of the action that was played with the score the hand really
  gave (mean squared error; decision log, 2026-11-12).
"""

import numpy as np
import torch
from torch import nn

from stacker.encode import ACTION_DIM, MOVE_DIM, STATE_DIM

# The control net's three sizes (decision log, 2026-11-09). They are the defaults of QNet, so
# QNet() is the control; the exploration arms pass other values (spec part 16).
# Size of the history summary z, the LSTM's output after the latest move.
DEFAULT_HISTORY_HIDDEN = 128
# Width of every hidden MLP layer, and how many of them there are before the output layer.
DEFAULT_MLP_WIDTH = 512
DEFAULT_HIDDEN_LAYERS = 5


def collate_decisions(decisions):
    """
    Stacks B decisions into one batch, padding each action list to the longest one.
    decisions: list of B tuples (state, history, mask, action_matrix) from encode_decision, all
      encoded with the same history window H. Decision b has K_b actions.
    Returns (state, history, history_mask, actions, action_mask), all torch tensors:
      state float32 [B, STATE_DIM]; history float32 [B, H, MOVE_DIM]; history_mask bool [B, H];
      actions float32 [B, K, ACTION_DIM] with K = the largest K_b, zero rows after a decision's
      real actions; action_mask bool [B, K], True for real actions.
    Row k of actions[b] is still action k of decision b, so an index into the padded row indexes
    the same legal action.
    """
    states = []
    histories = []
    history_masks = []
    max_actions = 0
    for state, history, mask, action_matrix in decisions:
        states.append(state)
        histories.append(history)
        history_masks.append(mask)
        max_actions = max(max_actions, len(action_matrix))

    # Actions start as all zeros and all "not real"; each decision's real rows fill the front.
    batch_size = len(decisions)
    actions = np.zeros((batch_size, max_actions, ACTION_DIM), dtype=np.float32)
    action_mask = np.zeros((batch_size, max_actions), dtype=bool)
    for index, decision in enumerate(decisions):
        action_matrix = decision[3]
        num_actions = len(action_matrix)
        actions[index, :num_actions] = action_matrix
        action_mask[index, :num_actions] = True

    return (
        torch.from_numpy(np.stack(states)),
        torch.from_numpy(np.stack(histories)),
        torch.from_numpy(np.stack(history_masks)),
        torch.from_numpy(actions),
        torch.from_numpy(action_mask),
    )


class QNet(nn.Module):
    """
    Scores each candidate action of a decision with one Q value: the predicted final score of
    the hand if that action is played. The same weights read every config.
    """

    def __init__(
        self,
        width=DEFAULT_MLP_WIDTH,
        hidden_layers=DEFAULT_HIDDEN_LAYERS,
        history_hidden=DEFAULT_HISTORY_HIDDEN,
    ):
        """
        Creates the two parts and their parameters; nothing is computed here.
        width: units in every hidden MLP layer (W in the spec).
        hidden_layers: number of (Linear, ReLU) layers before the output layer (L).
        history_hidden: size of the history summary z (D).
        The defaults are the control net, 2,323,457 parameters.
        history_lstm: reads history rows [B, H, MOVE_DIM] in order, gives [B, H, history_hidden].
        mlp: reads [..., history_hidden + STATE_DIM + ACTION_DIM], gives [..., 1]: hidden_layers
          layers of (Linear, ReLU), width wide, then one Linear to a single number with no
          activation, so Q can be any real value.
        """
        super().__init__()
        # The parts are created in a fixed order: the LSTM, then the MLP front to back. Each new
        # layer draws its starting weights from torch's one random stream, so a different order
        # would give the control different starting weights under the same seed.
        # batch_first: the batch is the first axis, [B, H, MOVE_DIM], as collate_decisions gives.
        self.history_lstm = nn.LSTM(MOVE_DIM, history_hidden, batch_first=True)

        # The MLP reads [history summary ; state ; action] joined end to end. Only the first
        # layer reads the joined input; every later layer reads the one before it.
        layers = []
        layer_input_dim = history_hidden + STATE_DIM + ACTION_DIM
        for _ in range(hidden_layers):
            layers.append(nn.Linear(layer_input_dim, width))
            layers.append(nn.ReLU())
            layer_input_dim = width
        layers.append(nn.Linear(width, 1))
        self.mlp = nn.Sequential(*layers)

    def summarize_history(self, history, history_mask):
        """
        Summarizes each decision's recent moves as one vector z (spec part 2, equations 1-2).
        history: float32 [B, H, MOVE_DIM], oldest move first, padded at the front.
        history_mask: bool [B, H], True for real moves.
        Returns z, float32 [B, history_hidden].
        """
        # Equation 1: a padded row is multiplied by 0 and a real row by 1, so whatever a padded
        # row holds cannot reach the LSTM. The new last axis lets [B, H] scale each row's values.
        masked_history = history * history_mask.unsqueeze(-1)

        # Equation 2: the LSTM reads the rows oldest to latest, updating its memory at each one.
        # No initial state is passed, so the memory starts at zero. It steps over padded rows too,
        # so z depends on how many there are, never on their content.
        step_outputs, _ = self.history_lstm(masked_history)  # [B, H, history_hidden]

        # Padding is at the front, so the last step is always the latest real move.
        return step_outputs[:, -1, :]

    def forward(self, state, history, history_mask, actions, action_mask):
        """
        Scores every candidate action of every decision (spec part 3, equations 3-4).
        state: float32 [B, STATE_DIM]. history: float32 [B, H, MOVE_DIM].
        history_mask: bool [B, H], True for real moves.
        actions: float32 [B, K, ACTION_DIM], padded at the back. action_mask: bool [B, K], True
          for real actions.
        Returns Q, float32 [B, K]. Q[b, k] is the predicted final score if decision b plays its
        action k. A padded slot holds exactly 0, which is not a score: code that picks the best
        action must apply action_mask itself.
        """
        history_summary = self.summarize_history(history, history_mask)  # [B, history_hidden]

        # A decision has one history summary and one state, but the MLP needs them beside each
        # action. The new middle axis is stretched to K; expand reads the same memory K times
        # and copies nothing.
        num_actions = actions.shape[1]
        summary_per_action = history_summary.unsqueeze(1).expand(-1, num_actions, -1)
        state_per_action = state.unsqueeze(1).expand(-1, num_actions, -1)

        # Equation 3: join end to end, in the order the first MLP layer expects.
        joined = torch.cat([summary_per_action, state_per_action, actions], dim=-1)

        # Equation 4: Linear layers work on the last axis, so one call scores all B x K rows.
        q_values = self.mlp(joined).squeeze(-1)  # [B, K, 1] -> [B, K]
        return q_values * action_mask


def q_loss(q_values, taken_index, returns):
    """
    Mean squared error between the taken action's Q and the hand's real return (spec part 4,
    equation 5).
    q_values: float32 [B, K] from QNet.forward.
    taken_index: int64 [B], the column of the action each decision played. It is always a real
      action, never a padded slot, so no action mask is needed here.
    returns: float32 [B], the hand's final score for the seat that decided (0, 1 or 2).
    Returns a scalar tensor: the squared error, averaged over the B decisions.
    """
    # gather reads one column per row: row b gives q_values[b, taken_index[b]]. It needs the
    # index to have as many axes as q_values, so [B] becomes [B, 1] and the result goes back
    # to [B].
    taken_q = q_values.gather(1, taken_index.unsqueeze(1)).squeeze(1)

    # Only the taken Q enters the loss, so the other K - 1 values of a row get no gradient from
    # this decision. mse_loss subtracts, squares and takes the mean over the batch.
    return nn.functional.mse_loss(taken_q, returns)
