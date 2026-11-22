"""
Stage 5 (exploration arms) scoring: the rule that labels an arm against the control band
(stage plan specialist_explore, "Rule"; decision log 2026-11-22, "the rule by which a result
counts").

- end_scores reads one run's per-deal results (the _deals.csv evaluate_checkpoints.py writes) and
  gives each deal its paired difference averaged over the run's last checkpoints.
- arm_verdict compares an arm's end scores with the 3 control seeds' on the same deals and labels
  the arm candidate, loses or tie. A winner is a candidate whose second seed is also a candidate;
  the caller makes that call from two verdicts.

The label is exploration only: it is not a claim, and Nate picks winners from the graphs.
"""

import csv

from stacker.evaluate import bootstrap_ci

# The labels arm_verdict gives.
CANDIDATE = "candidate"
LOSES = "loses"
TIE = "tie"


def end_scores(deals_csv, field, checkpoints):
    """
    Each deal's paired difference against one field, averaged over the named checkpoints.
    deals_csv: path to a <run>_eval_..._deals.csv, rows (checkpoint, field, deal_index,
      difference).
    field: the field column to read, e.g. "heuristic".
    checkpoints: the updates-done numbers to average, e.g. the last 3 at or before arm length.
      Passed in, not taken from the file: the 300M control is read at arm length, so its last
      3 checkpoints in the file are not the ones that count.
    Returns a list of floats, one per deal, in deal_index order.
    Raises ValueError if a named checkpoint is missing, or if the checkpoints do not all hold the
    same deals.
    """
    # difference_by_checkpoint[checkpoint][deal_index] = difference, for the field asked for.
    difference_by_checkpoint = {}
    for checkpoint in checkpoints:
        difference_by_checkpoint[checkpoint] = {}
    with open(deals_csv) as file:
        for row in csv.DictReader(file):
            checkpoint = int(row["checkpoint"])
            if row["field"] == field and checkpoint in difference_by_checkpoint:
                deal_index = int(row["deal_index"])
                difference_by_checkpoint[checkpoint][deal_index] = float(row["difference"])

    # Every named checkpoint must have scored the same deals, or a deal's mean would mix
    # different numbers of checkpoints.
    first_deals = None
    for checkpoint in checkpoints:
        deals = set(difference_by_checkpoint[checkpoint])
        if not deals:
            raise ValueError(f"{deals_csv}: no {field!r} rows for checkpoint {checkpoint}")
        if first_deals is None:
            first_deals = deals
        elif deals != first_deals:
            raise ValueError(f"{deals_csv}: checkpoint {checkpoint} holds different deals")

    # Per deal, the mean over checkpoints: deal d's end score.
    scores = []
    for deal_index in sorted(first_deals):
        total = 0.0
        for checkpoint in checkpoints:
            total += difference_by_checkpoint[checkpoint][deal_index]
        scores.append(total / len(checkpoints))
    return scores


def arm_verdict(arm_scores, control_scores, rng):
    """
    Labels an arm against the control band (the stage plan's rule).
    arm_scores: list of floats, one per deal, from end_scores.
    control_scores: list of the control seeds' end_scores lists (3 seeds), on the same deals.
    rng: a random.Random for the bootstrap; the caller names it from the seed, so a label
      replays.
    Returns (label, mean, ci_low, ci_high): label is CANDIDATE, LOSES or TIE; the other three are
      the mean and 95% bootstrap interval over deals of [arm - mean of the control seeds].
      Positive means the arm is stronger.
    Raises ValueError if the lists are not all the same length (not the same deals).

    candidate: interval low > 0, and the arm's end score is above every control seed's.
    loses: the mirror: interval high < 0, and the arm's end score is below every control seed's.
    tie: anything else; the control setting stays.
    The interval removes deal luck (every net plays the same deals). It does not see training
    luck, so the end score must also clear every single control seed (decision log 2026-11-22).
    """
    num_deals = len(arm_scores)
    for seed_scores in control_scores:
        if len(seed_scores) != num_deals:
            raise ValueError("arm and control seeds were not scored on the same deals")

    # Per deal: the arm minus the mean of the control seeds on that same deal.
    differences = []
    for deal_index in range(num_deals):
        control_total = 0.0
        for seed_scores in control_scores:
            control_total += seed_scores[deal_index]
        control_mean = control_total / len(control_scores)
        differences.append(arm_scores[deal_index] - control_mean)
    mean, ci_low, ci_high = bootstrap_ci(differences, rng)

    # A run's end score: its mean over deals.
    arm_end = sum(arm_scores) / num_deals
    control_ends = []
    for seed_scores in control_scores:
        control_ends.append(sum(seed_scores) / num_deals)

    if ci_low > 0 and arm_end > max(control_ends):
        label = CANDIDATE
    elif ci_high < 0 and arm_end < min(control_ends):
        label = LOSES
    else:
        label = TIE
    return label, mean, ci_low, ci_high
