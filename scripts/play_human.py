"""
Nate plays logged 4p/1d dev hands against greedy seats, to tune the heuristic to how he plays
(stage plan baselines, piece 3; decision log 2026-10-31 "Heuristic weights set from Nate's play").

Every screen is built from the Observation alone (the view of the game an agent gets at its
turn), so Nate sees exactly what the heuristic will see: his own hand, the table, card counts,
passes, finishers and the history. Nothing hidden.

Each finished hand is appended to data/human-play/hands.jsonl as one line per decision: the
hand's seed, the decision's number, the chosen cards and an optional reason. The greedy seats are
deterministic, so seed + choices replay the whole hand (scratch/check_choose_follow.py does).

Run from the repo root:
    PYTHONPATH=src .venv/bin/python scripts/play_human.py
"""

import json
import random
from pathlib import Path

from stacker.agents import greedy_agent
from stacker.cards import parse_cards
from stacker.game import PASS, new_hand, play_out

# The table Nate plays at: 4 players, 1 deck, Nate in seat 0, greedy in every other seat.
NUM_PLAYERS = 4
NATE_SEAT = 0

# The log: one JSON line per decision, finished hands only. Tracked: it is the evidence the
# heuristic's rules and weights rest on (manifest Paths).
LOG_PATH = Path("data/human-play/hands.jsonl")

# Hand seeds count up from here: clear of the test seeds (0-2) and evaluate_agents' dev draws,
# and each hand gets a seed no earlier hand used.
FIRST_SEED = 1000

MOVE_PROMPT = "> "
REASON_PROMPT = "why? (enter to skip, undo to re-pick) "

# How many of the latest actions the screen shows. One full lap at 4 players: enough to see how
# the current table play came about. The whole history is one keystroke away.
RECENT_ACTIONS = 4


def cards_text(cards):
    """Cards as the screen prints them, lowest first: '9♦ 9♥'."""
    names = []
    for card in sorted(cards):
        names.append(str(card))
    return " ".join(names)


def action_text(action):
    """A play's cards, or 'pass'."""
    if action == PASS:
        return "pass"
    return cards_text(action.cards)


def show_position(observation):
    """
    The screen for one decision, as a string, built only from the Observation.
    Lines: what is on the table (or that you lead), each seat's cards left with passed / out marks,
    the latest actions, and your hand. Seats are 'you' and s1, s2, ... by seat number.
    """
    you = observation.seat

    # Label a seat for the screen: "you" for the player, "s<number>" for the others.
    def seat_name(seat):
        if seat == you:
            return "you"
        return f"s{seat}"

    lines = []

    # Line 1: the table. Empty means this player leads; on the first lead the opening card
    # (the game's required starting card) must be part of the play.
    if observation.table is None:
        if observation.opening_card is not None:
            lines.append(
                f"table: empty, you lead; your play must include {observation.opening_card}"
            )
        else:
            lines.append("table: empty, you lead")
    else:
        lines.append(
            f"table: {seat_name(observation.table_seat)} played "
            f"{cards_text(observation.table.cards)}"
        )

    # One entry per seat: cards left, then why it can't act this round, if it can't.
    seat_entries = []
    for seat in range(len(observation.cards_left)):
        entry = f"{seat_name(seat)} {observation.cards_left[seat]}"
        if seat in observation.finish_order:
            entry += f" (out, #{observation.finish_order.index(seat) + 1})"
        elif observation.passed[seat]:
            entry += " (passed)"
        seat_entries.append(entry)
    lines.append("seats: " + " · ".join(seat_entries))

    # The last few actions, oldest first, so the table play can be traced back.
    recent = []
    for seat, action in observation.history[-RECENT_ACTIONS:]:
        recent.append(f"{seat_name(seat)} {action_text(action)}")
    if len(recent) > 0:
        lines.append("recent: " + " · ".join(recent))

    # Last line: the cards Nate can choose from.
    lines.append(f"your hand: {cards_text(observation.hand)}")
    return "\n".join(lines)


def read_choice(text, actions):
    """
    Turns what Nate typed into one of the legal actions.
    text: 'p' for pass, or the cards of a play in any order and case ('jh JD').
    actions: the legal actions at this turn (PASS is in it only when passing is allowed).
    Returns the matching action. Raises ValueError with a message for the screen when the text
    can't be read or names no legal action, so a mistyped or illegal play never reaches the log.
    """
    text = text.strip()
    if text == "":
        raise ValueError("type the cards to play, or p to pass")

    if text.lower() == "p":
        if PASS in actions:
            return PASS
        raise ValueError("you lead this round, so you can't pass")

    # A Play keeps its cards lowest first, so sorting the typed cards makes typing order
    # irrelevant. Then the typed play is matched against the legal actions, never built directly.
    typed = tuple(sorted(parse_cards(text)))
    for action in actions:
        if action != PASS and action.cards == typed:
            return action
    raise ValueError(f"{cards_text(typed)} is not a legal play here (? lists them)")


def play_hand(seed, ask=input):
    """
    Plays one 4p/1d dev hand with Nate in seat 0 and greedy elsewhere, through play_out, so the
    hand runs on exactly the engine loop the evaluations use.
    seed: deals the hand (random.Random(seed)); greedy ignores its rng, so seed + Nate's choices
    replay the hand. ask: reads one line of input; input by default, a script in the checks.
    Returns (decisions, points): one dict per Nate decision, in order, ready for the log, and the
    hand's points per seat.

    At each of Nate's turns: the other seats' actions since his last turn are printed, then the
    screen. He types a play or p; ? lists the legal actions, h prints the whole history. After a
    legal pick he gets the why? prompt: Enter skips it, a line is kept as the reason, and undo goes
    back to the pick before anything is played.
    """
    # decisions collects the log rows; printed_up_to is how much history Nate has already seen.
    decisions = []
    printed_up_to = 0

    # Agent for seat 0 with the same signature as greedy_agent, but the choice comes from the
    # keyboard. rng is unused: a human needs no randomness.
    def nate_agent(observation, actions, rng):
        nonlocal printed_up_to

        # Show what the other seats did since Nate's last turn, then the full screen.
        for seat, action in observation.history[printed_up_to:]:
            print(f"  s{seat}: {action_text(action)}")
        print()
        print(show_position(observation))

        # Ask until a legal play or pass is picked and its reason is given. "?" and "h" are
        # help commands, a bad pick reprints the error, and "undo" at the reason re-picks.
        while True:
            text = ask(MOVE_PROMPT).strip()
            if text == "?":
                options = []
                for action in actions:
                    options.append(action_text(action))
                print("legal: " + " · ".join(options))
                continue
            if text.lower() == "h":
                for seat, action in observation.history:
                    print(f"  s{seat}: {action_text(action)}")
                continue
            try:
                choice = read_choice(text, actions)
            except ValueError as error:
                print(error)
                continue
            reason = ask(REASON_PROMPT).strip()
            if reason.lower() == "undo":
                continue
            break

        # Record the decision. Cards are stored as [rank, suit, deck] so the log rebuilds them
        # exactly; choice_text is only for a human reading the file.
        if choice == PASS:
            stored_choice = PASS
        else:
            stored_choice = []
            for card in choice.cards:
                stored_choice.append(list(card))
        decisions.append({
            "seed": seed,
            "decision": len(decisions),
            "choice": stored_choice,
            "choice_text": action_text(choice),
            "reason": reason,
        })
        # Nate's own action is history[len(history)] once played; the next print starts after it.
        printed_up_to = len(observation.history) + 1
        return choice

    # Seat the players: Nate at NATE_SEAT, greedy everywhere else.
    agents = []
    for seat in range(NUM_PLAYERS):
        if seat == NATE_SEAT:
            agents.append(nate_agent)
        else:
            agents.append(greedy_agent)
    # Run the whole hand: play_out asks each agent in turn, returns all actions and the scores.
    trace, points = play_out(
        new_hand(NUM_PLAYERS, random.Random(seed)), agents, random.Random(seed)
    )

    # Show the actions after Nate's last turn (he is out or the hand ended), then the result.
    for seat, action in trace[printed_up_to:]:
        print(f"  s{seat}: {action_text(action)}")
    print(f"\nhand {seed} over · points: you {points[NATE_SEAT]}, others {points[1:]}")
    return decisions, points


def main():
    """
    Plays hands until Nate stops, appending each finished hand to LOG_PATH.
    The next seed is FIRST_SEED + the number of hands already logged, so a new sitting carries on
    where the last one stopped. A hand is written only once it ends: Ctrl-C (or Ctrl-D) mid-hand
    drops that hand, so the log never holds a partial one.
    """
    # Find which seeds are already logged so this sitting continues from the next unused one.
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logged_seeds = set()
    if LOG_PATH.exists():
        for line in LOG_PATH.read_text().splitlines():
            logged_seeds.add(json.loads(line)["seed"])
    seed = FIRST_SEED + len(logged_seeds)

    print(
        f"{len(logged_seeds)} hands logged so far. "
        "Type cards (jd jh) or p; ? lists legal actions, h shows history."
    )
    # One loop pass per hand. The log is written only after play_hand returns, so an
    # interrupted hand leaves no partial rows.
    while True:
        print(f"\n=== hand {seed} ===")
        try:
            decisions, _points = play_hand(seed)
        except (KeyboardInterrupt, EOFError):
            print(f"\nhand {seed} dropped, not logged. {len(logged_seeds)} hands in the log.")
            return

        # Append the finished hand's decisions as one JSON object per line.
        with LOG_PATH.open("a") as log:
            for decision in decisions:
                log.write(json.dumps(decision) + "\n")
        logged_seeds.add(seed)
        print(f"logged {len(decisions)} decisions to {LOG_PATH} ({len(logged_seeds)} hands)")

        # Offer another hand; Ctrl-C or Ctrl-D here quits cleanly (this hand is already saved).
        try:
            again = input("another hand? (enter = yes, q = quit) ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return
        if again == "q":
            return
        seed += 1


if __name__ == "__main__":
    main()
