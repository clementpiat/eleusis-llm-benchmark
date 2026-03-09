"""Human Eleusis evaluation"""

import hashlib, json, random, time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from eleusis.game import GameEngine, GameState, GuessRuleAction, PlayCardAction, Rule, RuleValidator
from eleusis.game.cards import Card, Suit
from eleusis.llm import create_client_from_config

load_dotenv()

ROOT = Path(__file__).parent.parent
RESULTS_FILE = ROOT / "results/260121_78_rounds/solo_evaluation_human/results.json"
PLAYER = "Human"
SUITS_ORDER = [Suit.HEARTS, Suit.DIAMONDS, Suit.SPADES, Suit.CLUBS]
ALPHABET = "abcdefghijkl"


def sort_hand(state: GameState) -> list[Card]:
    return sorted(
        state.player.hand.get_all_cards(),
        key=lambda card: (SUITS_ORDER.index(card.suit), card.rank),
    )


def print_state(
    state: GameState, turn_number: int, max_turns: int, failed_guesses: list[str]
) -> None:
    hand = sort_hand(state)
    print(f"\n—- Turn {turn_number}/{max_turns}")
    print(f"Board: {state.to_compact_string()}\n")

    index = 0
    for suit in SUITS_ORDER:
        suit_cards = [card for card in hand if card.suit == suit]
        entries = "  ".join(
            f"({ALPHABET[index + i]}) {str(card)}" for i, card in enumerate(suit_cards)
        )
        index += len(suit_cards)
        print(entries)

    if failed_guesses:
        print("Failed guesses: " + ", ".join(failed_guesses))


def get_action(state: GameState) -> tuple[str, Card | None, str]:
    """Ask the player to pick a card or guess the rule."""
    hand = sort_hand(state)
    while True:
        input_text = input("\n[a-l] play card or [guess] the rule > ").strip().lower()
        if input_text == "guess":
            rule_text = input("Describe the rule: ").strip()
            future_card_text = (
                input("[a-l] play future card in case your guess is wrong > ").strip().lower()
            )
            if rule_text and future_card_text in ALPHABET and len(future_card_text) == 1:
                index = ALPHABET.index(future_card_text)
                return "guess", hand[index], rule_text

        elif input_text in ALPHABET and len(input_text) == 1:
            index = ALPHABET.index(input_text)
            return "play", hand[index], ""


def play_round(engine: GameEngine, state: GameState, config: dict) -> dict:
    """Run one interactive round."""
    max_turns = config["game"]["max_turns"]
    start_time = time.time()
    turn_index, failed_guesses, turns_data = 0, [], []
    first_correct_turn = None
    while turn_index < max_turns and not engine.is_game_over():
        state.turn_number = turn_index + 1
        print_state(state, turn_index + 1, max_turns, failed_guesses)
        action, card, guess_text = get_action(state)

        turn_data = {
            "turn_number": turn_index + 1,
            "llm_response": {
                "guess_rule": action == "guess",
                "confidence_level": None,
                "tentative_rule": guess_text or None,
            },
            "action_result": {"accepted": None},
            "guess_attempt": None,
            "tokens": {"output_tokens": 0},
        }

        if action == "play":
            result = engine.play_turn(PlayCardAction(card))
        elif action == "guess":
            engine.play_turn(PlayCardAction(card))
            print("Evaluating rule...")
            result = engine.play_turn(GuessRuleAction(guess_text))
            complexity = result.get("complexity_metrics") or ()
            turn_data["guess_attempt"] = {
                "correct": result["correct"],
                "node_count": complexity.get("node_count"),
                "cyclomatic_complexity": complexity.get("cyclomatic"),
            }
            if result["correct"]:
                print("CORRECT ✅")
                turns_data.append(turn_data)
                first_correct_turn = turn_index + 1
                break

            print("INCORRECT ❌")
            failed_guesses.append(guess_text)
        else:
            continue

        turns_data.append(turn_data)
        turn_index += 1

    score = engine.calculate_score(max_turns, turn_index)
    print(f"Score: {max(0, score)} ({'found' if engine.rule_guessed else 'not_found'})")

    return {
        "turn_count": turn_index,
        "success": engine.rule_guessed,
        "score": score,
        "floored_score": max(0, score),
        "no_stakes_score": (max_turns - first_correct_turn + 1)
        if first_correct_turn is not None
        else 0,
        "first_correct_turn": first_correct_turn,
        "failed_guesses": engine.failed_guess_count,
        "game_over_reason": "correct_guess" if engine.rule_guessed else "max_turns",
        "llm_usage": {
            "rule_compiler": {},
            "player": {"output_tokens": 0, "reasoning_tokens": 0, "answer_tokens": 0},
        },
        "turns": turns_data,
        "wall_clock_seconds": round(time.time() - start_time, 2),
    }


def setup_engine(
    rule: Rule, config: dict, compiler: Any, round_number: int
) -> tuple[GameEngine, GameState]:
    """Create a GameEngine and GameState for a rule."""
    state = GameState(PLAYER)
    engine = GameEngine(
        state,
        rule,
        rule_compiler_client=compiler,
        rule_validator=RuleValidator(),
        hand_size=config["game"]["hand_size"],
        wrong_guess_penalty=config["game"]["wrong_guess_penalty"],
        num_simulations=config["rule_compiler"]["num_simulations"],
        turns_per_simulation=config["rule_compiler"]["turns_per_simulation"],
        simulation_seed=config["rule_compiler"]["simulation_seed"],
        compiler_max_retries=config["rule_compiler"]["max_retries"],
    )
    base_seed = config["game"].get("seed")
    rule_hash = int(hashlib.md5(rule.get_code().encode()).hexdigest(), 16) & 0xFFFFFFFF
    engine.setup_game(
        (base_seed + rule_hash + (round_number - 1)) & 0xFFFFFFFF if base_seed else None
    )
    return engine, state


def main():
    config = yaml.safe_load(open(ROOT / "config.yaml"))
    all_rules = json.load(open(ROOT / config["rules"]["library_path"]))["rules"]
    all_rules = [(rule, round_number) for rule in all_rules for round_number in range(3)]
    random.seed(0)
    random.shuffle(all_rules)

    compiler = create_client_from_config(config["rule_compiler"], max_tokens=10_000)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_FILE.exists():
        results = json.load(open(RESULTS_FILE))
        all_rules = all_rules[len(results["rounds"]) :]
    else:
        results = {
            "config": {
                "player": PLAYER,
                "player_model": PLAYER,
                "max_turns": config["game"]["max_turns"],
                "wrong_guess_penalty": config["game"]["wrong_guess_penalty"],
                "num_rounds_per_rule": 1,
            },
            "rounds": [],
        }

    for rule_data, round_number in all_rules:
        rule = Rule(rule_data["description"], rule_data["code"])
        engine, state = setup_engine(rule, config, compiler, round_number)

        round_result = play_round(engine, state, config)
        results["rounds"].append(
            {
                "round_number": round_number,
                "rule_name": rule_data["name"],
                "rule_description": rule_data["description"],
                "rule_code": rule_data["code"],
                **round_result,
            }
        )
        json.dump(results, open(RESULTS_FILE, "w"), indent=2)

        if input("\nNext rule? [Enter q to stop] ").strip().lower() == "q":
            break


if __name__ == "__main__":
    main()
