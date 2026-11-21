"""Grade a free-text diagnose answer: the last `ANSWER:` line against a list of accepted names.

Standard library only: harbor export copies this file into a task's tests/ as grade.py and runs it in the agent's
container after the agent exits (shared verifier), so grading needs no image build or installs. Targets import
normalized_answer from here, so in-repo scoring and the task grader cannot drift apart.

python3 grade.py --answer /app/out/answer.txt --expected /tests/expected.json --reward /logs/verifier/reward.json
"""

import argparse
import json
import re
from pathlib import Path

ENV_ID_DECORATIONS = re.compile(r"^ale/|(noframeskip|deterministic)?-v\d+$", re.IGNORECASE)


def normalized_answer(raw_answer: str) -> str:
    """'Ms. Pac-Man', 'MsPacmanNoFrameskip-v4' and 'ALE/MsPacman-v5' all become 'mspacman'."""
    undecorated = ENV_ID_DECORATIONS.sub("", raw_answer.strip())
    return re.sub(r"[^a-z0-9]", "", undecorated.lower())


def last_answer_line(text: str) -> str:
    answers = [line.removeprefix("ANSWER:").strip() for line in text.splitlines() if line.startswith("ANSWER:")]
    return answers[-1] if answers else ""


def is_accepted(given: str, accepted: list[str]) -> bool:
    normalized = normalized_answer(given)
    return bool(normalized) and normalized in {normalized_answer(name) for name in accepted}


def grade(*, answer_path: Path, accepted: list[str]) -> dict[str, float]:
    """A missing file or answer line scores 0 with valid_output 0, never an error."""
    text = answer_path.read_text(errors="replace") if answer_path.is_file() else ""
    given = last_answer_line(text)
    print(f"answer line: {given!r}")
    return {"reward": float(is_accepted(given, accepted)), "valid_output": float(bool(given))}


def main() -> None:
    parser = argparse.ArgumentParser(prog="grade")
    parser.add_argument("--answer", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--reward", type=Path, required=True)
    args = parser.parse_args()
    reward = grade(answer_path=args.answer, accepted=json.loads(args.expected.read_text())["accepted"])
    args.reward.parent.mkdir(parents=True, exist_ok=True)
    args.reward.write_text(json.dumps(reward, indent=2))
    print(json.dumps(reward))


if __name__ == "__main__":
    main()
