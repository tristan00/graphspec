import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REVIEWS = os.path.join(ROOT, "review")
INSTRUCTIONS = """Turn end refused: no adversarial review verdict exists for this session.
Do this before stopping:
1. Restate the user's request for this turn and every requirement in it, plus the project rules in Agents.md.
2. Spawn a fresh general-purpose Agent as an adversarial reviewer. Give it the request verbatim, the list of files changed (git status and diff) and every output produced, and instruct it to find every way the work falls short: scope narrowed or items deferred without the user's say-so, claims not verified by running things, tests not run or failing, project rules broken, questions asked instead of work done. It must write JSON {"verdict": "PASS" or "FAIL", "findings": [...], "blocked_on_user": bool, "blocked_reason": str} to %s
3. On FAIL, fix every finding, then run the review again with a fresh agent.
4. Stop only after a PASS verdict. Stopping while blocked on something only the user can give is allowed only with blocked_on_user true and the reason stated to the user in the final message."""


def main():
    data = json.load(sys.stdin)
    session = data.get("session_id", "unknown")
    path = os.path.join(REVIEWS, session + ".json")
    if not os.path.exists(path):
        print(json.dumps({"decision": "block", "reason": INSTRUCTIONS % path}))
        return
    with open(path, encoding="utf-8") as stream:
        verdict = json.load(stream)
    os.remove(path)
    if verdict.get("verdict") == "PASS":
        print(json.dumps({"systemMessage": "Adversarial review passed."}))
        return
    if verdict.get("blocked_on_user"):
        print(
            json.dumps(
                {
                    "systemMessage": "Adversarial review FAIL, stop allowed because blocked on user: "
                    + str(verdict.get("blocked_reason", ""))
                }
            )
        )
        return
    findings = "\n".join("- " + str(item) for item in verdict.get("findings", []))
    print(
        json.dumps(
            {
                "decision": "block",
                "reason": "Adversarial review FAILED. Fix every finding, then review again and write a new verdict to %s\n%s"
                % (path, findings),
            }
        )
    )


if __name__ == "__main__":
    main()
