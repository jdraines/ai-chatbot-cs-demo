"""View layer: CLI read-eval-print loop."""

from dotenv import load_dotenv

from .agent.controller import AgentController

load_dotenv()

BANNER = "Thoughtful AI Support  |  type 'quit' to exit  |  ensure OPENAI_API_KEY is set for full functionality"
DIVIDER = "-" * len(BANNER)


def main() -> None:
    print(DIVIDER)
    print(BANNER)
    print(DIVIDER)

    controller = AgentController()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        if not user_input:
            continue

        response = controller.handle(user_input)
        print(f"\nAgent: {response}")
