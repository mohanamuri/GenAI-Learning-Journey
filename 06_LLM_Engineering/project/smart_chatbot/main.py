"""
Smart Chatbot — CLI
--------------------
Demonstrates all Module 06 concepts in one project:
  - System prompt with persona
  - Multi-turn chat with sliding window history
  - Streaming responses
  - Per-session cost tracking
  - Clean exit with session summary

Run: python main.py
Run with local LLM: python main.py --local
"""

import argparse
from chatbot import SmartChatbot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Use Ollama local LLM")
    parser.add_argument("--model", default=None, help="Override model name")
    args = parser.parse_args()

    bot = SmartChatbot(use_local=args.local, model_override=args.model)

    print(bot.welcome_message())
    print("Type 'quit' or 'exit' to end. Type 'history' to see context. Type 'cost' for usage.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            break

        if user_input.lower() == "history":
            bot.print_history()
            continue

        if user_input.lower() == "cost":
            bot.print_cost_summary()
            continue

        print("Bot: ", end="", flush=True)
        bot.chat(user_input)
        print()

    print("\n" + "=" * 50)
    bot.print_cost_summary()
    print("Goodbye!")


if __name__ == "__main__":
    main()
