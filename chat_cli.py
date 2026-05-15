"""
Interactive CLI to chat with the SHL Assessment Recommender.
Run: python chat_cli.py
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"


def main():
    print("=" * 60)
    print("  SHL Assessment Recommender — Interactive Chat")
    print("=" * 60)
    print("Type your message and press Enter. Type 'quit' to exit.\n")

    # Check server is running
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        resp.raise_for_status()
        print("[Server is running]\n")
    except Exception:
        print("[ERROR] Server not running. Start it first with: python main.py")
        sys.exit(1)

    messages = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        try:
            resp = requests.post(
                f"{BASE_URL}/chat",
                json={"messages": messages},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[ERROR] {e}\n")
            messages.pop()  # Remove failed message
            continue

        # Show reply
        print(f"\nBot: {data['reply']}")

        # Show recommendations if any
        if data["recommendations"]:
            print(f"\n  Recommendations ({len(data['recommendations'])}):")
            for i, rec in enumerate(data["recommendations"], 1):
                print(f"  {i}. {rec['name']} [{rec['test_type']}]")
                print(f"     {rec['url']}")

        if data["end_of_conversation"]:
            print("\n[Conversation complete]")

        print()

        # Add assistant reply to history
        messages.append({"role": "assistant", "content": data["reply"]})


if __name__ == "__main__":
    main()
