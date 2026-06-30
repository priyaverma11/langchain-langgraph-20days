# mini_project.py
# Day 1 Mini Project: Enhanced Python Tutor Chatbot

import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

# ── Constants ──────────────────────────────────────
SYSTEM_PROMPT = """You are my_tutor, a friendly and thorough Python tutor.

Teaching style:
- Always explain the concept first before showing any code
- Give exactly 2-3 examples per explanation
- Always include a code example for every concept
- Always include a real-world analogy to make it relatable
- Keep explanations clear and beginner-friendly
- After explaining, ask if the student understood"""

# ── State ──────────────────────────────────────────
chat_history = []
turn_number = 0
total_tokens = 0

# ── Helper Functions ───────────────────────────────

def reset_chat():
    global chat_history, turn_number, total_tokens
    chat_history = []
    turn_number = 0
    total_tokens = 0
    print("✅ Chat reset! Starting fresh.")

def show_history():
    print("\n--- Conversation History ---")
    for message in chat_history:
        role = message["role"]
        content = message["content"]
        print(f"{role}: {content}")
    print("----------------------------")

def show_stats():
    print("\n--- Stats ---")
    print(f"Total turns  : {turn_number}")
    print(f"Total tokens : {total_tokens}")
    print("-------------")

def chat(user_message: str) -> str:
    global total_tokens

    # Add user message to history
    chat_history.append({
        "role": "user",
        "content": user_message
    })

    # Call API with full history
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=chat_history
    )

    # Extract reply
    assistant_reply = response.content[0].text

    # Add reply to history
    chat_history.append({
        "role": "assistant",
        "content": assistant_reply
    })

    # Update total tokens
    total_tokens += response.usage.input_tokens + response.usage.output_tokens

    return assistant_reply

# ── Main Loop ──────────────────────────────────────

def main():
    global turn_number

    print("🐍 my_tutor is ready!")
    print("Commands: !reset | !history | !stats | quit")
    print("=" * 50)

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        elif user_input.lower() == "quit":
            print("my_tutor: Bye! Keep coding! 🐍")
            break

        elif user_input == "!reset":
            reset_chat()

        elif user_input == "!history":
            show_history()

        elif user_input == "!stats":
            show_stats()

        else:
            turn_number += 1
            print(f"\n[Turn {turn_number}]")
            try:
                response = chat(user_input)
                print(f"[Tokens this turn: {total_tokens}]")
                print(f"\n🐍 my_tutor: {response}")
            except anthropic.APIConnectionError:
                print("❌ Cannot connect. Check internet.")
            except anthropic.AuthenticationError:
                print("❌ Invalid API key.")
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
