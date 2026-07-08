# mini_project.py — Day 6
# Personal Assistant that remembers across turns
# Uses the modern LangChain memory approach

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-haiku-4-5", temperature=0.7)

# Store for all sessions
store = {}

def get_or_create_session(session_id: str) -> InMemoryChatMessageHistory:
    """Get existing session or create new one."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def chat(user_input: str, session_id: str) -> str:
    """
    Send a message and get a response.
    History is managed automatically per session.
    """
    # Get this session's history
    history = get_or_create_session(session_id)

    # Build prompt with history
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a helpful personal assistant with excellent memory. "
         "Always remember what the user tells you about themselves."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = prompt | model

    # Add user message to history
    history.add_user_message(user_input)

    # Run the chain
    result = chain.invoke({
        "history": history.messages[:-1],  # all except latest
        "input": user_input
    })

    # Add AI response to history
    history.add_ai_message(result.content)

    return result.content

def show_history(session_id: str):
    """Show conversation history for a session."""
    if session_id not in store:
        print("No history found for this session.")
        return
    history = store[session_id]
    print(f"\n--- History for {session_id} ---")
    for i, msg in enumerate(history.messages, 1):
        role = "You" if msg.type == "human" else "Assistant"
        print(f"{i}. {role}: {msg.content[:80]}...")
    print(f"Total messages: {len(history.messages)}\n")

def main():
    print("🤖 Personal Assistant with Memory")
    print("Commands: !history | !clear | !switch <name> | quit")
    print("=" * 50)

    session_id = "default-session"
    print(f"Session: {session_id}\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        elif user_input.lower() == "quit":
            print("Goodbye!")
            break

        elif user_input.lower() == "!history":
            show_history(session_id)

        elif user_input.lower() == "!clear":
            store[session_id] = InMemoryChatMessageHistory()
            print("✅ History cleared!\n")

        elif user_input.lower().startswith("!switch "):
            session_id = user_input.split(" ")[1]
            print(f"✅ Switched to session: {session_id}\n")

        else:
            try:
                response = chat(user_input, session_id)
                msgs = len(store.get(session_id, 
                       InMemoryChatMessageHistory()).messages)
                print(f"\n🤖 Assistant: {response}")
                print(f"[Messages in history: {msgs}]\n")
            except Exception as e:
                print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    main()
