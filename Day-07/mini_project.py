# mini_project.py — Day 7
# Customer Support Router
# Combines routing + parallel analysis + custom logic

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

# ── Step 1: Classifier ──────────────────────────────────────
classifier_prompt = ChatPromptTemplate.from_messages([
    ("system", """Classify this message into ONE category.
Return ONLY one word: billing, technical, or general."""),
    ("human", "{input}")
])
classifier = classifier_prompt | model | StrOutputParser()

# ── Step 2: Specialist Chains ───────────────────────────────
billing_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a billing specialist. Be empathetic and solution-focused."),
        ("human", "{input}")
    ]) | model | StrOutputParser()
)

technical_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a technical engineer. Give step-by-step troubleshooting."),
        ("human", "{input}")
    ]) | model | StrOutputParser()
)

general_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a friendly customer service agent. Be warm and helpful."),
        ("human", "{input}")
    ]) | model | StrOutputParser()
)

# ── Step 3: Priority Classifier ─────────────────────────────
def get_priority(message: str) -> str:
    """Determine priority based on keywords."""
    message_lower = message.lower()
    if any(word in message_lower for word in
           ["urgent", "immediately", "critical", "broken", "can't access"]):
        return "🔴 HIGH"
    elif any(word in message_lower for word in
             ["wrong", "issue", "problem", "not working"]):
        return "🟡 MEDIUM"
    else:
        return "🟢 LOW"

# ── Step 4: Main Router ─────────────────────────────────────
def handle_message(message: str):
    """Route message to correct department and analyze."""
    print(f"\n{'='*50}")
    print(f"Customer: {message}")

    # Get category
    category = classifier.invoke({"input": message}).strip().lower()
    priority = get_priority(message)

    print(f"Category: {category.upper()}")
    print(f"Priority: {priority}")

    # Route to correct chain
    if "billing" in category:
        response = billing_chain.invoke({"input": message})
    elif "technical" in category:
        response = technical_chain.invoke({"input": message})
    else:
        response = general_chain.invoke({"input": message})

    print(f"\nResponse: {response[:200]}...")
    print(f"{'='*50}")

# ── Test Messages ───────────────────────────────────────────
test_messages = [
    "I was charged twice this month, this is urgent!",
    "The app keeps crashing when I try to login",
    "What are your refund policies?",
    "My account is completely broken, I can't access anything!",
    "Do you offer student discounts?"
]

print("🎧 Customer Support Router")
for message in test_messages:
    handle_message(message)
