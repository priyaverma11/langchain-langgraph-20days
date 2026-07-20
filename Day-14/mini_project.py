# mini_project.py — Day 14
# Content Moderation Pipeline with LangGraph

from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

# ── State Definition ────────────────────────────────────────
class ModerationState(TypedDict):
    content: str          # original content
    content_type: str     # article, comment, review, etc.
    toxicity_score: str   # low, medium, high
    category: str         # spam, hate, appropriate, etc.
    action: str           # approve, flag, reject
    reason: str           # why this decision was made
    final_output: str     # formatted final result


# ── Nodes ───────────────────────────────────────────────────

def detect_content_type(state: ModerationState) -> dict:
    """Node 1: Identify what type of content this is."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Classify content type. "
                   "Return ONLY one word: "
                   "article, comment, review, or spam"),
        ("human", "{content}")
    ])
    chain = prompt | model | StrOutputParser()
    content_type = chain.invoke(
        {"content": state["content"]}
    ).strip().lower()

    print(f"  [detect_type] → {content_type}")
    return {"content_type": content_type}


def score_toxicity(state: ModerationState) -> dict:
    """Node 2: Score how toxic the content is."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Rate toxicity of this content. "
                   "Return ONLY one word: low, medium, or high"),
        ("human", "{content}")
    ])
    chain = prompt | model | StrOutputParser()
    score = chain.invoke(
        {"content": state["content"]}
    ).strip().lower()

    print(f"  [score_toxicity] → {score}")
    return {"toxicity_score": score}


def categorize_content(state: ModerationState) -> dict:
    """Node 3: Categorize the specific issue."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Categorize this content. "
                   "Return ONLY one word: "
                   "appropriate, spam, offensive, or harmful"),
        ("human", "{content}")
    ])
    chain = prompt | model | StrOutputParser()
    category = chain.invoke(
        {"content": state["content"]}
    ).strip().lower()

    print(f"  [categorize] → {category}")
    return {"category": category}


def approve_content(state: ModerationState) -> dict:
    """Node 4a: Approve clean content."""
    print(f"  [approve] ✅ Content approved")
    return {
        "action": "APPROVED",
        "reason": "Content meets community guidelines"
    }


def flag_content(state: ModerationState) -> dict:
    """Node 4b: Flag content for human review."""
    print(f"  [flag] ⚠️ Content flagged for review")
    return {
        "action": "FLAGGED",
        "reason": f"Detected {state['category']} content - needs human review"
    }


def reject_content(state: ModerationState) -> dict:
    """Node 4c: Reject harmful content."""
    print(f"  [reject] ❌ Content rejected")
    return {
        "action": "REJECTED",
        "reason": f"High toxicity {state['category']} content violates guidelines"
    }


def generate_report(state: ModerationState) -> dict:
    """Node 5: Generate final moderation report."""
    emoji = {"APPROVED": "✅", "FLAGGED": "⚠️", "REJECTED": "❌"}
    action = state["action"]

    report = (
        f"\n{'='*45}\n"
        f"MODERATION REPORT {emoji.get(action, '?')}\n"
        f"{'='*45}\n"
        f"Content:   {state['content'][:50]}...\n"
        f"Type:      {state['content_type']}\n"
        f"Toxicity:  {state['toxicity_score']}\n"
        f"Category:  {state['category']}\n"
        f"Action:    {action}\n"
        f"Reason:    {state['reason']}\n"
        f"{'='*45}"
    )
    return {"final_output": report}


# ── Routing Function ────────────────────────────────────────

def decide_action(state: ModerationState) -> str:
    """Decide what to do based on toxicity and category."""
    toxicity = state["toxicity_score"]
    category = state["category"]

    if toxicity == "high" or category == "harmful":
        return "reject"
    elif toxicity == "medium" or category in ["spam", "offensive"]:
        return "flag"
    else:
        return "approve"


# ── Build Graph ─────────────────────────────────────────────

graph = StateGraph(ModerationState)

# Add all nodes
graph.add_node("detect_type", detect_content_type)
graph.add_node("score_toxicity", score_toxicity)
graph.add_node("categorize", categorize_content)
graph.add_node("approve", approve_content)
graph.add_node("flag", flag_content)
graph.add_node("reject", reject_content)
graph.add_node("report", generate_report)

# Linear edges for analysis phase
graph.add_edge(START, "detect_type")
graph.add_edge("detect_type", "score_toxicity")
graph.add_edge("score_toxicity", "categorize")

# Conditional routing after categorization
graph.add_conditional_edges(
    "categorize",
    decide_action,
    {
        "approve": "approve",
        "flag": "flag",
        "reject": "reject",
    }
)

# All paths lead to report
graph.add_edge("approve", "report")
graph.add_edge("flag", "report")
graph.add_edge("reject", "report")
graph.add_edge("report", END)

# Compile
app = graph.compile()


# ── Test Cases ──────────────────────────────────────────────

test_contents = [
    "I absolutely love this product! Best purchase I've made.",
    "BUY NOW!!! CLICK HERE!!! LIMITED TIME OFFER!!!",
    "This movie was disappointing but had some good moments.",
]

print("🛡️ Content Moderation Pipeline")
print("=" * 45)

for content in test_contents:
    print(f"\nProcessing: '{content[:45]}...'")
    result = app.invoke({
        "content": content,
        "content_type": "",
        "toxicity_score": "",
        "category": "",
        "action": "",
        "reason": "",
        "final_output": ""
    })
    print(result["final_output"])
