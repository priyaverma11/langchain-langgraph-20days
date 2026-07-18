# mini_project.py — Day 11
# Research Assistant with Multiple Tools

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv
import math

load_dotenv()

model = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

# ── Tools ───────────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """
    Evaluate any mathematical expression.
    Use for ALL math: arithmetic, percentages, square roots.
    Examples: "100 * 0.15", "sqrt(144)", "2 ** 10"
    """
    try:
        allowed = {'sqrt': math.sqrt, 'pi': math.pi, 'abs': abs}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert between units: km/miles, celsius/fahrenheit,
    kg/pounds, liters/gallons.
    Example: value=5, from_unit='km', to_unit='miles'
    """
    conversions = {
        ('km', 'miles'): lambda x: x * 0.621371,
        ('miles', 'km'): lambda x: x * 1.60934,
        ('celsius', 'fahrenheit'): lambda x: (x * 9/5) + 32,
        ('fahrenheit', 'celsius'): lambda x: (x - 32) * 5/9,
        ('kg', 'pounds'): lambda x: x * 2.20462,
        ('pounds', 'kg'): lambda x: x * 0.453592,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    return f"Conversion {from_unit}→{to_unit} not supported"


@tool
def word_counter(text: str) -> str:
    """
    Count words, characters and sentences in any text.
    Use when asked about text length or statistics.
    """
    words = len(text.split())
    chars = len(text)
    sentences = text.count('.') + text.count('!') + text.count('?')
    return f"Words: {words} | Characters: {chars} | Sentences: {sentences}"


@tool
def percentage_calculator(part: float, whole: float) -> str:
    """
    Calculate what percentage 'part' is of 'whole'.
    Also calculates the remaining percentage.
    Example: part=25, whole=200 → 12.5%
    """
    if whole == 0:
        return "Error: Cannot divide by zero"
    pct = (part / whole) * 100
    remaining = 100 - pct
    return f"{part} is {pct:.2f}% of {whole} (remaining: {remaining:.2f}%)"


tools = [calculator, unit_converter, word_counter, percentage_calculator]

# ── Tool Processing ─────────────────────────────────────────

tool_map = {t.name: t for t in tools}
model_with_tools = model.bind_tools(tools)

def run_assistant(user_message: str) -> str:
    """Run the research assistant with full tool support."""
    messages = [HumanMessage(content=user_message)]
    max_iterations = 5  # prevent infinite loops
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        response = model_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            # No more tools needed — return final answer
            return response.content

        # Process all tool calls
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"  🔧 Tool: {tool_name}")
            print(f"  📥 Args: {tool_args}")

            if tool_name in tool_map:
                result = tool_map[tool_name].invoke(tool_args)
            else:
                result = f"Tool '{tool_name}' not found"

            print(f"  📤 Result: {result}")

            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            ))

    return "Max iterations reached"


# ── Testing──────────────────────────────────────

print("🔬 Research Assistant with Tools")
print("=" * 50)

questions = [
    "If I have 45 out of 180 marks, what percentage did I score?",
    "Convert my weight of 68 kg to pounds",
    "What is the compound interest formula result for 1000 * (1 + 0.05) ** 3?",
    "How many words are in 'To be or not to be that is the question'?",
    "What is 20% of 1500 and then add 75 to it?",
]

for question in questions:
    print(f"\n❓ {question}")
    answer = run_assistant(question)
    print(f"✅ {answer}")
    print("-" * 40)
