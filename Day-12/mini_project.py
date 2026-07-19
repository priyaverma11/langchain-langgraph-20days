# mini_project.py — Day 12
# Personal Research Agent with full ReAct loop

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from dotenv import load_dotenv
import math

load_dotenv()

model = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

# ── Tools ───────────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """
    Evaluate any math expression.
    Use for ALL calculations. Always use this before
    passing results to other tools.
    Examples: "1000 * 0.08", "sqrt(225)", "2 ** 8"
    """
    try:
        allowed = {'sqrt': math.sqrt, 'pi': math.pi,
                   'abs': abs, 'round': round}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert between units. Value must be a plain number.
    Supports: km/miles, celsius/fahrenheit, kg/pounds,
    meters/feet, liters/gallons.
    """
    conversions = {
        ('km', 'miles'): lambda x: x * 0.621371,
        ('miles', 'km'): lambda x: x * 1.60934,
        ('celsius', 'fahrenheit'): lambda x: (x * 9/5) + 32,
        ('fahrenheit', 'celsius'): lambda x: (x - 32) * 5/9,
        ('kg', 'pounds'): lambda x: x * 2.20462,
        ('pounds', 'kg'): lambda x: x * 0.453592,
        ('meters', 'feet'): lambda x: x * 3.28084,
        ('feet', 'meters'): lambda x: x * 0.3048,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    return f"Unsupported: {from_unit} → {to_unit}"


@tool
def percentage_tool(part: float, whole: float) -> str:
    """
    Calculate what percentage part is of whole.
    Both must be plain numbers — use calculator first
    if you need to compute either value.
    """
    try:
        part = float(part)
        whole = float(whole)
        if whole == 0:
            return "Error: whole cannot be zero"
        pct = (part / whole) * 100
        return f"{part} is {pct:.2f}% of {whole}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def bmi_calculator(weight_kg: float, height_m: float) -> str:
    """
    Calculate BMI (Body Mass Index).
    weight_kg: weight in kilograms (plain number)
    height_m: height in meters (plain number)
    Returns BMI value and category.
    """
    try:
        bmi = weight_kg / (height_m ** 2)
        if bmi < 18.5:
            category = "Underweight"
        elif bmi < 25:
            category = "Normal weight"
        elif bmi < 30:
            category = "Overweight"
        else:
            category = "Obese"
        return f"BMI: {bmi:.1f} ({category})"
    except Exception as e:
        return f"Error: {str(e)}"


tools = [calculator, unit_converter, percentage_tool, bmi_calculator]
tool_map = {t.name: t for t in tools}
model_with_tools = model.bind_tools(tools)

SYSTEM_PROMPT = """You are a smart personal research assistant.

Rules:
1. Use calculator FIRST for any math — never guess numbers
2. Pass only plain numbers to tools — never expressions
3. For multi-step problems, build on each result
4. Show a clear, organized final answer
5. If a tool errors, try a different approach"""


def run_agent(question: str) -> str:
    """Run the full ReAct agent loop."""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question)
    ]

    print(f"\n{'='*55}")
    print(f"❓ {question}")
    print(f"{'='*55}")

    max_iterations = 8
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        response = model_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            print(f"\n✅ Answer: {response.content}")
            return response.content

        print(f"\n🔄 Step {iteration}:")
        for tool_call in response.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]
            print(f"  🔧 {name}({args})")

            try:
                result = tool_map[name].invoke(args)
            except Exception as e:
                result = f"Error: {str(e)}"

            print(f"  📤 {result}")
            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            ))

    return "Reached max iterations"


# ── Test the Agent ──────────────────────────────────────────

print("🧠 Personal Research Agent")
print("=" * 55)

scenarios = [
    # Finance
    "I earn $75,000 per year. I want to save 20% monthly. "
    "How much is that per month, and if I save for 3 years "
    "what total will I have (no interest)?",

    # Health
    "I weigh 75 kg and I am 1.75 meters tall. "
    "What is my BMI and what category am I in? "
    "Also, what would my weight in pounds be?",

    # Travel
    "I need to drive 250 miles. My car gets 35 miles per gallon. "
    "How many gallons do I need, and how many liters is that?",

    # Complex multi-step
    "I scored 67 out of 85 on my first test and "
    "91 out of 110 on my second test. "
    "What percentage did I score on each? "
    "What is my combined percentage across both tests?"
]

for scenario in scenarios:
    run_agent(scenario)
