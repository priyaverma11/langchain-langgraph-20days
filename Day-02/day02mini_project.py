# mini_project.py — Day 2
# Prompt Template Library

# mini_project.py — Day 2
# Prompt Template Library

import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()


def run_prompt(system: str, user: str, temperature: float = 0) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=500,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}]
    )
    return response.content[0].text


# ── Template 1: Job Extractor ───────────────────────
JOB_SYSTEM = """You are a job description parser.
Extract information and return ONLY a JSON object.
No explanation. No extra text. No code fences. Just the JSON.

Keys: job_title, company_name, salary (null if missing), location (null if missing)"""

JOB_USER = """Extract from this job description:

{text}"""

job_description = "We're hiring a Senior Data Analyst at Acme Corp. \
The role pays $95,000-$115,000 annually and is based in Dallas, TX."


# ── Template 2: Sentiment Analyzer ─────────────────
SENTIMENT_SYSTEM = """You are a sentiment classifier.
Return ONLY one word: positive, negative, or neutral."""

SENTIMENT_USER = "Classify this review: {review}"

reviews = [
    "This product is absolutely amazing, I love it!",
    "Terrible experience, would not recommend.",
    "It arrived on time and works as expected."
]


# ── Template 3: Summarizer ──────────────────────────
SUMMARIZER_SYSTEM = """You are a text summarizer.
Summarize in exactly {num_sentences} sentence(s). Be concise."""

SUMMARIZER_USER = """Summarize this text:

{text}"""

sample_text = """Python is a high-level programming language known for
its simplicity and readability. It was created by Guido van Rossum
and first released in 1991."""


# ── Template 4: Few-Shot Email Classifier ───────────
FEW_SHOT_SYSTEM = """You are an email classifier.
Classify emails into exactly one category: billing, technical, or general.
Return ONLY the category word. No explanation."""

FEW_SHOT_USER = """Here are examples:

Email: "Why was I overcharged this month?"
Category: billing

Email: "Why is my phone so hot and won't unlock?"
Category: technical

Email: "How long does this product typically last?"
Category: general

Now classify this email:
Email: "{email}"
Category:"""

test_emails = [
    "Why was I charged twice this month?",
    "The app keeps freezing on my phone",
    "What are your refund policies?"
]


# ── Template 5: Chain-of-Thought Problem Solver ─────
COT_SYSTEM = """You are a math problem solver.
Break down every problem into clear numbered steps.
Show your calculation at each step.
Always end with a clearly labeled final answer.

Format your response exactly like this:
Step 1: [description and calculation]
Step 2: [description and calculation]
...
Answer: [final number]"""

COT_USER = """Solve this problem:

{problem}"""

problems = [
    "I have 24 chocolates. I give 1/3 to my sister and eat 4 myself. How many are left?",
    "A train travels 60mph for 2.5 hours. How far does it travel?"
]


# ── Run Everything ──────────────────────────────────

if __name__ == "__main__":
    print("🚀 Prompt Template Library")
    print("=" * 50)

    print("\n=== 1. Job Extractor ===")
    result = run_prompt(JOB_SYSTEM, JOB_USER.format(text=job_description))
    print(result)

    print("\n=== 2. Sentiment Analyzer ===")
    for review in reviews:
        sentiment = run_prompt(SENTIMENT_SYSTEM, SENTIMENT_USER.format(review=review))
        print(f"Review: {review[:40]}... → {sentiment}")

    print("\n=== 3. Summarizer ===")
    system = SUMMARIZER_SYSTEM.format(num_sentences=2)
    summary = run_prompt(system, SUMMARIZER_USER.format(text=sample_text))
    print(summary)

    print("\n=== 4. Email Classifier (Few-Shot) ===")
    for email in test_emails:
        prompt = FEW_SHOT_USER.format(email=email)
        category = run_prompt(FEW_SHOT_SYSTEM, prompt)
        print(f"Email:    {email}")
        print(f"Category: {category}\n")

    print("\n=== 5. Chain-of-Thought Solver ===")
    for problem in problems:
        prompt = COT_USER.format(problem=problem)
        solution = run_prompt(COT_SYSTEM, prompt)
        print(f"Problem:  {problem}")
        print(f"Solution:\n{solution}")
        print("-" * 50)
