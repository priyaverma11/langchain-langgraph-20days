from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

class BookReview(BaseModel):
    title: str = Field(description="the book title")
    sentiment: str = Field(description="positive, negative, or neutral")
    score: int = Field(description="score from 1 to 10")
    key_themes: List[str] = Field(default=[], description="main themes of the book")
    would_recommend: bool = Field(description="True if worth reading, False if not")

structured_model = model.with_structured_output(BookReview)

reviews = [
    "The Great Gatsby is a masterpiece! Fitzgerald's prose is beautiful and the themes of the American Dream are timeless. 10/10 would read again.",
    "I found Twilight boring and poorly written. The characters are flat and the plot is predictable. Not recommended.",
    "1984 by Orwell is thought-provoking and dark. It makes you think about society and freedom. Difficult read but worth it."
]

print("📚 Book Review Analyzer")
print("=" * 50)

for review in reviews:
    try:
        result = structured_model.invoke(review)
        print(f"\nTitle:     {result.title}")
        print(f"Sentiment: {result.sentiment}")
        print(f"Score:     {result.score}/10")
        print(f"Themes:    {', '.join(result.key_themes)}")
        print(f"Recommend: {result.would_recommend}")
        print("-" * 30)
    except Exception as e:
        print(f"Error: {e}")
