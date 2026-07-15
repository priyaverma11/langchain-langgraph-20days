# ============================================================
# File: day09_rag.py
# Day 9 — RAG Part 1: Document Loading & Chunking
# ============================================================

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

# ── Step 1: Create a sample document ───────────────────────
# We'll create a text file about Python to use as our knowledge base

sample_text = """
Python Programming Guide

Chapter 1: Introduction to Python
Python is a high-level, interpreted programming language created by 
Guido van Rossum in 1991. It emphasizes code readability and simplicity.
Python uses indentation to define code blocks, making it unique among
major programming languages. It supports multiple programming paradigms
including procedural, object-oriented, and functional programming.

Chapter 2: Python Data Types
Python has several built-in data types. Integers are whole numbers like
1, 2, 3. Floats are decimal numbers like 3.14. Strings are text enclosed
in quotes. Lists are ordered collections that can be modified. Tuples are
ordered collections that cannot be modified. Dictionaries store key-value
pairs. Sets store unique values without order.

Chapter 3: Python Functions
Functions in Python are defined using the def keyword. They can accept
parameters and return values using the return statement. Python supports
default parameter values, *args for variable arguments, and **kwargs for
keyword arguments. Lambda functions are anonymous one-line functions.
Decorators modify function behavior without changing the function itself.

Chapter 4: Python Libraries
Python has thousands of libraries. NumPy provides numerical computing.
Pandas is used for data manipulation and analysis. Matplotlib creates
visualizations and charts. Scikit-learn provides machine learning tools.
Django and Flask are web frameworks. Requests handles HTTP requests.
LangChain helps build AI applications with language models.

Chapter 5: Error Handling
Python uses try-except blocks for error handling. The try block contains
code that might raise an exception. The except block handles the error.
Finally block runs regardless of whether an error occurred. You can raise
custom exceptions using the raise statement. Common exceptions include
ValueError, TypeError, KeyError, and IndexError.
"""

# Save to a text file
with open("python_guide.txt", "w") as f:
    f.write(sample_text)
print("✅ Created python_guide.txt")


# ── Step 2: Load the document ───────────────────────────────
print("\n=== 1. Document Loading ===")

loader = TextLoader("python_guide.txt")
documents = loader.load()

print(f"Documents loaded: {len(documents)}")
print(f"Total characters: {len(documents[0].page_content)}")
print(f"Source: {documents[0].metadata}")


# ── Step 3: Split into chunks ───────────────────────────────
print("\n=== 2. Document Chunking ===")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,      # each chunk ~300 characters
    chunk_overlap=50,    # 50 character overlap between chunks
    length_function=len, # measure by character count
)

chunks = splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")
print(f"\nFirst 3 chunks:")
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Content: {chunk.page_content[:150]}...")
    print(f"Length: {len(chunk.page_content)} chars")


# ── Step 4: Create embeddings and vector store ──────────────
print("\n=== 3. Building Vector Store ===")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(chunks, embeddings)
print(f"✅ Vector store built with {len(chunks)} chunks!")


# ── Step 5: Search the document ─────────────────────────────
print("\n=== 4. Searching the Document ===")

queries = [
    "What data types does Python have?",
    "How do you handle errors in Python?",
    "What libraries are used for data analysis?",
]

for query in queries:
    print(f"\nQuestion: {query}")
    results = vector_store.similarity_search(query, k=2)
    print("Relevant chunks found:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc.page_content[:100]}...")


# ── Step 6: Full RAG Pipeline ───────────────────────────────
print("\n=== 5. Full RAG Pipeline ===")

model = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

def answer_question(question: str) -> str:
    """
    Full RAG pipeline:
    1. Search for relevant chunks
    2. Use chunks as context
    3. Generate answer
    """
    # Step 1: Retrieve relevant chunks
    relevant_chunks = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

    # Step 2: Generate answer using context
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Answer questions using "
         "ONLY the provided context. If the answer isn't in "
         "the context, say 'I don't have that information.'"),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    chain = prompt | model | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


# Test the full pipeline
test_questions = [
    "Who created Python and when?",
    "What is the difference between lists and tuples?",
    "What is LangChain used for?",
    "What is the capital of France?",  # Not in our document!
]

for question in test_questions:
    print(f"\nQ: {question}")
    answer = answer_question(question)
    print(f"A: {answer}")
    print("-" * 40)
