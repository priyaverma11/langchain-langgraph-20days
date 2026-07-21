# mini_project.py — Day 15
# Job Application Triage System with LangGraph

from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

# ── State ───────────────────────────────────────────────────
class ApplicationState(TypedDict):
    applicant_name: str
    resume_text: str
    experience_years: int
    key_skills: str
    education_level: str
    qualification_status: str
    interview_type: str
    feedback: str
    final_decision: str
    review_notes: Annotated[list, add_messages]


# ── Nodes ───────────────────────────────────────────────────

def extract_experience(state: ApplicationState) -> dict:
    """Extract years of experience from resume."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract years of experience from this resume. "
                   "Return ONLY a number like: 3"),
        ("human", "{resume}")
    ])
    chain = prompt | model | StrOutputParser()
    result = chain.invoke({"resume": state["resume_text"]})

    try:
        years = int(''.join(filter(str.isdigit, result)))
    except:
        years = 0

    note = "Experience extracted: " + str(years) + " years"
    print("  [extract_experience] " + note)
    return {
        "experience_years": years,
        "review_notes": [AIMessage(content=note)]
    }


def extract_skills(state: ApplicationState) -> dict:
    """Extract key technical skills."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "List the top 5 technical skills from this resume. "
                   "Return as comma-separated list only."),
        ("human", "{resume}")
    ])
    chain = prompt | model | StrOutputParser()
    skills = chain.invoke({"resume": state["resume_text"]})

    note = "Skills found: " + skills
    print("  [extract_skills] " + skills[:50] + "...")
    return {
        "key_skills": skills,
        "review_notes": [AIMessage(content=note)]
    }


def check_education(state: ApplicationState) -> dict:
    """Check education level."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "What is the highest education level in this resume? "
                   "Return ONLY one: high_school, bachelors, masters, phd"),
        ("human", "{resume}")
    ])
    chain = prompt | model | StrOutputParser()
    education = chain.invoke(
        {"resume": state["resume_text"]}
    ).strip().lower()

    note = "Education: " + education
    print("  [check_education] " + education)
    return {
        "education_level": education,
        "review_notes": [AIMessage(content=note)]
    }


def qualify_applicant(state: ApplicationState) -> dict:
    """Decide if applicant is qualified for the role."""
    years = state["experience_years"]
    education = state["education_level"]

    if years < 2:
        status = "underqualified"
        reason = "Only " + str(years) + " years experience (need 2+)"
    elif years > 8:
        status = "overqualified"
        reason = str(years) + " years experience may be overqualified"
    elif education == "high_school":
        status = "underqualified"
        reason = "Bachelor's degree required"
    else:
        status = "qualified"
        reason = str(years) + " years experience with " + education + " degree"

    note = "Qualification: " + status + " - " + reason
    print("  [qualify] " + status)
    return {
        "qualification_status": status,
        "feedback": reason,
        "review_notes": [AIMessage(content=note)]
    }


def schedule_phone_screen(state: ApplicationState) -> dict:
    """Schedule phone screening for qualified applicants."""
    print("  [phone_screen] Scheduling phone interview")
    note = "Action: Phone screen scheduled"
    name = state["applicant_name"]
    feedback = state["feedback"]
    decision = "ADVANCING: " + name + " selected for phone screen. Reason: " + feedback
    return {
        "interview_type": "Phone Screen",
        "final_decision": decision,
        "review_notes": [AIMessage(content=note)]
    }


def schedule_technical(state: ApplicationState) -> dict:
    """Schedule technical interview for strong candidates."""
    print("  [technical] Scheduling technical interview")
    note = "Action: Technical interview scheduled"
    name = state["applicant_name"]
    feedback = state["feedback"]
    decision = "FAST TRACK: " + name + " to technical interview! Profile: " + feedback
    return {
        "interview_type": "Technical Interview",
        "final_decision": decision,
        "review_notes": [AIMessage(content=note)]
    }


def send_rejection(state: ApplicationState) -> dict:
    """Send rejection for unqualified applicants."""
    print("  [reject] Sending rejection")
    note = "Action: Rejection sent"
    name = state["applicant_name"]
    feedback = state["feedback"]
    decision = "NOT ADVANCING: " + name + " - " + feedback
    return {
        "interview_type": "None",
        "final_decision": decision,
        "review_notes": [AIMessage(content=note)]
    }


def generate_summary(state: ApplicationState) -> dict:
    """Generate final application summary."""
    notes_text = "\n".join([
        msg.content for msg in state["review_notes"]
    ])

    name = state["applicant_name"]
    years = str(state["experience_years"])
    education = state["education_level"]
    skills = state["key_skills"][:60]
    status = state["qualification_status"].upper()
    interview = state["interview_type"]
    decision = state["final_decision"]

    summary = (
        "\n" + "="*50 + "\n" +
        "APPLICATION REVIEW: " + name + "\n" +
        "="*50 + "\n" +
        "Experience:  " + years + " years\n" +
        "Education:   " + education + "\n" +
        "Skills:      " + skills + "...\n" +
        "Status:      " + status + "\n" +
        "Interview:   " + interview + "\n" +
        "-"*50 + "\n" +
        "DECISION: " + decision + "\n" +
        "-"*50 + "\n" +
        "REVIEW NOTES:\n" + notes_text + "\n" +
        "="*50
    )
    return {"final_decision": summary}


# ── Routing Function ─────────────────────────────────────────

def route_by_qualification(state: ApplicationState) -> str:
    """Route based on qualification status and experience."""
    status = state["qualification_status"]
    years = state["experience_years"]

    if status == "underqualified" or status == "overqualified":
        return "reject"
    elif years >= 4:
        return "technical"
    else:
        return "phone"


# ── Build Graph ──────────────────────────────────────────────

graph = StateGraph(ApplicationState)

graph.add_node("experience", extract_experience)
graph.add_node("skills", extract_skills)
graph.add_node("education", check_education)
graph.add_node("qualify", qualify_applicant)
graph.add_node("phone", schedule_phone_screen)
graph.add_node("technical", schedule_technical)
graph.add_node("reject", send_rejection)
graph.add_node("summary", generate_summary)

graph.add_edge(START, "experience")
graph.add_edge("experience", "skills")
graph.add_edge("skills", "education")
graph.add_edge("education", "qualify")

graph.add_conditional_edges(
    "qualify",
    route_by_qualification,
    {
        "phone": "phone",
        "technical": "technical",
        "reject": "reject",
    }
)

graph.add_edge("phone", "summary")
graph.add_edge("technical", "summary")
graph.add_edge("reject", "summary")
graph.add_edge("summary", END)

app = graph.compile()


# ── Test Applications ────────────────────────────────────────

applicants = [
    {
        "name": "Sarah Chen",
        "resume": """
        Sarah Chen - Senior Data Analyst
        Experience: 5 years in data analytics and business intelligence
        Education: Masters in Data Science, Stanford University
        Skills: Python, SQL, Power BI, Tableau, Machine Learning,
                LangChain, dbt, Snowflake
        Previous roles: Data Analyst at Google (3 years),
                        Senior Analyst at Meta (2 years)
        """
    },
    {
        "name": "John Smith",
        "resume": """
        John Smith - Recent Graduate
        Experience: 6 months internship at local startup
        Education: Bachelor's in Computer Science
        Skills: Python basics, Excel, some SQL experience
        Looking for entry-level opportunity to grow
        """
    },
    {
        "name": "Dr. Michael Park",
        "resume": """
        Dr. Michael Park - Chief Data Officer
        Experience: 15 years in data science and analytics
        Education: PhD in Statistics, MIT
        Skills: Python, R, Scala, Spark, ML, Deep Learning,
                Leadership, Strategy
        Previous roles: CDO at Fortune 500 companies
        """
    },
]

print("Job Application Triage System")
print("=" * 50)
print("Role: Data Analyst (2-5 years experience required)")

for applicant in applicants:
    print("\n\nProcessing: " + applicant["name"])
    result = app.invoke({
        "applicant_name": applicant["name"],
        "resume_text": applicant["resume"],
        "experience_years": 0,
        "key_skills": "",
        "education_level": "",
        "qualification_status": "",
        "interview_type": "",
        "feedback": "",
        "final_decision": "",
        "review_notes": []
    })
    print(result["final_decision"])