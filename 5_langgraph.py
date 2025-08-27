import operator
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langsmith import traceable
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
import json
import re

# ---------- Setup ----------
load_dotenv()
api_key = os.getenv("test_groq")

# Use a model that supports structured output or use alternative approach
model = ChatGroq(api_key=api_key, model_name="llama-3.1-8b-instant")

# ---------- Structured schema ----------
class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description="Score out of 10", ge=0, le=10)

# ---------- Sample essay ----------
essay2 = """India and AI Time

Now world change very fast because new tech call Artificial Intel… something (AI). India also want become big in this AI thing. If work hard, India can go top. But if no careful, India go back.

India have many good. We have smart student, many engine-ear, and good IT peoples. Big company like TCS, Infosys, Wipro already use AI. Government also do program "AI for All". It want AI in farm, doctor place, school and transport.

In farm, AI help farmer know when to put seed, when rain come, how stop bug. In health, AI help doctor see sick early. In school, AI help student learn good. Government office use AI to find bad people and work fast.

But problem come also. First is many villager no have phone or internet. So AI not help them. Second, many people lose job because AI and machine do work. Poor people get more bad.

One more big problem is privacy. AI need big big data. Who take care? India still make data rule. If no strong rule, AI do bad.

India must all people together – govern, school, company and normal people. We teach AI and make sure AI not bad. Also talk to other country and learn from them.

If India use AI good way, we become strong, help poor and make better life. But if only rich use AI, and poor no get, then big bad thing happen.

So, in short, AI time in India have many hope and many danger. We must go right road. AI must help all people, not only some. Then India grow big and world say "good job India".
"""

# ---------- LangGraph state ----------
class UPSCState(TypedDict, total=False):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[List[int], operator.add]
    avg_score: float

# ---------- Helper function for evaluation ----------
def evaluate_essay_dimension(state: UPSCState, dimension: str):
    prompt_template = ChatPromptTemplate.from_template("""
    Evaluate the {dimension} quality of the following essay and provide feedback.
    Also assign a score out of 10 at the end with "Score: X/10".

    ESSAY:
    {essay}

    Please provide detailed feedback focusing on {dimension} aspects.
    """)
    
    prompt = prompt_template.format(dimension=dimension, essay=state["essay"])
    response = model.invoke(prompt)
    
    # Extract score from response
    content = response.content
    score = 5  # default
    
    # Try to extract score
    score_match = re.search(r"Score:\s*(\d+)/10", content, re.IGNORECASE)
    if score_match:
        try:
            score = int(score_match.group(1))
        except:
            pass
    
    return content, score

# ---------- Traced node functions ----------
@traceable(name="evaluate_language_fn", tags=["dimension:language"], metadata={"dimension": "language"})
def evaluate_language(state: UPSCState):
    feedback, score = evaluate_essay_dimension(state, "language")
    return {"language_feedback": feedback, "individual_scores": [score]}

@traceable(name="evaluate_analysis_fn", tags=["dimension:analysis"], metadata={"dimension": "analysis"})
def evaluate_analysis(state: UPSCState):
    feedback, score = evaluate_essay_dimension(state, "depth of analysis")
    return {"analysis_feedback": feedback, "individual_scores": [score]}

@traceable(name="evaluate_thought_fn", tags=["dimension:clarity"], metadata={"dimension": "clarity_of_thought"})
def evaluate_thought(state: UPSCState):
    feedback, score = evaluate_essay_dimension(state, "clarity of thought")
    return {"clarity_feedback": feedback, "individual_scores": [score]}

@traceable(name="final_evaluation_fn", tags=["aggregate"])
def final_evaluation(state: UPSCState):
    prompt = (
        "Based on the following feedback, create a summarized overall feedback.\n\n"
        f"Language feedback: {state.get('language_feedback','')}\n"
        f"Depth of analysis feedback: {state.get('analysis_feedback','')}\n"
        f"Clarity of thought feedback: {state.get('clarity_feedback','')}\n\n"
        "Please provide a comprehensive overall evaluation and final score assessment."
    )
    overall = model.invoke(prompt).content
    
    scores = state.get("individual_scores", []) or []
    avg = (sum(scores) / len(scores)) if scores else 0.0
    
    return {"overall_feedback": overall, "avg_score": avg}

# ---------- Build graph ----------
graph = StateGraph(UPSCState)

graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("final_evaluation", final_evaluation)

# Fan-out → join
graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_thought")
graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_thought", "final_evaluation")
graph.add_edge("final_evaluation", END)

workflow = graph.compile()

# ---------- Direct invoke ----------
if __name__ == "__main__":
    try:
        result = workflow.invoke(
            {"essay": essay2},
            config={
                "run_name": "evaluate_upsc_essay",
                "tags": ["essay", "langgraph", "evaluation"],
                "metadata": {
                    "essay_length": len(essay2),
                    "model": "llama-3.1-8b-instant",
                    "dimensions": ["language", "analysis", "clarity"],
                },
            },
        )

        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        print("\n--- LANGUAGE FEEDBACK ---")
        print(result.get("language_feedback", ""))
        
        print("\n--- ANALYSIS FEEDBACK ---")
        print(result.get("analysis_feedback", ""))
        
        print("\n--- CLARITY FEEDBACK ---")
        print(result.get("clarity_feedback", ""))
        
        print("\n--- OVERALL FEEDBACK ---")
        print(result.get("overall_feedback", ""))
        
        print("\n--- SCORES ---")
        print(f"Individual scores: {result.get('individual_scores', [])}")
        print(f"Average score: {result.get('avg_score', 0.0):.1f}/10.0")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Trying fallback approach...")
        
        # Fallback: Simple evaluation
        prompt = f"Evaluate this UPSC essay:\n\n{essay2}\n\nProvide feedback on language, analysis, and clarity with scores."
        response = model.invoke(prompt)
        print("\nFallback Evaluation:")
        print(response.content)