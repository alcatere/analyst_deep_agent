import time
import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

def calculate_latency(start_time: float) -> float:
    """Calculates the time elapsed since start_time in seconds."""
    return round(time.time() - start_time, 2)

def evaluate_relevance(question: str, answer: str, model_name: str = "qwen3.5:9b") -> int:
    """
    Uses an LLM-as-a-Judge to evaluate how relevant and correct the answer is.
    Returns a score from 1 to 5.
    """
    llm = ChatOllama(model=model_name, temperature=0.0)
    
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an impartial judge. Your task is to evaluate the relevance and correctness of an AI's answer to a user's question. Rate the answer from 1 to 5, where 1 means completely irrelevant or incorrect, and 5 means perfectly relevant and accurate. ONLY output the integer number."),
        ("user", "Question: {question}\n\nAnswer: {answer}\n\nScore (1-5):")
    ])
    
    chain = eval_prompt | llm
    try:
        result = chain.invoke({"question": question, "answer": answer})
        
        # Try to extract the first digit from the response
        match = re.search(r'\d+', result.content)
        if match:
            score = int(match.group())
            return min(max(score, 1), 5) # clamp between 1-5
        return 0
    except Exception as e:
        print(f"Evaluation error: {e}")
        return -1

def run_evaluation_suite(workflow, test_cases: list[str]) -> dict:
    """
    Runs a test suite against the agent and calculates average latency and score.
    """
    results = []
    total_score = 0
    total_latency = 0
    
    for query in test_cases:
        print(f"Evaluating: {query}")
        start_time = time.time()
        
        try:
            response = workflow.invoke({"messages": [("user", query)]})
            ai_msg = response["messages"][-1].content
            latency = calculate_latency(start_time)
            
            score = evaluate_relevance(query, ai_msg)
            
            total_latency += latency
            total_score += score
            
            results.append({
                "query": query,
                "latency_sec": latency,
                "relevance_score": score
            })
        except Exception as e:
            print(f"Error on query '{query}': {e}")
            
    avg_score = total_score / len(test_cases) if test_cases else 0
    avg_latency = total_latency / len(test_cases) if test_cases else 0
    
    return {
        "average_relevance": avg_score,
        "average_latency_sec": avg_latency,
        "details": results
    }
