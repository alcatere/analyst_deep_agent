import json
from graph.workflow import create_workflow
from evaluation.metrics import run_evaluation_suite

def main():
    print("Initializing Smart Analyst Agent workflow...")
    # Use the default model
    workflow = create_workflow(model_name="qwen3.5:9b")
    
    # Define some sample questions to evaluate
    test_cases = [
        "What is 2 + 2?",
        "Who is the current president of the United States?",
        "How do you append an item to a list in Python?"
    ]
    
    print(f"Running evaluation suite with {len(test_cases)} test cases...\n")
    
    results = run_evaluation_suite(workflow, test_cases)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS:")
    print("="*50)
    print(f"Average Relevance Score (1-5): {results['average_relevance']:.2f}/5.00")
    print(f"Average Latency: {results['average_latency_sec']:.2f} seconds")
    print("-" * 50)
    print("Detailed Breakdowns:")
    
    for detail in results['details']:
        print(f"\nQuery: {detail['query']}")
        print(f"  - Latency: {detail['latency_sec']}s")
        print(f"  - Score: {detail['relevance_score']}/5")

if __name__ == "__main__":
    main()
