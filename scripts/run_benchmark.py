import sys
from biomuse.zotero_parser import parse_zotero_rdf
from biomuse.graph_builder import build_semantic_graph
from biomuse.task_generator import (
    generate_retrieval_task, generate_tag_prediction_task, generate_summarization_task
)
from biomuse.llm_interface import call_openai
from biomuse.evaluation_engine import evaluate_tag_prediction

def main():
    # Parse library
    papers = parse_zotero_rdf('data/example_zotero_export.rdf')
    print(f"Parsed {len(papers)} papers.")

    # Build graph
    G = build_semantic_graph(papers)
    print(f"Graph built with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    # Generate a tag prediction task
    task = generate_tag_prediction_task(G, papers)
    print("Task prompt:", task['input_text'])
    print("Gold tags:", task['gold_tags'])

    # Call LLM
    prompt = f"List relevant tags for this paper: {task['input_text']}"
    pred, latency = call_openai(prompt)
    print("LLM tags:", pred)
    print(f"LLM call latency: {latency:.2f}s")

    # Evaluate (dummy eval: assume LLM output is a comma-separated tag list)
    pred_tags = [t.strip() for t in pred.split(',')]
    results = evaluate_tag_prediction(pred_tags, task['gold_tags'])
    print("Evaluation:", results)

if __name__ == "__main__":
    main()
