import random

def generate_retrieval_task(G, papers):
    node_idx = random.choice(list(G.nodes))
    node = G.nodes[node_idx]
    if not node['tags']:
        return None  # skip papers without tags
    query = f"Find papers related to the topic: {random.choice(node['tags'])}"
    target_papers = [
        i for i, paper in enumerate(papers)
        if any(col in node['collections'] for col in paper['collections'])
    ]
    return {
        'task_type': 'retrieval',
        'query': query,
        'target_papers': target_papers
    }

def generate_tag_prediction_task(G, papers):
    idx = random.choice(list(G.nodes))
    paper = G.nodes[idx]
    input_text = paper['title'] + ' ' + paper['abstract']
    gold_tags = paper['tags']
    return {
        'task_type': 'tag_prediction',
        'input_text': input_text,
        'gold_tags': gold_tags
    }

def generate_summarization_task(G, papers):
    idx = random.choice(list(G.nodes))
    paper = G.nodes[idx]
    input_text = paper['title'] + ' ' + paper['notes']
    gold_summary = paper['abstract']
    return {
        'task_type': 'summarization',
        'input_text': input_text,
        'gold_summary': gold_summary
    }
