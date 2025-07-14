![python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Conference](https://img.shields.io/badge/IEEE%20MIPR-2025-blueviolet)
![Model Support](https://img.shields.io/badge/Models-GPT4%2C%20Claude%2C%20LLaMA%2C%20BioMuse--RAG-orange)
![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)# BioMuse: Benchmarking LLMs for Trustworthy Multimedia Retrieval in Computational Biology

### Accepted to IEEE International Conference on Multimedia Information Processing and Retrieval (MIPR) 2025

BioMuse is a novel benchmarking framework specifically designed to assess the structured retrieval and reasoning capabilities of Large Language Models (LLMs) using metadata from researcher-curated Zotero libraries. The framework converts Zotero metadata—including collections, tags, and notes—into semantic graphs and structured document clusters, providing context-rich, intent-aligned benchmarks.

## Overview

BioMuse addresses the challenge of reliably evaluating LLM accuracy, domain alignment, and multimedia retrieval capabilities in scientific research workflows. By leveraging researcher-curated Zotero libraries, BioMuse provides:

- **Citation-aware retrieval tasks** that test model ability to find relevant papers
- **Abstract-consistent summarization** that measures factual alignment
- **Semantic tag prediction** that evaluates domain understanding
- **Domain classification** that tests structural awareness

## Architecture

The BioMuse pipeline comprises four primary components:

1. **Zotero Semantic Graph Extractor**: Parses exported Zotero libraries and converts metadata into structured JSON
2. **Task Generator**: Automatically creates prompt-based tasks based on graph structure
3. **LLM Interaction Layer**: Supports inference over multiple model providers (OpenAI, Anthropic, local models)
4. **Evaluation Engine**: Scores model outputs using accuracy, ranking metrics, and semantic similarity measures

<img width="3840" height="1010" alt="Untitled diagram _ Mermaid Chart-2025-07-14-044953" src="https://github.com/user-attachments/assets/37ffb39b-69c2-4f03-aa1c-8bb544c3ed21" />

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd BioMuse-MIPR-1

# Install dependencies
pip install -r requirements.txt

# Set up environment variables for API keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## Quick Start

### 1. Prepare Your Zotero Library

Export your Zotero library as RDF/XML:
1. In Zotero, select the collection you want to benchmark
2. Right-click and select "Export Collection"
3. Choose "RDF" format
4. Save the file to `data/your_library.rdf`

### 2. Configure the Benchmark

Edit `configs/task_config.yaml` to customize:
- Model configurations (GPT-4, Claude, LLaMA, BioMuse-RAG)
- Task generation parameters
- Evaluation metrics
- Graph construction settings

### 3. Run the Benchmark

```bash
# Run with default settings
python scripts/run_benchmark.py

# Run with custom parameters
python scripts/run_benchmark.py \
    --config configs/task_config.yaml \
    --data data/your_library.rdf \
    --output results \
    --models gpt4 claude \
    --task-types retrieval tag_prediction \
    --num-tasks 20
```

### 4. Analyze Results

Results are saved to the specified output directory:
- `semantic_graph.json`: The constructed semantic graph
- `biomuse_benchmark_*.json`: Detailed evaluation results
- `benchmark_summary.json`: Summary of model performance

## Task Types

### 1. Citation-Aware Retrieval
Tests model ability to find relevant papers given a query. The model must return paper indices from the collection.

**Example Task:**
```
Query: "Find papers related to the topic: cancer genomics"
Expected: List of paper indices that match the query
```

### 2. Tag Prediction
Evaluates model understanding of domain-specific terminology by predicting relevant tags for papers.

**Example Task:**
```
Input: Paper title and abstract
Expected: Comma-separated list of relevant tags
```

### 3. Abstract-Consistent Summarization
Measures model ability to generate summaries that remain factually aligned with original abstracts.

**Example Task:**
```
Input: Paper title and notes
Expected: 2-3 sentence summary consistent with abstract
```

### 4. Domain Classification
Tests structural awareness by classifying papers into predefined domains.

**Example Task:**
```
Input: Paper title and abstract
Expected: Domain classification (cancer_evolution, protein_structure, gene_regulation)
```

## Model Support

BioMuse supports evaluation of multiple model types:

- **GPT-4** (OpenAI): State-of-the-art performance with strong reasoning capabilities
- **Claude** (Anthropic): Excellent for scientific text understanding
- **LLaMA 2** (Local): Open-source model for privacy-sensitive applications
- **BioMuse-RAG** (Custom): Retrieval-augmented generation system

## Evaluation Metrics

### Retrieval Metrics
- **Precision@K**: Accuracy of top-K retrieved papers
- **Recall@K**: Coverage of relevant papers in top-K
- **F1 Score**: Harmonic mean of precision and recall
- **NDCG**: Normalized Discounted Cumulative Gain

### Tag Prediction Metrics
- **Macro F1**: Average F1 score across all tags
- **Micro F1**: Overall F1 score considering all predictions
- **Jaccard Similarity**: Set overlap between predicted and gold tags

### Summarization Metrics
- **ROUGE-L**: Longest common subsequence-based evaluation
- **BERTScore**: Semantic similarity using BERT embeddings
- **Hallucination Rate**: Detection of factual inconsistencies

### Classification Metrics
- **Accuracy**: Percentage of correct domain classifications
- **Adjusted Rand Index**: Clustering quality measure

## Experimental Results

Our evaluations across diverse computational biology domains reveal significant performance differences:

| Model | Retrieval (F1) | Tag Prediction (F1) | Summarization (BERTScore) | Hallucination Rate |
|-------|----------------|---------------------|---------------------------|-------------------|
| GPT-4 | 82.4% | 0.74 | 0.89 | 11% |
| Claude | 76.1% | 0.66 | 0.73 | 27% |
| LLaMA 2 | 51.2% | 0.52 | 0.59 | 41% |
| BioMuse-RAG | 71.6% | 0.61 | 0.86 | 13.8% |

## Key Findings

1. **Graph Structure Matters**: Removing related-item links from semantic graphs significantly impacts retrieval accuracy (GPT-4: 82.4% → 69.5%)

2. **Hallucination Patterns**: Different models exhibit distinct hallucination patterns:
   - GPT-4: Fabrication (invented validations)
   - Claude: Semantic drift (swapping technical terms)
   - LLaMA 2: High hallucination rate (41%)

3. **Structural Awareness**: Most LLMs lack true awareness of document structure, even when scoring well on individual tasks

4. **BioMuse-RAG Advantage**: Our custom RAG system shows strong performance on citation fidelity and hallucination control despite smaller parameter count

## Configuration

The framework is highly configurable through YAML configuration files:

```yaml
# Model configurations
models:
  gpt4:
    name: "gpt-4-0125-preview"
    temperature: 0.2
    max_tokens: 256
    provider: "openai"

# Task generation parameters
tasks:
  retrieval:
    num_tasks: 50
    top_k: 3
    query_types: ["tag_based", "abstract_paraphrase", "cluster_centroid"]

# Evaluation metrics
evaluation:
  retrieval_metrics: ["precision", "recall", "f1", "ndcg"]
  tag_metrics: ["macro_f1", "micro_f1", "jaccard_similarity"]
```

<img width="959" height="629" alt="Screenshot 2025-07-13 at 22 00 30" src="https://github.com/user-attachments/assets/199b96f7-84d4-4551-8309-aafc15d702e5" />

## Extending BioMuse

### Adding New Models

To add support for a new model:

1. Implement the model interface in `biomuse/llm_interface.py`
2. Add configuration in `configs/task_config.yaml`
3. Update the `_call_*` methods in `LLMInterface`

### Adding New Task Types

To add a new task type:

1. Implement task generation in `biomuse/task_generator.py`
2. Add evaluation metrics in `biomuse/evaluation_engine.py`
3. Update the benchmark script in `scripts/run_benchmark.py`

### Custom Evaluation Metrics

To add custom evaluation metrics:

1. Implement the metric in `biomuse/evaluation_engine.py`
2. Add configuration in `configs/task_config.yaml`
3. Update the evaluation pipeline

## Future Work

- **Multimodal Integration**: Support for figures, images, and tables
- **Domain Expansion**: Extend to neuroscience and environmental modeling
- **Open-Source Toolkit**: Release Zotero → BioMuse conversion toolkit
- **Real-time Evaluation**: Live benchmarking during research workflows

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Special thanks to Dr. Emily Weigel (School of Biological Sciences, Georgia Institute of Technology) for invaluable insights on experimental design and biological reasoning, which grounded the Zotero-based evaluation workflow in real-world scientific practice.
