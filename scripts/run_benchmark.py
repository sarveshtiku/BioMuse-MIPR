#!/usr/bin/env python3
"""
BioMuse Benchmarking Script

This script implements the complete BioMuse benchmarking pipeline:
1. Parse Zotero library
2. Build semantic graph
3. Generate tasks
4. Run LLM evaluations
5. Evaluate results
"""

import sys
import os
import argparse
import logging
from typing import Dict, Any, List, Optional
import json

logger = logging.getLogger(__name__)

# Add the biomuse package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from biomuse.zotero_parser import parse_zotero_export
from biomuse.graph_builder import SemanticGraphBuilder
from biomuse.task_generator import TaskGenerator
from biomuse.llm_interface import LLMInterface
from biomuse.evaluation_engine import EvaluationEngine
from biomuse.utils import (
    load_config, validate_config, setup_logging, ensure_dir,
    save_results, format_metrics, parse_model_response, convert_to_json_serializable
)

def run_benchmark(config_path: str, data_path: str, output_dir: str, 
                  models: Optional[List[str]] = None, task_types: Optional[List[str]] = None,
                  num_tasks: int = 10) -> Dict[str, Any]:
    """
    Run the complete BioMuse benchmarking pipeline.
    
    Args:
        config_path: Path to configuration file
        data_path: Path to Zotero export file
        output_dir: Directory to save results
        models: List of models to evaluate (if None, use all from config)
        task_types: List of task types to run (if None, use all from config)
        num_tasks: Number of tasks to generate per type
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info("Starting BioMuse benchmark")
    
    # Load and validate configuration
    config = load_config(config_path)
    if not validate_config(config):
        raise ValueError("Invalid configuration")
    
    # Setup logging
    log_level = config.get('output', {}).get('log_level', 'INFO')
    setup_logging(log_level)
    
    # Ensure output directory exists
    ensure_dir(output_dir)
    
    # Step 1: Parse Zotero library
    logger.info(f"Parsing Zotero library from {data_path}")
    papers = parse_zotero_export(data_path)
    logger.info(f"Parsed {len(papers)} papers")
    
    if len(papers) == 0:
        raise ValueError("No papers found in the Zotero export")
    
    # Step 2: Build semantic graph
    logger.info("Building semantic graph")
    graph_config = config.get('graph', {})
    graph_builder = SemanticGraphBuilder(graph_config)
    G = graph_builder.build_semantic_graph(papers)
    logger.info(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Export graph for inspection
    graph_builder.export_graph(G, os.path.join(output_dir, 'semantic_graph.json'))
    
    # Step 3: Generate tasks
    logger.info("Generating benchmark tasks")
    task_config = config.get('tasks', {})
    task_generator = TaskGenerator(config)
    
    if task_types is None:
        task_types = list(task_config.keys())
    
    all_tasks = []
    for task_type in task_types:
        num_tasks_per_type = task_config.get(task_type, {}).get('num_tasks', num_tasks)
        logger.info(f"Generating {num_tasks_per_type} {task_type} tasks")
        
        for _ in range(num_tasks_per_type):
            if task_type == 'retrieval':
                task = task_generator.generate_retrieval_task(G, papers)
            elif task_type == 'tag_prediction':
                task = task_generator.generate_tag_prediction_task(G, papers)
            elif task_type == 'summarization':
                task = task_generator.generate_summarization_task(G, papers)
            elif task_type == 'classification':
                task = task_generator.generate_classification_task(G, papers)
            else:
                logger.warning(f"Unknown task type: {task_type}")
                continue
            
            if task:
                all_tasks.append(task)
    
    logger.info(f"Generated {len(all_tasks)} tasks total")
    
    # Step 4: Initialize LLM interface and evaluation engine
    llm_interface = LLMInterface(config)
    evaluation_engine = EvaluationEngine(config)
    
    # Determine models to evaluate
    if models is None:
        models = list(config.get('models', {}).keys())
    
    # Step 5: Run evaluations for each model
    all_results = {}
    
    for model_name in models:
        logger.info(f"Evaluating model: {model_name}")
        
        model_config = config.get('models', {}).get(model_name, {})
        if not model_config:
            logger.warning(f"No configuration found for model {model_name}")
            continue
        
        model_results = {
            'model': model_name,
            'config': model_config,
            'tasks': [],
            'metrics': {}
        }
        
        predictions = []
        
        for i, task in enumerate(all_tasks):
            logger.info(f"Running task {i+1}/{len(all_tasks)} for {model_name}")
            
            # Prepare prompt based on task type
            prompt = prepare_prompt(task, model_config)
            
            # Call model
            response, latency, metadata = llm_interface.call_model(
                prompt=prompt,
                model_name=model_config.get('name', model_name),
                temperature=model_config.get('temperature', 0.2),
                max_tokens=model_config.get('max_tokens', 256)
            )
            
            # Parse response
            prediction = parse_model_response(response, task['task_type'])
            prediction.update({
                'task_id': i,
                'task_type': task['task_type'],
                'latency': latency,
                'metadata': metadata
            })
            
            predictions.append(prediction)
            
            # Store task and prediction
            model_results['tasks'].append({
                'task': task,
                'prediction': prediction,
                'response': response
            })
        
        # Step 6: Evaluate results
        logger.info(f"Evaluating results for {model_name}")
        evaluation_results = evaluation_engine.evaluate_task_batch(all_tasks, predictions)
        model_results['metrics'] = evaluation_results
        
        all_results[model_name] = model_results
        
        # Log summary
        logger.info(f"Results for {model_name}:")
        for task_type, metrics in evaluation_results.items():
            if isinstance(metrics, dict):
                logger.info(f"  {task_type}: {format_metrics(metrics)}")
    
    # Step 7: Save results
    logger.info("Saving results")
    save_results(all_results, output_dir, "biomuse_benchmark")
    
    # Step 8: Generate summary
    summary = generate_benchmark_summary(all_results, config)
    summary_file = os.path.join(output_dir, "benchmark_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(convert_to_json_serializable(summary), f, indent=2, ensure_ascii=False)
    
    logger.info("Benchmark completed successfully")
    return all_results

def prepare_prompt(task: Dict[str, Any], model_config: Dict[str, Any]) -> str:
    """Prepare prompt for a specific task."""
    task_type = task['task_type']
    
    if task_type == 'retrieval':
        return f"Given the query: '{task['query']}', find relevant papers from the collection. Return the paper indices as a list of numbers."
    
    elif task_type == 'tag_prediction':
        return f"Given this paper: '{task['input_text'][:500]}...', predict relevant tags. Return the tags as a comma-separated list."
    
    elif task_type == 'summarization':
        return f"Summarize this paper: '{task['input_text'][:500]}...'. Provide a concise 2-3 sentence summary."
    
    elif task_type == 'classification':
        domains = task.get('possible_domains', ['cancer_evolution', 'protein_structure', 'gene_regulation'])
        return f"Classify this paper into one of these domains: {', '.join(domains)}. Paper: '{task['input_text'][:500]}...'. Return only the domain name."
    
    else:
        return f"Task: {task.get('input_text', 'Unknown task')}"

def generate_benchmark_summary(results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a summary of benchmark results."""
    summary = {
        'timestamp': config.get('timestamp', ''),
        'config': config,
        'model_comparison': {},
        'best_performers': {}
    }
    
    # Compare models across tasks
    for model_name, model_results in results.items():
        metrics = model_results.get('metrics', {})
        summary['model_comparison'][model_name] = {}
        
        for task_type, task_metrics in metrics.items():
            if isinstance(task_metrics, dict):
                # Extract key metrics
                key_metrics = {}
                for key, value in task_metrics.items():
                    if isinstance(value, (int, float)) and 'mean' in key:
                        key_metrics[key] = value
                
                summary['model_comparison'][model_name][task_type] = key_metrics
    
    # Find best performers
    for task_type in ['retrieval', 'tag_prediction', 'summarization', 'classification']:
        best_model = None
        best_score = -1
        
        for model_name, model_metrics in summary['model_comparison'].items():
            task_metrics = model_metrics.get(task_type, {})
            
            # Use F1 score for retrieval and tag prediction, accuracy for classification
            if task_type in ['retrieval', 'tag_prediction']:
                score = task_metrics.get('f1_mean', 0)
            elif task_type == 'classification':
                score = task_metrics.get('accuracy_mean', 0)
            else:  # summarization
                score = task_metrics.get('bertscore_mean', 0)
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        if best_model:
            summary['best_performers'][task_type] = {
                'model': best_model,
                'score': best_score
            }
    
    return summary

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Run BioMuse benchmark')
    parser.add_argument('--config', default='configs/task_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', default='data/example_zotero_export.rdf',
                       help='Path to Zotero export file')
    parser.add_argument('--output', default='results',
                       help='Output directory for results')
    parser.add_argument('--models', nargs='+',
                       help='Models to evaluate (default: all from config)')
    parser.add_argument('--task-types', nargs='+',
                       help='Task types to run (default: all from config)')
    parser.add_argument('--num-tasks', type=int, default=10,
                       help='Number of tasks per type')
    
    args = parser.parse_args()
    
    try:
        results = run_benchmark(
            config_path=args.config,
            data_path=args.data,
            output_dir=args.output,
            models=args.models,
            task_types=args.task_types,
            num_tasks=args.num_tasks
        )
        
        print(f"\nBenchmark completed successfully!")
        print(f"Results saved to: {args.output}")
        
        # Print summary
        summary = generate_benchmark_summary(results, {})
        print("\nBest performers:")
        for task_type, performer in summary.get('best_performers', {}).items():
            print(f"  {task_type}: {performer['model']} (score: {performer['score']:.3f})")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
