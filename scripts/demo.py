#!/usr/bin/env python3
"""
BioMuse Demo Script

This script demonstrates the basic BioMuse functionality without requiring API keys.
It shows the pipeline steps and generates mock results.
"""

import sys
import os
import json
from typing import Dict, Any

# Add the biomuse package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from biomuse.zotero_parser import parse_zotero_export
from biomuse.graph_builder import SemanticGraphBuilder
from biomuse.task_generator import TaskGenerator
from biomuse.evaluation_engine import EvaluationEngine
from biomuse.utils import load_config, setup_logging

def run_demo():
    """Run a demonstration of the BioMuse pipeline."""
    print("🚀 BioMuse Demo - Trustworthy LLM Benchmarking")
    print("=" * 60)
    
    # Setup logging
    setup_logging('INFO')
    
    # Step 1: Load configuration
    print("\n📋 Step 1: Loading Configuration")
    try:
        config = load_config('../configs/task_config.yaml')
        print("✅ Configuration loaded successfully")
        print(f"   Available models: {list(config.get('models', {}).keys())}")
        print(f"   Available task types: {list(config.get('tasks', {}).keys())}")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return
    
    # Step 2: Parse Zotero library
    print("\n📚 Step 2: Parsing Zotero Library")
    data_path = '../data/example_zotero_export.rdf'
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("   Please ensure you have a Zotero export file in the data directory")
        return
    
    try:
        papers = parse_zotero_export(data_path)
        print(f"✅ Successfully parsed {len(papers)} papers")
        
        if papers:
            sample_paper = papers[0]
            print(f"   Sample paper: {sample_paper.get('title', 'N/A')[:50]}...")
            print(f"   Tags: {sample_paper.get('tags', [])[:3]}")
        else:
            print("   No papers found in the export file")
            return
            
    except Exception as e:
        print(f"❌ Failed to parse Zotero library: {e}")
        return
    
    # Step 3: Build semantic graph
    print("\n🕸️  Step 3: Building Semantic Graph")
    try:
        graph_config = config.get('graph', {})
        graph_builder = SemanticGraphBuilder(graph_config)
        G = graph_builder.build_semantic_graph(papers)
        
        print(f"✅ Graph built successfully")
        print(f"   Nodes: {G.number_of_nodes()}")
        print(f"   Edges: {G.number_of_edges()}")
        
        # Analyze graph structure
        if G.number_of_edges() > 0:
            edge_types = {}
            for u, v, data in G.edges(data=True):
                edge_type = data.get('type', 'unknown')
                edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
            
            print("   Edge types:")
            for edge_type, count in edge_types.items():
                print(f"     - {edge_type}: {count}")
        
    except Exception as e:
        print(f"❌ Failed to build semantic graph: {e}")
        return
    
    # Step 4: Generate tasks
    print("\n🎯 Step 4: Generating Benchmarking Tasks")
    try:
        task_generator = TaskGenerator(config)
        
        # Generate sample tasks
        task_types = ['retrieval', 'tag_prediction', 'summarization', 'classification']
        tasks = {}
        
        for task_type in task_types:
            print(f"   Generating {task_type} tasks...")
            task_list = []
            
            for i in range(2):  # Generate 2 tasks per type
                if task_type == 'retrieval':
                    task = task_generator.generate_retrieval_task(G, papers)
                elif task_type == 'tag_prediction':
                    task = task_generator.generate_tag_prediction_task(G, papers)
                elif task_type == 'summarization':
                    task = task_generator.generate_summarization_task(G, papers)
                elif task_type == 'classification':
                    task = task_generator.generate_classification_task(G, papers)
                
                if task:
                    task_list.append(task)
            
            tasks[task_type] = task_list
            print(f"     Generated {len(task_list)} {task_type} tasks")
        
        total_tasks = sum(len(task_list) for task_list in tasks.values())
        print(f"✅ Generated {total_tasks} tasks total")
        
    except Exception as e:
        print(f"❌ Failed to generate tasks: {e}")
        return
    
    # Step 5: Mock evaluation
    print("\n🤖 Step 5: Mock Model Evaluation")
    try:
        evaluation_engine = EvaluationEngine(config)
        
        # Mock responses for demonstration
        mock_responses = {
            'retrieval': [1, 3, 5],
            'tag_prediction': ['cancer', 'genomics', 'machine learning'],
            'summarization': 'This paper presents a novel approach to cancer genomics using machine learning techniques.',
            'classification': 'cancer_evolution'
        }
        
        results = {}
        
        for task_type, task_list in tasks.items():
            if task_list:
                print(f"   Evaluating {task_type}...")
                
                # Mock evaluation
                if task_type == 'retrieval':
                    task = task_list[0]
                    ground_truth = task.get('ground_truth', [])
                    predicted = mock_responses['retrieval']
                    eval_result = evaluation_engine.evaluate_retrieval(predicted, ground_truth)
                    
                elif task_type == 'tag_prediction':
                    task = task_list[0]
                    gold_tags = task.get('gold_tags', [])
                    predicted_tags = mock_responses['tag_prediction']
                    eval_result = evaluation_engine.evaluate_tag_prediction(predicted_tags, gold_tags)
                    
                elif task_type == 'summarization':
                    task = task_list[0]
                    gold_summary = task.get('gold_summary', '')
                    predicted_summary = mock_responses['summarization']
                    eval_result = evaluation_engine.evaluate_summarization(predicted_summary, gold_summary)
                    
                elif task_type == 'classification':
                    task = task_list[0]
                    true_domain = task.get('true_domain', '')
                    predicted_domain = mock_responses['classification']
                    possible_domains = task.get('possible_domains', [])
                    eval_result = evaluation_engine.evaluate_classification(predicted_domain, true_domain, possible_domains)
                
                results[task_type] = eval_result
                print(f"     {task_type}: {eval_result}")
        
        print("✅ Mock evaluation completed")
        
    except Exception as e:
        print(f"❌ Failed to run evaluation: {e}")
        return
    
    # Step 6: Summary
    print("\n📊 Step 6: Demo Summary")
    print("=" * 60)
    print("✅ BioMuse pipeline demonstration completed successfully!")
    print("\n📈 Pipeline Statistics:")
    print(f"   • Papers processed: {len(papers)}")
    print(f"   • Graph nodes: {G.number_of_nodes()}")
    print(f"   • Graph edges: {G.number_of_edges()}")
    print(f"   • Tasks generated: {total_tasks}")
    print(f"   • Task types: {list(tasks.keys())}")
    
    print("\n🎯 Key Features Demonstrated:")
    print("   • Zotero library parsing and metadata extraction")
    print("   • Semantic graph construction with multiple edge types")
    print("   • Dynamic task generation from graph structure")
    print("   • Comprehensive evaluation metrics")
    print("   • Multi-model benchmarking framework")
    
    print("\n🚀 Next Steps:")
    print("   • Set up API keys for real model evaluation")
    print("   • Run full benchmark with: python scripts/run_benchmark.py")
    print("   • Explore results in the notebooks/ directory")
    print("   • Extend with your own Zotero libraries")
    
    print("\n" + "=" * 60)
    print("BioMuse: Advancing Trustworthy AI in Scientific Research")
    print("=" * 60)

if __name__ == "__main__":
    run_demo() 