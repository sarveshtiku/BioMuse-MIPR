import random
import logging
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

logger = logging.getLogger(__name__)

class TaskGenerator:
    """Generate benchmarking tasks from Zotero semantic graphs."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.task_configs = self.config.get('tasks', {})
        
        # Initialize TF-IDF vectorizer for query generation
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def generate_retrieval_task(self, G: nx.MultiDiGraph, papers: List[Dict[str, Any]], 
                               query_type: str = 'tag_based') -> Optional[Dict[str, Any]]:
        """
        Generate a citation-aware retrieval task.
        
        Args:
            G: Semantic graph
            papers: List of papers
            query_type: Type of query to generate ('tag_based', 'abstract_paraphrase', 'cluster_centroid')
            
        Returns:
            Task dictionary with query and ground truth
        """
        if G.number_of_nodes() == 0:
            return None
        
        # Select a source paper
        source_idx = random.choice(list(G.nodes))
        source_paper = G.nodes[source_idx]
        
        if query_type == 'tag_based':
            return self._generate_tag_based_query(G, papers, source_idx, source_paper)
        elif query_type == 'abstract_paraphrase':
            return self._generate_abstract_paraphrase_query(G, papers, source_idx, source_paper)
        elif query_type == 'cluster_centroid':
            return self._generate_cluster_centroid_query(G, papers, source_idx, source_paper)
        else:
            logger.warning(f"Unknown query type: {query_type}")
            return self._generate_tag_based_query(G, papers, source_idx, source_paper)
    
    def _generate_tag_based_query(self, G: nx.MultiDiGraph, papers: List[Dict[str, Any]], 
                                 source_idx: int, source_paper: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate query based on tags."""
        if not source_paper.get('tags'):
            return None
        
        # Select a tag from the source paper
        selected_tag = random.choice(source_paper['tags'])
        
        # Find papers with similar tags
        target_papers = []
        for node, data in G.nodes(data=True):
            if node != source_idx and selected_tag in data.get('tags', []):
                target_papers.append(node)
        
        if not target_papers:
            return None
        
        query = f"Find papers related to the topic: {selected_tag}"
        
        return {
            'task_type': 'retrieval',
            'query_type': 'tag_based',
            'query': query,
            'source_paper_idx': source_idx,
            'target_papers': target_papers,
            'ground_truth': target_papers,
            'metadata': {
                'selected_tag': selected_tag,
                'num_targets': len(target_papers)
            }
        }
    
    def _generate_abstract_paraphrase_query(self, G: nx.MultiDiGraph, papers: List[Dict[str, Any]], 
                                          source_idx: int, source_paper: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate query by paraphrasing abstract."""
        abstract = source_paper.get('abstract', '')
        if len(abstract) < 50:
            return None
        
        # Simple paraphrasing - take first sentence and modify
        sentences = abstract.split('.')
        if not sentences:
            return None
        
        first_sentence = sentences[0].strip()
        if len(first_sentence) < 20:
            return None
        
        # Create a query by focusing on key terms
        query = f"Find papers about: {first_sentence[:100]}..."
        
        # Find papers with similar abstracts
        target_papers = []
        for node, data in G.nodes(data=True):
            if node != source_idx:
                # Simple similarity check
                other_abstract = data.get('abstract', '')
                if other_abstract and self._abstracts_similar(abstract, other_abstract):
                    target_papers.append(node)
        
        return {
            'task_type': 'retrieval',
            'query_type': 'abstract_paraphrase',
            'query': query,
            'source_paper_idx': source_idx,
            'target_papers': target_papers,
            'ground_truth': target_papers,
            'metadata': {
                'source_abstract_length': len(abstract),
                'num_targets': len(target_papers)
            }
        }
    
    def _generate_cluster_centroid_query(self, G: nx.MultiDiGraph, papers: List[Dict[str, Any]], 
                                       source_idx: int, source_paper: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate query based on collection/cluster membership."""
        collections = source_paper.get('collections', [])
        if not collections:
            return None
        
        selected_collection = random.choice(collections)
        
        # Find papers in the same collection
        target_papers = []
        for node, data in G.nodes(data=True):
            if node != source_idx and selected_collection in data.get('collections', []):
                target_papers.append(node)
        
        if not target_papers:
            return None
        
        query = f"Find papers in the collection: {selected_collection}"
        
        return {
            'task_type': 'retrieval',
            'query_type': 'cluster_centroid',
            'query': query,
            'source_paper_idx': source_idx,
            'target_papers': target_papers,
            'ground_truth': target_papers,
            'metadata': {
                'selected_collection': selected_collection,
                'num_targets': len(target_papers)
            }
        }
    
    def generate_tag_prediction_task(self, G: nx.MultiDiGraph, papers: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate a tag prediction task."""
        if G.number_of_nodes() == 0:
            return None
        
        # Select a paper with tags
        papers_with_tags = []
        for node, data in G.nodes(data=True):
            if data.get('tags'):
                papers_with_tags.append((node, data))
        
        if not papers_with_tags:
            return None
        
        paper_idx, paper_data = random.choice(papers_with_tags)
        
        # Prepare input text
        title = paper_data.get('title', '')
        abstract = paper_data.get('abstract', '')
        input_text = f"{title} {abstract}".strip()
        
        if len(input_text) < 20:
            return None
        
        gold_tags = paper_data.get('tags', [])
        
        return {
            'task_type': 'tag_prediction',
            'input_text': input_text,
            'paper_idx': paper_idx,
            'gold_tags': gold_tags,
            'ground_truth': gold_tags,
            'metadata': {
                'title_length': len(title),
                'abstract_length': len(abstract),
                'num_tags': len(gold_tags)
            }
        }
    
    def generate_summarization_task(self, G: nx.MultiDiGraph, papers: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate an abstract-consistent summarization task."""
        if G.number_of_nodes() == 0:
            return None
        
        # Select a paper with both abstract and notes
        papers_with_content = []
        for node, data in G.nodes(data=True):
            if data.get('abstract') and data.get('notes'):
                papers_with_content.append((node, data))
        
        if not papers_with_content:
            return None
        
        paper_idx, paper_data = random.choice(papers_with_content)
        
        title = paper_data.get('title', '')
        notes = paper_data.get('notes', '')
        gold_summary = paper_data.get('abstract', '')
        
        # Create input text from title and notes
        input_text = f"{title} {notes}".strip()
        
        if len(input_text) < 20 or len(gold_summary) < 20:
            return None
        
        return {
            'task_type': 'summarization',
            'input_text': input_text,
            'paper_idx': paper_idx,
            'gold_summary': gold_summary,
            'ground_truth': gold_summary,
            'metadata': {
                'title_length': len(title),
                'notes_length': len(notes),
                'summary_length': len(gold_summary)
            }
        }
    
    def generate_classification_task(self, G: nx.MultiDiGraph, papers: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate a domain classification task."""
        if G.number_of_nodes() == 0:
            return None
        
        # Map collections to domains
        domain_mapping = {
            'cancer_evolution': ['cancer', 'evolution', 'genomics'],
            'protein_structure': ['protein', 'structure', 'prediction'],
            'gene_regulation': ['gene', 'regulation', 'transcription']
        }
        
        # Find papers that can be classified
        classifiable_papers = []
        for node, data in G.nodes(data=True):
            collections = data.get('collections', [])
            tags = data.get('tags', [])
            
            # Determine domain based on collections and tags
            domain = self._determine_domain(collections, tags, domain_mapping)
            if domain:
                classifiable_papers.append((node, data, domain))
        
        if not classifiable_papers:
            return None
        
        paper_idx, paper_data, true_domain = random.choice(classifiable_papers)
        
        title = paper_data.get('title', '')
        abstract = paper_data.get('abstract', '')
        input_text = f"{title} {abstract}".strip()
        
        return {
            'task_type': 'classification',
            'input_text': input_text,
            'paper_idx': paper_idx,
            'true_domain': true_domain,
            'ground_truth': true_domain,
            'possible_domains': list(domain_mapping.keys()),
            'metadata': {
                'title_length': len(title),
                'abstract_length': len(abstract)
            }
        }
    
    def _abstracts_similar(self, abstract1: str, abstract2: str, threshold: float = 0.3) -> bool:
        """Check if two abstracts are similar using TF-IDF."""
        if not abstract1 or not abstract2:
            return False
        
        try:
            # Create TF-IDF vectors
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([abstract1, abstract2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity >= threshold
        except Exception as e:
            logger.warning(f"Error computing abstract similarity: {e}")
            return False
    
    def _determine_domain(self, collections: List[str], tags: List[str], 
                         domain_mapping: Dict[str, List[str]]) -> Optional[str]:
        """Determine the domain of a paper based on collections and tags."""
        all_text = ' '.join(collections + tags).lower()
        
        best_domain = None
        best_score = 0
        
        for domain, keywords in domain_mapping.items():
            score = sum(1 for keyword in keywords if keyword.lower() in all_text)
            if score > best_score:
                best_score = score
                best_domain = domain
        
        return best_domain if best_score > 0 else None
    
    def generate_task_batch(self, G: nx.MultiDiGraph, papers: List[Dict[str, Any]], 
                           task_types: List[str], num_tasks: int = 10) -> List[Dict[str, Any]]:
        """Generate a batch of tasks of different types."""
        tasks = []
        
        for _ in range(num_tasks):
            task_type = random.choice(task_types)
            
            if task_type == 'retrieval':
                query_types = self.task_configs.get('retrieval', {}).get('query_types', ['tag_based'])
                query_type = random.choice(query_types)
                task = self.generate_retrieval_task(G, papers, query_type)
            elif task_type == 'tag_prediction':
                task = self.generate_tag_prediction_task(G, papers)
            elif task_type == 'summarization':
                task = self.generate_summarization_task(G, papers)
            elif task_type == 'classification':
                task = self.generate_classification_task(G, papers)
            else:
                logger.warning(f"Unknown task type: {task_type}")
                continue
            
            if task:
                tasks.append(task)
        
        logger.info(f"Generated {len(tasks)} tasks")
        return tasks

# Legacy functions for backward compatibility
def generate_retrieval_task(G, papers):
    generator = TaskGenerator()
    return generator.generate_retrieval_task(G, papers)

def generate_tag_prediction_task(G, papers):
    generator = TaskGenerator()
    return generator.generate_tag_prediction_task(G, papers)

def generate_summarization_task(G, papers):
    generator = TaskGenerator()
    return generator.generate_summarization_task(G, papers)
