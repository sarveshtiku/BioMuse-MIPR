import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
from biomuse.utils import convert_to_json_serializable

logger = logging.getLogger(__name__)

class SemanticGraphBuilder:
    """Build semantic graphs from Zotero paper metadata."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
        self.max_edges_per_node = self.config.get('max_edges_per_node', 10)
        self.edge_types = self.config.get('edge_types', ['shared_tag', 'collection', 'related_item', 'abstract_similarity'])
        
        # Initialize sentence transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
            self.sentence_model = None
    
    def build_semantic_graph(self, papers: List[Dict[str, Any]]) -> nx.MultiDiGraph:
        """
        Build a semantic graph from paper metadata.
        
        Args:
            papers: List of paper dictionaries with metadata
            
        Returns:
            NetworkX MultiDiGraph with papers as nodes and semantic relationships as edges
        """
        G = nx.MultiDiGraph()
        
        # Add nodes with metadata
        for idx, paper in enumerate(papers):
            G.add_node(idx, **paper)
        
        # Build different types of edges
        if 'shared_tag' in self.edge_types:
            self._add_tag_based_edges(G, papers)
        
        if 'collection' in self.edge_types:
            self._add_collection_based_edges(G, papers)
        
        if 'related_item' in self.edge_types:
            self._add_related_item_edges(G, papers)
        
        if 'abstract_similarity' in self.edge_types:
            self._add_semantic_similarity_edges(G, papers)
        
        # Compute graph statistics
        self._compute_graph_statistics(G)
        
        logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def _add_tag_based_edges(self, G: nx.MultiDiGraph, papers: List[Dict[str, Any]]):
        """Add edges based on shared tags between papers."""
        tag_map = {}
        for i, paper in enumerate(papers):
            for tag in paper.get('tags', []):
                tag_map.setdefault(tag, []).append(i)
        
        for tag, idxs in tag_map.items():
            if len(idxs) > 1:  # Only create edges if multiple papers share the tag
                for i in idxs:
                    for j in idxs:
                        if i != j:
                            G.add_edge(i, j, type='shared_tag', tag=tag, weight=1.0)
    
    def _add_collection_based_edges(self, G: nx.MultiDiGraph, papers: List[Dict[str, Any]]):
        """Add edges based on shared collections between papers."""
        collection_map = {}
        for i, paper in enumerate(papers):
            for col in paper.get('collections', []):
                collection_map.setdefault(col, []).append(i)
        
        for col, idxs in collection_map.items():
            if len(idxs) > 1:  # Only create edges if multiple papers share the collection
                for i in idxs:
                    for j in idxs:
                        if i != j:
                            G.add_edge(i, j, type='collection', collection=col, weight=1.0)
    
    def _add_related_item_edges(self, G: nx.MultiDiGraph, papers: List[Dict[str, Any]]):
        """Add edges based on explicit related-item links."""
        for i, paper in enumerate(papers):
            for rel in paper.get('related', []):
                if not rel:
                    continue
                
                # Try to find matching papers by title similarity
                for j, other in enumerate(papers):
                    if i != j and rel and self._titles_similar(rel, other['title']):
                        G.add_edge(i, j, type='related', related_item=rel, weight=1.0)
    
    def _add_semantic_similarity_edges(self, G: nx.MultiDiGraph, papers: List[Dict[str, Any]]):
        """Add edges based on semantic similarity of abstracts."""
        if not self.sentence_model:
            logger.warning("Sentence transformer not available, skipping semantic similarity edges")
            return
        
        # Prepare texts for embedding
        texts = []
        valid_indices = []
        for i, paper in enumerate(papers):
            text = f"{paper['title']} {paper['abstract']}"
            if len(text.strip()) > 10:  # Only include papers with meaningful text
                texts.append(text)
                valid_indices.append(i)
        
        if len(texts) < 2:
            return
        
        # Compute embeddings
        try:
            embeddings = self.sentence_model.encode(texts)
            similarities = cosine_similarity(embeddings)
            
            # Add edges for high similarity pairs
            for i in range(len(valid_indices)):
                for j in range(i + 1, len(valid_indices)):
                    sim_score = similarities[i][j]
                    if sim_score >= self.similarity_threshold:
                        idx_i, idx_j = valid_indices[i], valid_indices[j]
                        G.add_edge(idx_i, idx_j, type='abstract_similarity', 
                                 similarity=sim_score, weight=sim_score)
                        G.add_edge(idx_j, idx_i, type='abstract_similarity', 
                                 similarity=sim_score, weight=sim_score)
        except Exception as e:
            logger.error(f"Error computing semantic similarities: {e}")
    
    def _titles_similar(self, title1: str, title2: str) -> bool:
        """Check if two titles are similar (for related item matching)."""
        if not title1 or not title2:
            return False
        
        # Simple similarity check - can be enhanced with more sophisticated matching
        title1_lower = title1.lower()
        title2_lower = title2.lower()
        
        # Check if one title contains the other
        if title1_lower in title2_lower or title2_lower in title1_lower:
            return True
        
        # Check word overlap
        words1 = set(title1_lower.split())
        words2 = set(title2_lower.split())
        if len(words1) > 0 and len(words2) > 0:
            overlap = len(words1.intersection(words2))
            min_len = min(len(words1), len(words2))
            if min_len > 0 and overlap / min_len > 0.5:
                return True
        
        return False
    
    def _compute_graph_statistics(self, G: nx.MultiDiGraph):
        """Compute and log graph statistics."""
        if G.number_of_nodes() == 0:
            return
        
        stats = {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
        }
        
        # Handle clustering coefficient - convert to simple graph for MultiDiGraph
        try:
            # Convert to simple undirected graph for clustering calculation
            G_simple = nx.Graph()
            for u, v, data in G.edges(data=True):
                G_simple.add_edge(u, v)
            stats['avg_clustering'] = nx.average_clustering(G_simple)
        except Exception as e:
            logger.warning(f"Could not compute clustering coefficient: {e}")
            stats['avg_clustering'] = 0.0
        
        # Handle shortest path - convert to simple graph for MultiDiGraph
        try:
            # Convert to simple undirected graph for path calculation
            G_simple = nx.Graph()
            for u, v, data in G.edges(data=True):
                G_simple.add_edge(u, v)
            if nx.is_connected(G_simple):
                stats['avg_shortest_path'] = nx.average_shortest_path_length(G_simple)
            else:
                stats['avg_shortest_path'] = float('inf')
            stats['connected_components'] = nx.number_connected_components(G_simple)
        except Exception as e:
            logger.warning(f"Could not compute path statistics: {e}")
            stats['avg_shortest_path'] = float('inf')
            stats['connected_components'] = 1
        
        # Edge type distribution
        edge_types = {}
        for u, v, data in G.edges(data=True):
            edge_type = data.get('type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        stats['edge_type_distribution'] = edge_types
        
        logger.info(f"Graph statistics: {stats}")
        return stats
    
    def get_paper_neighbors(self, G: nx.MultiDiGraph, paper_idx: int, edge_type: Optional[str] = None) -> List[int]:
        """Get neighboring papers for a given paper."""
        neighbors = []
        for neighbor in G.neighbors(paper_idx):
            if edge_type is None or any(data.get('type') == edge_type for _, _, data in G.get_edge_data(paper_idx, neighbor).values()):
                neighbors.append(neighbor)
        return neighbors
    
    def get_papers_by_tag(self, G: nx.MultiDiGraph, tag: str) -> List[int]:
        """Get papers that have a specific tag."""
        papers = []
        for node, data in G.nodes(data=True):
            if tag in data.get('tags', []):
                papers.append(node)
        return papers
    
    def get_papers_by_collection(self, G: nx.MultiDiGraph, collection: str) -> List[int]:
        """Get papers that belong to a specific collection."""
        papers = []
        for node, data in G.nodes(data=True):
            if collection in data.get('collections', []):
                papers.append(node)
        return papers
    
    def export_graph(self, G: nx.MultiDiGraph, file_path: str):
        """Export graph to JSON format."""
        graph_data = {
            'nodes': [],
            'edges': [],
            'metadata': {
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges(),
                'density': nx.density(G),
            }
        }
        
        # Export nodes
        for node, data in G.nodes(data=True):
            graph_data['nodes'].append({
                'id': node,
                'data': data
            })
        
        # Export edges
        for u, v, data in G.edges(data=True):
            graph_data['edges'].append({
                'source': u,
                'target': v,
                'data': data
            })
        
        with open(file_path, 'w') as f:
            json.dump(convert_to_json_serializable(graph_data), f, indent=2)
        
        logger.info(f"Graph exported to {file_path}")

def build_semantic_graph(papers: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None) -> nx.MultiDiGraph:
    """
    Convenience function to build semantic graph from papers.
    
    Args:
        papers: List of paper dictionaries
        config: Optional configuration dictionary
        
    Returns:
        NetworkX MultiDiGraph
    """
    builder = SemanticGraphBuilder(config)
    return builder.build_semantic_graph(papers)

if __name__ == "__main__":
    from biomuse.zotero_parser import parse_zotero_export
    
    # Load and parse papers
    papers = parse_zotero_export('data/example_zotero_export.rdf')
    print(f"Parsed {len(papers)} papers.")
    
    # Build graph
    config = {
        'similarity_threshold': 0.7,
        'max_edges_per_node': 10,
        'edge_types': ['shared_tag', 'collection', 'related_item', 'abstract_similarity']
    }
    
    G = build_semantic_graph(papers, config)
    print(f"Graph built with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    
    # Export graph
    builder = SemanticGraphBuilder(config)
    builder.export_graph(G, 'data/sample_graph.json')
