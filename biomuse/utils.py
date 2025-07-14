import json
import yaml
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import pickle
import numpy as np

logger = logging.getLogger(__name__)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

def convert_to_json_serializable(obj: Any) -> Any:
    """Convert object to JSON serializable format, aggressively handling float/int types."""
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, bool):
        return str(obj)  # Convert bool to string
    elif isinstance(obj, (int, float, str)) or obj is None:
        return obj
    # Aggressively convert anything that looks like a float/int
    elif hasattr(obj, 'dtype') and hasattr(obj, 'item'):
        # numpy scalar
        if 'float' in str(obj.dtype):
            return float(obj)
        elif 'int' in str(obj.dtype):
            return int(obj)
        else:
            return obj.item()
    elif hasattr(obj, '__float__'):
        try:
            return float(obj)
        except Exception:
            pass
    elif hasattr(obj, '__int__'):
        try:
            return int(obj)
        except Exception:
            pass
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    print(f"[DEBUG] Fallback serialization: type={type(obj)}, value={obj}")
    return str(obj)  # Fallback: convert to string

def save_json(obj: Any, path: str):
    """Save object to JSON file."""
    # Convert to JSON serializable format
    json_obj = convert_to_json_serializable(obj)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)

def load_json(path: str) -> Any:
    """Load object from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_yaml(obj: Any, path: str):
    """Save object to YAML file."""
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(obj, f, default_flow_style=False, allow_unicode=True)

def load_yaml(path: str) -> Any:
    """Load object from YAML file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_pickle(obj: Any, path: str):
    """Save object to pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path: str) -> Any:
    """Load object from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None):
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logger.info(f"Logging setup complete. Level: {log_level}")

def ensure_dir(path: str):
    """Ensure directory exists, create if it doesn't."""
    os.makedirs(path, exist_ok=True)

def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    import re
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def extract_tags_from_text(text: str) -> List[str]:
    """Extract tags from text (comma-separated or line-separated)."""
    if not text:
        return []
    
    # Split by commas or newlines
    tags = []
    for line in text.split('\n'):
        for tag in line.split(','):
            tag = tag.strip()
            if tag:
                tags.append(tag)
    
    return tags

def parse_model_response(response: str, task_type: str) -> Dict[str, Any]:
    """Parse model response based on task type."""
    if task_type == 'tag_prediction':
        # Extract tags from response
        tags = extract_tags_from_text(response)
        return {'predicted_tags': tags}
    
    elif task_type == 'retrieval':
        # Try to extract paper indices from response
        # This is a simplified parser - in practice you'd need more sophisticated parsing
        import re
        numbers = re.findall(r'\b\d+\b', response)
        paper_indices = [int(n) for n in numbers if int(n) < 1000]  # Assume reasonable paper count
        return {'predicted_papers': paper_indices}
    
    elif task_type == 'summarization':
        # Return the response as the summary
        return {'predicted_summary': response.strip()}
    
    elif task_type == 'classification':
        # Try to extract domain from response
        response_lower = response.lower()
        domains = ['cancer_evolution', 'protein_structure', 'gene_regulation']
        
        for domain in domains:
            if domain.lower() in response_lower:
                return {'predicted_domain': domain}
        
        # Default to first domain if none found
        return {'predicted_domain': domains[0]}
    
    else:
        return {'raw_response': response}

def create_results_summary(results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a summary of benchmark results."""
    summary = {
        'timestamp': get_timestamp(),
        'config': config,
        'overall_metrics': {},
        'task_metrics': {},
        'model_performance': {}
    }
    
    # Aggregate metrics across tasks
    for task_type, metrics in results.items():
        if isinstance(metrics, dict) and 'mean' in str(metrics):
            summary['task_metrics'][task_type] = metrics
    
    return summary

def save_results(results: Dict[str, Any], output_dir: str, experiment_name: str):
    """Save benchmark results to files."""
    ensure_dir(output_dir)
    
    timestamp = get_timestamp()
    
    # Save detailed results
    results_file = os.path.join(output_dir, f"{experiment_name}_{timestamp}_results.json")
    save_json(results, results_file)
    
    # Save summary
    summary = create_results_summary(results, {})
    summary_file = os.path.join(output_dir, f"{experiment_name}_{timestamp}_summary.json")
    save_json(summary, summary_file)
    
    logger.info(f"Results saved to {output_dir}")
    return results_file, summary_file

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        return load_yaml(config_path)
    elif config_path.endswith('.json'):
        return load_json(config_path)
    else:
        raise ValueError(f"Unsupported config file format: {config_path}")

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration dictionary."""
    required_sections = ['models', 'tasks', 'evaluation']
    
    for section in required_sections:
        if section not in config:
            logger.error(f"Missing required config section: {section}")
            return False
    
    return True

def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """Format metrics for display."""
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted.append(f"{key}: {value:.{precision}f}")
        else:
            formatted.append(f"{key}: {value}")
    
    return ", ".join(formatted)
