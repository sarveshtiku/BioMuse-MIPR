import openai
import anthropic
import time
import logging
import json
from typing import Dict, Any, Optional, Tuple, List
import os
from datetime import datetime
from biomuse.utils import convert_to_json_serializable

logger = logging.getLogger(__name__)

class LLMInterface:
    """Unified interface for interacting with different LLM providers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.cache = {}
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize API clients
        self._init_openai()
        self._init_anthropic()
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.openai_client = openai.OpenAI(api_key=api_key)
            self.openai_available = True
        else:
            logger.warning("OpenAI API key not found")
            self.openai_available = False
    
    def _init_anthropic(self):
        """Initialize Anthropic client."""
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=api_key)
            self.anthropic_available = True
        else:
            logger.warning("Anthropic API key not found")
            self.anthropic_available = False
    
    def call_model(self, prompt: str, model_name: str = 'gpt-4', 
                   temperature: float = 0.2, max_tokens: int = 256,
                   include_context: bool = True, context_papers: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, float, Dict[str, Any]]:
        """
        Call an LLM with the given prompt and parameters.
        
        Args:
            prompt: The input prompt
            model_name: Name of the model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            include_context: Whether to include additional context
            context_papers: List of papers to include as context
            
        Returns:
            Tuple of (response_text, latency_seconds, metadata)
        """
        # Create cache key
        cache_key = f"{model_name}_{hash(prompt)}_{temperature}_{max_tokens}_{include_context}"
        
        if cache_key in self.cache:
            logger.info(f"Using cached response for {model_name}")
            return self.cache[cache_key]
        
        # Prepare full prompt with context if requested
        full_prompt = self._prepare_prompt(prompt, include_context, context_papers)
        
        start_time = time.time()
        
        try:
            if model_name.startswith('gpt-'):
                response, metadata = self._call_openai(full_prompt, model_name, temperature, max_tokens)
            elif model_name.startswith('claude-'):
                response, metadata = self._call_anthropic(full_prompt, model_name, temperature, max_tokens)
            elif model_name.startswith('llama-'):
                response, metadata = self._call_local_llama(full_prompt, model_name, temperature, max_tokens)
            elif model_name.startswith('biomuse-rag'):
                response, metadata = self._call_biomuse_rag(full_prompt, model_name, temperature, max_tokens, context_papers)
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            latency = time.time() - start_time
            
            # Prepare metadata
            metadata.update({
                'model': model_name,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'latency': latency,
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'prompt_length': len(full_prompt),
                'response_length': len(response),
                'include_context': include_context
            })
            
            # Cache the result
            result = (response, latency, metadata)
            self.cache[cache_key] = result
            
            logger.info(f"Successfully called {model_name} in {latency:.2f}s")
            return result
            
        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Error calling {model_name}: {e}")
            return "", latency, {
                'error': str(e),
                'model': model_name,
                'latency': latency,
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat()
            }
    
    def _call_openai(self, prompt: str, model_name: str, temperature: float, max_tokens: int) -> Tuple[str, Dict[str, Any]]:
        """Call OpenAI API."""
        if not self.openai_available:
            raise RuntimeError("OpenAI API not available")
        
        response = self.openai_client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        response_text = response.choices[0].message.content or ""
        metadata = {
            'provider': 'openai',
            'usage': {
                'input_tokens': response.usage.prompt_tokens if response.usage else 0,
                'output_tokens': response.usage.completion_tokens if response.usage else 0,
                'total_tokens': response.usage.total_tokens if response.usage else 0
            },
            'finish_reason': response.choices[0].finish_reason or 'unknown'
        }
        
        return response_text, metadata
    
    def _call_anthropic(self, prompt: str, model_name: str, temperature: float, max_tokens: int) -> Tuple[str, Dict[str, Any]]:
        """Call Anthropic API."""
        if not self.anthropic_available:
            raise RuntimeError("Anthropic API not available")
        
        response = self.anthropic_client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text if response.content else ""
        metadata = {
            'provider': 'anthropic',
            'usage': {
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens
            },
            'finish_reason': response.stop_reason
        }
        
        return response_text, metadata
    
    def _call_local_llama(self, prompt: str, model_name: str, temperature: float, max_tokens: int) -> Tuple[str, Dict[str, Any]]:
        """Call local LLaMA model (placeholder for local deployment)."""
        # This is a placeholder - in practice, you'd integrate with local LLaMA
        logger.warning("Local LLaMA not implemented - using mock response")
        
        # Mock response for testing
        response_text = f"Mock LLaMA response to: {prompt[:50]}..."
        metadata = {
            'provider': 'local_llama',
            'usage': {'input_tokens': len(prompt.split()), 'output_tokens': len(response_text.split())},
            'finish_reason': 'length'
        }
        
        return response_text, metadata
    
    def _call_biomuse_rag(self, prompt: str, model_name: str, temperature: float, max_tokens: int, 
                          context_papers: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, Dict[str, Any]]:
        """Call BioMuse-RAG system (custom retrieval-augmented generation)."""
        # This is a placeholder for the BioMuse-RAG system described in the paper
        logger.warning("BioMuse-RAG not implemented - using mock response")
        
        # Mock RAG response
        if context_papers:
            context_summary = f"Using {len(context_papers)} context papers"
        else:
            context_summary = "No context papers provided"
        
        response_text = f"BioMuse-RAG response with {context_summary}: {prompt[:50]}..."
        metadata = {
            'provider': 'biomuse_rag',
            'usage': {'input_tokens': len(prompt.split()), 'output_tokens': len(response_text.split())},
            'finish_reason': 'length',
            'context_papers_count': len(context_papers) if context_papers else 0
        }
        
        return response_text, metadata
    
    def _prepare_prompt(self, prompt: str, include_context: bool, 
                       context_papers: Optional[List[Dict[str, Any]]] = None) -> str:
        """Prepare the full prompt with optional context."""
        if not include_context or not context_papers:
            return prompt
        
        # Add context papers to the prompt
        context_text = "\n\nRelevant papers for context:\n"
        for i, paper in enumerate(context_papers[:5]):  # Limit to 5 papers
            title = paper.get('title', 'Unknown title')
            abstract = paper.get('abstract', '')[:200]  # Truncate abstract
            tags = ', '.join(paper.get('tags', [])[:3])  # Limit tags
            
            context_text += f"{i+1}. Title: {title}\n"
            context_text += f"   Abstract: {abstract}...\n"
            context_text += f"   Tags: {tags}\n\n"
        
        return context_text + "\n" + prompt
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        model_configs = self.config.get('models', {})
        
        if model_name in model_configs:
            return model_configs[model_name]
        
        # Default configurations
        defaults = {
            'gpt-4': {
                'name': 'gpt-4-0125-preview',
                'temperature': 0.2,
                'max_tokens': 256,
                'provider': 'openai',
                'available': self.openai_available
            },
            'claude': {
                'name': 'claude-3-sonnet-20240229',
                'temperature': 0.2,
                'max_tokens': 256,
                'provider': 'anthropic',
                'available': self.anthropic_available
            },
            'llama2': {
                'name': 'llama-2-13b',
                'temperature': 0.0,
                'max_tokens': 256,
                'provider': 'local',
                'available': False  # Would need local setup
            },
            'biomuse-rag': {
                'name': 'biomuse-rag',
                'temperature': 0.2,
                'max_tokens': 256,
                'provider': 'custom',
                'available': True  # Always available as mock
            }
        }
        
        return defaults.get(model_name, {})
    
    def save_session_log(self, log, path):
        """Save session log to JSON file, ensuring all data is JSON serializable."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(convert_to_json_serializable(log), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Session log saved to {path}")

# Legacy function for backward compatibility
def call_openai(prompt, model='gpt-3.5-turbo', temperature=0.2):
    interface = LLMInterface()
    response, latency, metadata = interface.call_model(prompt, model, temperature)
    return response, latency
