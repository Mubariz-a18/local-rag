"""
LLM Module
Interact with Ollama for local LLM generation
"""

from typing import List
import requests
import json
from config import MAX_TOKENS, TEMPERATURE


class LLMGenerator:
    """Generate responses using Ollama locally."""
    
    def __init__(self, model: str = "mistral", temperature: float = TEMPERATURE):
        """Initialize the LLM generator.
        
        Args:
            model: Model name to use (default: mistral)
            temperature: Temperature for generation
        """
        self.model = model
        self.temperature = temperature
        self.ollama_url = "http://localhost:11434/api/generate"
    
    def generate(self, prompt: str, context: List[str] = None) -> str:
        """Generate response based on prompt and context.
        
        Args:
            prompt: User query/prompt
            context: Relevant context chunks
            
        Returns:
            Generated response
        """
        if context is None:
            context = []
        
        # Build the full prompt with context
        full_prompt = self._build_prompt(prompt, context)
        
        # Call Mistral LLM
        response = self._call_llm(full_prompt)
        
        return response
    
    def _build_prompt(self, prompt: str, context: List[str]) -> str:
        """Build the full prompt with context.
        
        Args:
            prompt: Original prompt
            context: List of context chunks
            
        Returns:
            Full prompt string
        """
        context_str = "\n".join(context)
        
        full_prompt = f"""Use the context below to answer the question.

Context:
{context_str}

Question:
{prompt}

Answer:"""
        
        return full_prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM via Ollama.
        
        Args:
            prompt: Full prompt to send to LLM
            
        Returns:
            LLM response
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "stream": False
            }
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.ConnectionError:
            print(f"Error: Ollama not running. Start it with: ollama serve")
            return self._generate_fallback_response(prompt)
        except Exception as e:
            print(f"Error calling Ollama API: {e}")
            return self._generate_fallback_response(prompt)
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate a fallback response when API fails.
        
        Args:
            prompt: Original prompt
            
        Returns:
            Fallback response
        """
        # if "risk" in prompt.lower():
        #     return "Based on the provided documents, there are several important risk factors to consider. These include lifestyle factors, medical conditions, and genetic predispositions. Managing these factors through regular monitoring and preventive care is essential."
        # elif "prevent" in prompt.lower():
        #     return "Prevention strategies include maintaining a healthy lifestyle, regular exercise, proper diet, stress management, and regular health check-ups. These measures can significantly reduce various health risks."
        # elif "symptoms" in prompt.lower() or "signs" in prompt.lower():
        #     return "Early warning signs should be monitored carefully. If you experience any concerning symptoms, it's important to seek professional medical advice promptly."
        # else:
        #     return "Based on the retrieved documents, here's what you should know: The information provided covers important health and wellness topics. For more specific guidance, please consult with a healthcare professional."


def generate_response(prompt: str, context_chunks: List[dict] = None) -> str:
    """Generate a response using RAG with Mistral.
    
    Args:
        prompt: User query
        context_chunks: Retrieved context chunks
        
    Returns:
        Generated response
    """
    if context_chunks is None:
        context_chunks = []
    
    context_texts = [chunk["content"] for chunk in context_chunks]
    
    generator = LLMGenerator()
    response = generator.generate(prompt, context_texts)
    
    return response
