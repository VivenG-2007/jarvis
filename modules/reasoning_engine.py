"""
modules/reasoning_engine.py — Phase 4 Core Brain

Responsible for ingesting the visual environment payload (JSON Context) 
and combining it with vocal queries to generate Grounded Local AI Responses.

Assumes a local LLM is running via Ollama (e.g., Llama 3 or Mistral).
Download Ollama from https://ollama.com and run: `ollama run llama3`
"""

import json
import logging
import requests
from typing import Dict, Any, Tuple

logger = logging.getLogger("jarvis.reasoning")

class ReasoningEngine:
    def __init__(self, model_name: str = "llama3", ollama_host: str = "http://localhost:11434"):
        """
        Initializes the Reasoning Engine for local Edge Intelligence.
        
        Args:
            model_name: The local LLM to query (llama3, mistral, or phi3 for extreme speed).
            ollama_host: The local Ollama server endpoint.
        """
        self.model = model_name
        self.api_url = f"{ollama_host}/api/chat"
        
        # Core Prompt Engineering logic to strictly prevent Hallucinations.
        self.system_prompt = (
            "You are JARVIS, a highly advanced local AI assistant. "
            "You are currently connected to a real-time computer vision system. "
            "You will be provided with a JSON Context payload representing the CURRENT state of the room. "
            "Your rules are strictly as follows:\n"
            "1. You must ONLY answer based on the data inside the JSON payload.\n"
            "2. If the user asks about something not visible in the JSON, say 'I cannot see that'.\n"
            "3. Be extremely brief, concise, and direct (1-2 sentences maximum).\n"
            "4. Never hallucinate facts about the environment. Trust ONLY the JSON."
        )

    def analyze_scene(self, context_json: str, user_query: str) -> str:
        """
        Queries the local LLM using the generated semantic context.
        
        Args:
            context_json: The processed JSON string from context_builder.py
            user_query: The question transcribed from audio_engine.py
            
        Returns:
            Grounded string response from JARVIS.
        """
        # Combine the literal JSON state with the user's question
        prompt_content = f"CURRENT ROOM CONTEXT:\n{context_json}\n\nUSER QUESTION:\n{user_query}"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt_content}
        ]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature strictly restricts creative hallucinations
                "top_p": 0.9
            }
        }
        
        try:
            # Query the Local LLM
            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            reply = result["message"]["content"].strip()
            return reply
            
        except requests.exceptions.ConnectionError:
            logger.error("Ollama server unreachable. Is Ollama running?")
            return "Error: My semantic reasoning node is currently offline."
        except Exception as e:
            logger.error(f"Reasoning failure: {e}")
            return "I am having trouble processing the environment right now."

# ── Example Usage ────────────────────────────────────────────────
if __name__ == "__main__":
    engine = ReasoningEngine(model_name="llama3")
    
    # 1. Mock Context from vision system
    mock_context = json.dumps({
        "people": ["Viven"],
        "unknown_count": 1,
        "objects": ["cell phone", "laptop"],
        "interactions": ["using phone", "group activity"],
        "timestamp": "2026-04-08T20:55:00.0000"
    }, indent=2)
    
    # 2. Simulate User Queries
    queries = [
        "JARVIS, what is Viven doing?",
        "Is there anyone else in the room?",
        "Do you see a dog anywhere?",
        "Who is using the laptop?"
    ]
    
    print("--- JARVIS LLM Edge Reasoning Test ---")
    for q in queries:
        print(f"\nUser: {q}")
        reply = engine.analyze_scene(mock_context, q)
        print(f"JARVIS: {reply}")
