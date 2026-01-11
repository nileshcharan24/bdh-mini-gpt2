"""
API Model Wrapper for Gemini, Groq, and HuggingFace.
Handles external API calls for narrative consistency checking.
"""

import os
import time
import requests
import google.generativeai as genai
from groq import Groq
from typing import Optional, Dict, Any

class APIModelWrapper:
    """Unified interface for querying LLMs via API."""

    def __init__(self, provider: str, api_key: str, model_name: str = ""):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model_name = model_name

        if self.provider == "gemini":
            genai.configure(api_key=self.api_key)
            # Default to Flash if not specified
            name = self.model_name if self.model_name else "gemini-1.5-flash"
            self.client = genai.GenerativeModel(name)
        
        elif self.provider == "groq":
            self.client = Groq(api_key=self.api_key)
            # Default to Llama 3 70B if not specified
            if not self.model_name:
                self.model_name = "llama3-70b-8192"

        elif self.provider == "huggingface":
            # Just store headers for REST calls
            self.headers = {"Authorization": f"Bearer {self.api_key}"}
            if not self.model_name:
                # Default to Mistral Nemo
                self.model_name = "mistralai/Mistral-Nemo-Instruct-2407" 
            self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"

    def predict_consistency(self, backstory: str, context: str) -> int:
        """
        Ask the model if the backstory is consistent with the context.
        Returns: 1 (Consistent), 0 (Contradictory), or -1 (Error).
        """
        # Construct the prompt
        prompt = (
            "You are a narrative consistency expert. "
            "I will provide you with a Context (text from a novel) and a Backstory. "
            "Your task is to determine if the Backstory contradicts the Context.\n\n"
            "If the backstory fits or is plausible, reply with just 'CONSISTENT'. "
            "If the backstory directly contradicts facts in the context, reply with just 'CONTRADICTORY'.\n\n"
            f"--- CONTEXT ---\n{context[:100000]}..." # Truncate if massive, Gemini handles more
            f"\n\n--- BACKSTORY ---\n{backstory}\n\n"
            "ANSWER (CONSISTENT/CONTRADICTORY):"
        )
        
        try:
            response_text = ""
            
            # --- GEMINI CALL ---
            if self.provider == "gemini":
                response = self.client.generate_content(prompt)
                response_text = response.text

            # --- GROQ CALL ---
            elif self.provider == "groq":
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful consistency checker."},
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model_name,
                )
                response_text = chat_completion.choices[0].message.content

            # --- HUGGINGFACE CALL ---
            elif self.provider == "huggingface":
                payload = {"inputs": prompt}
                response = requests.post(self.api_url, headers=self.headers, json=payload)
                result = response.json()
                # HF returns list of dicts usually
                if isinstance(result, list) and 'generated_text' in result[0]:
                    response_text = result[0]['generated_text']
                elif isinstance(result, dict) and 'error' in result:
                    print(f"HF Error: {result['error']}")
                    return -1
                else:
                    response_text = str(result)

            # --- PARSE RESPONSE ---
            clean_resp = response_text.strip().upper()
            if "CONTRADICTORY" in clean_resp:
                return 0
            elif "CONSISTENT" in clean_resp:
                return 1
            else:
                print(f"Unknown response format: {clean_resp}")
                return 1 # Default to consistent (innocent until proven guilty)

        except Exception as e:
            print(f"API Error ({self.provider}): {e}")
            return -1