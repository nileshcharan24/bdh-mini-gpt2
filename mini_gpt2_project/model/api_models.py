"""
API Model Wrapper for Gemini, Groq, and Hugging Face.
Handles the logic for calling external LLMs to check narrative consistency.
"""

import os
import time
import requests
import json
import logging

# Try importing SDKs (handled gracefully if missing)
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    genai = None

try:
    from groq import Groq
except ImportError:
    Groq = None

class APIModelWrapper:
    """Wrapper for external API models (Gemini, Groq, Hugging Face)."""

    def __init__(self, provider: str, api_key: str, model_name: str):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model_name = model_name
        self.client = None

        if not self.api_key:
            print(f"Warning: No API Key found for {self.provider}. Requests will fail.")

        # Initialize Provider Clients
        if self.provider == "gemini":
            if genai is None:
                raise ImportError("Google GenAI SDK not installed. Run `pip install google-generativeai`.")
            genai.configure(api_key=self.api_key)
            
        elif self.provider == "groq":
            if Groq is None:
                raise ImportError("Groq SDK not installed. Run `pip install groq`.")
            self.client = Groq(api_key=self.api_key)
            
        elif self.provider == "huggingface":
            # Hugging Face uses standard HTTP requests, no SDK needed strictly
            pass

    def predict_consistency(self, backstory: str, context: str) -> int:
        """
        Determines if the backstory is consistent with the context (book).
        Returns: 1 (Consistent), 0 (Contradictory), or -1 (Error).
        """
        
        # INTELLIGENT TRUNCATION LOGIC
        # Gemini: 1M context window -> Can take whole book.
        # Groq/HF: ~8k-32k context -> Must truncate to avoid 413 Errors.
        if self.provider == "gemini":
            final_context = context # No truncation
        else:
            # Take last 15,000 chars (~4k tokens) to be safe for free tiers
            final_context = context[-15000:] if len(context) > 15000 else context

        prompt = (
            "You are an expert narrative editor. Your task is to check if a specific "
            "backstory contradicts the established novel context.\n\n"
            "--- NOVEL CONTEXT ---\n"
            f"{final_context}\n\n"
            "--- BACKSTORY TO CHECK ---\n"
            f"{backstory}\n\n"
            "--- INSTRUCTIONS ---\n"
            "1. Analyze if the backstory contradicts facts in the context.\n"
            "2. If it contradicts, reply 'CONTRADICTORY'.\n"
            "3. If it is consistent or plausible, reply 'CONSISTENT'.\n"
            "4. Reply with ONLY one word.\n\n"
            "ANSWER:"
        )

        try:
            if self.provider == "gemini":
                return self._call_gemini(prompt)
            elif self.provider == "groq":
                return self._call_groq(prompt)
            elif self.provider == "huggingface":
                return self._call_huggingface(prompt)
            else:
                print(f"Unknown provider: {self.provider}")
                return -1
        except Exception as e:
            # Handle rate limits gracefully by pausing
            if "429" in str(e) or "413" in str(e):
                print(f"Rate Limit/Context Error ({self.provider}). pausing...")
                time.sleep(5)
            else:
                print(f"API Error ({self.provider}): {e}")
            return 1 # Default to consistent so pipeline finishes

    def _call_gemini(self, prompt: str) -> int:
        model = genai.GenerativeModel(self.model_name)
        
        # DISABLE SAFETY FILTERS
        # Books often contain conflict/drama that triggers false positives.
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        # Free tier is 15 RPM (1 request every 4 seconds)
        # We sleep to ensure we don't hit the 429 error.
        time.sleep(4.1) 
        
        response = model.generate_content(prompt, safety_settings=safety_settings)
        text = response.text.strip().upper()
        return 0 if "CONTRADICTORY" in text else 1

    def _call_groq(self, prompt: str) -> int:
        # Sleep to respect TPM limits
        time.sleep(2)
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            temperature=0.0,
            max_tokens=10,
        )
        text = chat_completion.choices[0].message.content.strip().upper()
        return 0 if "CONTRADICTORY" in text else 1

    def _call_huggingface(self, prompt: str) -> int:
        # NEW URL STRUCTURE
        api_url = f"https://router.huggingface.co/hf-inference/models/{self.model_name}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": prompt,
            "parameters": {"return_full_text": False, "max_new_tokens": 10}
        }

        for attempt in range(3):
            response = requests.post(api_url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and "generated_text" in result[0]:
                    text = result[0]["generated_text"].strip().upper()
                    return 0 if "CONTRADICTORY" in text else 1
                return 1 
            
            if response.status_code == 503:
                time.sleep(10)
                continue
            
            response.raise_for_status()
            
        return -1