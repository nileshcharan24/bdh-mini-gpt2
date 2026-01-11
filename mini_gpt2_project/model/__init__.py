"""Model package containing Mini-GPT2, Recurrent BDH, and API architectures."""

# Model module
from .base_model import BaseModel
from .mini_gpt2 import MiniGPT2
from .bdh_recurrent import RecurrentBDH, RecurrentState

# API Wrapper
# We use a try-except block here to ensure that if the API dependencies 
# (google-generativeai, groq) are missing, the rest of the model package still loads.
try:
    from .api_models import APIModelWrapper
except ImportError:
    APIModelWrapper = None

__all__ = [
    "BaseModel",
    "MiniGPT2",
    "RecurrentBDH",
    "RecurrentState",
    "APIModelWrapper",
]