"""High-level prediction interface for running Narrative Consistency models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union, Any

import torch
import torch.nn as nn
from torch import Tensor

from ..config.model_config import InferenceConfig, ModelConfig
from ..model.mini_gpt2 import MiniGPT2
from ..model.bdh_recurrent import RecurrentBDH, RecurrentState
# Import the API wrapper. Assuming user created it in model/api_models.py
try:
    from ..model.api_models import APIModelWrapper
except ImportError:
    APIModelWrapper = None

from ..utils.data_loader import get_tokenizer


class NarrativePredictor:
    """Unified wrapper for Narrative Consistency models (GPT-2, BDH, or API).

    This class abstracts the underlying model differences:
    - GPT-2: Uses mean hidden states for representation.
    - BDH: Uses accumulated ρ-matrix (associative memory) for representation.
    - API: Uses external LLM calls for direct consistency prediction.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        inference_config: InferenceConfig,
        device: torch.device,
        model: Optional[Any] = None, # model can be nn.Module or APIModelWrapper
        lm_head: Optional[nn.Module] = None,
    ) -> None:
        """Initialize the Predictor.

        Args:
            model_config: Configuration for the architecture.
            inference_config: Configuration for inference-time behavior.
            device: Torch device.
            model: Optional pre-trained model instance.
            lm_head: Optional external language model head (for GPT-2).
        """
        self.model_config = model_config
        self.inference_config = inference_config
        self.device = device
        self.tokenizer = get_tokenizer(self.model_config)

        # 1. Initialize Model if not provided
        if model is not None:
            self.model = model
            if isinstance(self.model, nn.Module):
                self.model.to(device)
                self.model.eval()
        else:
            if self.model_config.model_type == "bdh":
                print("Initializing RecurrentBDH for inference...")
                self.model = RecurrentBDH(model_config).to(device)
                self.model.eval()
            elif self.model_config.model_type == "api":
                print(f"Initializing API Model ({self.model_config.api_provider})...")
                if APIModelWrapper is None:
                    raise ImportError("APIModelWrapper not found. Please ensure api_models.py exists.")
                
                self.model = APIModelWrapper(
                    provider=self.model_config.api_provider,
                    api_key=self.model_config.get_active_api_key(),
                    model_name=self.model_config.api_model_name
                )
            else:
                print("Initializing MiniGPT2 for inference...")
                self.model = MiniGPT2(model_config).to(device)
                self.model.eval()
        
        # 2. Handle LM Head (Only for local torch models)
        self.lm_head = None
        if isinstance(self.model, nn.Module):
            if self.model_config.model_type == "bdh":
                # BDH has internal head, exposed as property or attribute
                self.lm_head = getattr(self.model, 'lm_head', None)
            else:
                # GPT-2 needs external or internal head management
                if lm_head is not None:
                    self.lm_head = lm_head.to(device)
                    self.lm_head.eval()
                else:
                    self.lm_head = getattr(self.model, 'classifier', None)

    def compute_novel_state(
        self,
        book_path: Union[str, Path],
        verbose: bool = False,
    ) -> Union[Tensor, str]:
        """Compute the state representation for an entire novel.
        
        Dispatches to specific logic based on model architecture.
        """
        path = Path(book_path)
        if verbose:
            print(f"Loading book from: {path}")

        text = path.read_text(encoding="utf-8", errors="replace")
        
        # API Mode: State is just the text
        if self.model_config.model_type == "api":
            return text

        # Local Model Mode: Tokenize and compute state
        chunk_size = self.inference_config.chunk_size
        token_chunks = self.tokenizer.chunk_text(text, chunk_size)

        if verbose:
            print(f"Tokenized into {len(token_chunks)} chunks.")

        if self.model_config.model_type == "bdh":
            return self._compute_bdh_state(token_chunks)
        else:
            return self._compute_gpt2_state(token_chunks)

    def prime_with_backstory(
        self,
        text: str,
        verbose: bool = False,
    ) -> Tuple[Union[Tensor, str], None]:
        """Compute state representation for a backstory.
        
        Args:
            text: Backstory text.
            verbose: Print debug info.
            
        Returns:
            Tuple of (Representation, None).
        """
        if self.model_config.model_type == "api":
            return text, None

        chunk_size = self.inference_config.chunk_size
        token_chunks = self.tokenizer.chunk_text(text, chunk_size)

        if self.model_config.model_type == "bdh":
            return self._compute_bdh_state(token_chunks), None
        else:
            return self._compute_gpt2_state(token_chunks), None

    def compute_velocity_from_states(
        self, 
        state_backstory: Union[Tensor, str], 
        state_novel: Union[Tensor, str]
    ) -> float:
        """Compute distance between backstory and novel states.
        
        For Local Models: Calculates L2 Euclidean distance.
        For API Models: Calls API to check consistency (mapped to 0.0 or 1.0).
        """
        if self.model_config.model_type == "api":
            # state_backstory is backstory text, state_novel is context text
            # predict_consistency returns: 1 (Consistent), 0 (Contradictory), -1 (Error)
            result = self.model.predict_consistency(
                backstory=str(state_backstory), 
                context=str(state_novel)
            )
            
            if result == 1:
                return 0.0 # Low velocity/distance = Consistent
            elif result == 0:
                return 1.0 # High velocity/distance = Contradictory
            else:
                return 0.5 # Ambiguous

        # Local models: Ensure both states are tensors on the same device
        if isinstance(state_backstory, Tensor) and isinstance(state_novel, Tensor):
            sb = state_backstory.to(self.device)
            sn = state_novel.to(self.device)
            
            # Calculate L2 Norm of difference
            distance = torch.norm(sb - sn)
            return float(distance.item())
        
        return 0.0

    # --- Internal Logic for BDH (Recurrent ρ-Matrix) ---
    def _compute_bdh_state(self, token_chunks: list[list[int]]) -> Tensor:
        """Accumulate ρ-matrix across chunks sequentially."""
        state = self.model.reset_state()
        
        with torch.no_grad():
            for chunk in token_chunks:
                if not chunk: continue
                
                input_ids = torch.tensor([chunk], dtype=torch.long, device=self.device)
                
                # Forward pass updating state
                # Returns: logits, state, rho_update
                _, state, _ = self.model(
                    idx=input_ids, 
                    state=state, 
                    return_state=True
                )
                
                # Detach to prevent graph buildup
                if state: 
                    state = state.detach()
        
        # Return flattened ρ-matrix or zero vector
        if state and state.rho_matrix is not None:
            return state.rho_matrix.squeeze(0).cpu()
        else:
            # Calculate dimension: N * nh * D
            # We access internal config to calculate correct zero vector size
            multiplier = getattr(self.model_config, 'mlp_internal_dim_multiplier', 4)
            N = multiplier * self.model_config.n_embd // self.model_config.n_head
            dim = self.model_config.n_head * N * self.model_config.n_embd
            return torch.zeros(dim, device="cpu")

    # --- Internal Logic for GPT-2 (Hidden State Averaging) ---
    def _compute_gpt2_state(self, token_chunks: list[list[int]]) -> Tensor:
        """Compute average of last hidden states across all chunks."""
        hidden_means = []
        
        with torch.no_grad():
            for chunk in token_chunks:
                if not chunk: continue
                
                input_ids = torch.tensor(chunk, dtype=torch.long, device=self.device).unsqueeze(0)
                attention_mask = torch.ones_like(input_ids)
                
                # Forward pass
                outputs = self.model.gpt(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # Get last hidden state: [1, seq_len, n_embd]
                last_hidden = outputs.last_hidden_state
                
                # Average over sequence length: [n_embd]
                chunk_mean = last_hidden.mean(dim=1).squeeze(0)
                hidden_means.append(chunk_mean)
        
        if not hidden_means:
            return torch.zeros(self.model_config.n_embd, device="cpu")
        
        # Average over all chunks
        final_state = torch.stack(hidden_means).mean(dim=0).cpu()
        return final_state

    def compute_loss(self, text: str) -> float:
        """Compute language modeling loss (surprisal) for given text.
        
        Adapts to the input requirements of the specific architecture.
        """
        # API models don't support calculating loss this way usually (no access to logits)
        if self.model_config.model_type == "api":
            return 0.0

        self.model.eval()
        tokens = self.tokenizer.encode(text)
        chunk_size = self.inference_config.chunk_size
        
        # Split into chunks
        if len(tokens) > chunk_size:
            token_chunks = self.tokenizer.chunk_text(text, chunk_size)
        else:
            token_chunks = [tokens] if tokens else []
        
        total_loss = 0.0
        total_tokens = 0
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # BDH needs state tracking across chunks
        state = self.model.reset_state() if self.model_config.model_type == "bdh" else None

        with torch.no_grad():
            for chunk in token_chunks:
                if not chunk: continue
                
                # Prepare inputs
                input_ids = torch.tensor([chunk], dtype=torch.long, device=self.device)
                labels = input_ids.clone()
                
                logits = None

                if self.model_config.model_type == "bdh":
                    # BDH Forward
                    logits, state, _ = self.model(
                        idx=input_ids, 
                        state=state, 
                        return_state=True
                    )
                else:
                    # GPT-2 Forward
                    # We need explicit attention mask for GPT-2
                    attention_mask = (input_ids != 0).long()
                    outputs = self.model.gpt(
                        input_ids=input_ids, 
                        attention_mask=attention_mask
                    )
                    
                    # Project hidden states to vocab if we have a head
                    if self.lm_head is not None:
                        logits = self.lm_head(outputs.last_hidden_state)
                    else:
                        # Cannot compute LM loss without a head
                        return 0.0

                # Shift logits and labels for Causal LM loss
                # logits[..., :-1, :] predicts labels[..., 1:]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = criterion(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1)
                )
                
                # Accumulate weighted by token count
                num_tokens = shift_labels.ne(0).sum().item()
                if num_tokens > 0:
                    total_loss += loss.item() * num_tokens
                    total_tokens += num_tokens

        return total_loss / total_tokens if total_tokens > 0 else 0.0