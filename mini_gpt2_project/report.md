# **Mini-GPT2 Approach**

Our initial approach was to fine-tune a lightweight Transformer model locally. We selected a Mini-GPT2 architecture (~25 million parameters) to ensure it could be trained and inferenced rapidly within limited compute environments (like Kaggle or Colab). This model size was specifically chosen to balance performance with computational efficiency, serving as a robust local baseline.

**Architecture & Training Procedure:**
* Model: A custom implementation of GPT-2 (MiniGPT2) with a reduced number of layers (4) and heads (11) to fit memory constraints while maintaining the causal attention mechanism.
* Objective: We treated this primarily as a Language Modeling (LM) task. The model was first fine-tuned on the raw text of the two target novels (In Search of the Castaways and The Count of Monte Cristo) to learn the specific style, characters, and plot points of the authors.
* Optimization Techniques: The model was trained using the AdamW optimizer with a learning rate of 5e-5 and weight decay of 0.1 for regularization, along with gradient clipping to a maximum norm of 1.0 to prevent exploding gradients on long sequences. Standard cross-entropy loss was used while ignoring padding tokens, and focal loss was additionally experimented with to emphasize harder-to-predict tokens, leading to slightly more stable convergence. Label smoothing was applied during training to reduce over-confidence in predictions and improve generalization on unseen backstories.

Encoding & Tokenization Technique:
1. Byte-Level Tokenization (BLE): A raw approach that maps every byte to a token. This ensures zero "unknown token" (<UNK>) errors but results in extremely long sequence lengths, making it harder for the model to capture long-range dependencies. This approach achieved an accuracy of 68%.
2. Byte-Pair Encoding (BPE): We implemented a custom BPE tokenizer trained specifically on the two novels. This compressed the text significantly (reducing sequence length by ~3-4x compared to raw bytes), allowing the model's 1024-token context window to "see" more narrative events at once. This was our most successful local configuration, achieving a peak accuracy of 73.33%.

**Evaluation Strategy:**
1. Surprisal (Loss): We hypothesized that a "consistent" backstory would have low perplexity (high probability) given the fine-tuned model's knowledge, while a contradictory one would have high surprisal.
2. Velocity (Hidden State Shift): We measured the L2 Euclidean distance between the model's internal state when processing the novel vs. when processing the backstory. A large "jump" in state space (high velocity) indicated a semantic clash or inconsistency.

**Performance Metrics (Mini-GPT2 with BPE):**
1. Accuracy: 73.33%
2. Precision: 0.75
3. Recall: 0.71
4. F1 Score: 0.73


# **API-Based Large Language Models Approach**

To establish a high-performance ceiling and test "Zero-Shot" capabilities, we utilized state-of-the-art Foundation Models via APIs. This approach leveraged the massive reasoning capabilities and context windows of modern LLMs without local fine-tuning.

**Architecture: RAG / Long-Context Prompting**
Instead of training, we treated this as a Retrieval-Augmented Generation (RAG) or Long-Context task

* Context: The entire text of the novel (or a large relevant chunk) is provided as the "System Context".
* Query: The backstory is provided as the user query.
* Prompt Engineering: We used a strict prompt instructing the model to act as a "Narrative Consistency Expert" and output a binary classification (CONSISTENT or CONTRADICTORY).

We tested several models to see how they handled the noise and specific narrative details of the books. Contrary to expectation, the APIs struggled with the specific nuance of the provided dataset compared to our fine-tuned local model.

**API Performance Results (Accuracy):**
* Gemma 3.0: 61.2%
* Gemma 2.0: 37.5%
* Llama 3.1: 64.5%
* Qwen 2.5 1B: 35.6%
* Mistral 7B: 59.78%
* Claude 2.0: 33.0%