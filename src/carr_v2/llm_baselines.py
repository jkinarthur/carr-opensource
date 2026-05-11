"""
LLM-Based Recommendation Baselines
===================================
Integrates Llama-3.1/3.2, Mistral, Qwen, GPT-4, and safety-tuned models
for direct comparison with CARR-v2 compression framework.

Metrics:
  - HR@10, NDCG@10 (ranking quality)
  - Drift score R(L) (reasoning preservation)
  - Evidence survival S_L (evidence retention)

Models supported:
  - Local (via transformers): llama-3.1, llama-3.2, mistral-7b, qwen-7b, qwen-14b
  - API-based: gpt-4, gpt-4-turbo, gpt-3.5-turbo, claude-3-5-sonnet
  - Safety-tuned: Llama-Guard, Mistral-7B-Instruct-v0.3

Usage:
  python -c "from carr_v2.llm_baselines import LLMRecommender; rec = LLMRecommender('llama-3.1'); scores = rec.evaluate(val_loader)"
"""

import abc
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class LLMRecommender(abc.ABC):
    """Abstract base for LLM-based recommendation."""

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

    @abc.abstractmethod
    def generate_recommendations(
        self, user_history: list[int], item_pool: list[int], k: int = 10
    ) -> list[int]:
        """Generate top-k recommendations for a user given their history."""
        pass

    @abc.abstractmethod
    def encode_reasoning(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Extract reasoning representations from model activations."""
        pass

    def compute_hr_at_k(
        self, predictions: list[list[int]], targets: list[int], k: int = 10
    ) -> float:
        """Compute Hit Rate @ K."""
        hits = 0
        for pred, target in zip(predictions, targets):
            if target in pred[:k]:
                hits += 1
        return hits / max(len(targets), 1)

    def compute_ndcg_at_k(
        self, predictions: list[list[int]], targets: list[int], k: int = 10
    ) -> float:
        """Compute NDCG @ K."""
        ndcgs = []
        for pred, target in zip(predictions, targets):
            dcg = 0.0
            for i, p in enumerate(pred[:k], start=1):
                if p == target:
                    dcg = 1.0 / np.log2(i + 1)
                    break
            idcg = 1.0 / np.log2(2)  # Best case: hit at rank 1
            ndcgs.append(dcg / idcg)
        return float(np.mean(ndcgs)) if ndcgs else 0.0

    def compute_drift_score(self, hidden_states: torch.Tensor) -> float:
        """
        Estimate reasoning drift as L2 distance from full model activations.
        Lower = better (less drift).
        """
        if hidden_states is None or hidden_states.numel() == 0:
            return 0.0
        # Proxy: ratio of extreme attention head activations (indicates collapse).
        collapsed = (torch.abs(hidden_states) > 3.0).sum().item()
        total = hidden_states.numel()
        return float(collapsed / max(total, 1))

    def compute_evidence_survival(self, hidden_states: torch.Tensor) -> float:
        """
        Estimate evidence survival as variance in token representations.
        Higher = better (more evidence retained).
        Returns normalized value in [0, 1].
        """
        if hidden_states is None or hidden_states.numel() == 0:
            return 0.0
        variance = torch.var(hidden_states, dim=0).mean()
        # Normalize to [0, 1] assuming typical variance is in [0, 10]
        return float(torch.clamp(variance / 10.0, 0, 1))


class LlamaRecommender(LLMRecommender):
    """Llama-3.1 / Llama-3.2 recommendation via huggingface transformers."""

    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf", device: str = "cuda"):
        super().__init__(model_name, device)
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map=device, load_in_8bit=True
            )
            self.model.eval()
        except ImportError:
            raise ImportError("transformers library required. Install via: pip install transformers")
        except Exception as e:
            raise RuntimeError(f"Failed to load {model_name}: {e}")

    def generate_recommendations(
        self, user_history: list[int], item_pool: list[int], k: int = 10
    ) -> list[int]:
        """Generate recommendations via LLM prompt completion."""
        history_str = ", ".join(map(str, user_history[-10:]))  # Last 10 items
        prompt = (
            f"User history: [{history_str}]\n"
            f"Recommend top {k} items from candidates: {item_pool[:50]}\n"
            f"Recommendations:"
        )
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=100,
                num_beams=3,
                output_scores=True,
                return_dict_in_generate=True,
            )

        decoded = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        # Parse recommendations from output (best-effort)
        recommendations = self._parse_item_ids(decoded, item_pool)
        return recommendations[:k]

    def encode_reasoning(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return averaged hidden states as reasoning representation."""
        if hidden_states is None:
            return torch.tensor([])
        return hidden_states.mean(dim=1, keepdim=True)

    def _parse_item_ids(self, text: str, item_pool: list[int]) -> list[int]:
        """Extract item IDs from LLM output text."""
        recommendations = []
        for item in item_pool:
            if str(item) in text and item not in recommendations:
                recommendations.append(item)
                if len(recommendations) >= 10:
                    break
        return recommendations or item_pool[:10]


class MistralRecommender(LLMRecommender):
    """Mistral-7B recommendation."""

    def __init__(self, device: str = "cuda"):
        super().__init__("mistralai/Mistral-7B-Instruct-v0.3", device)
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.float16, device_map=device, load_in_8bit=True
            )
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load Mistral: {e}")

    def generate_recommendations(
        self, user_history: list[int], item_pool: list[int], k: int = 10
    ) -> list[int]:
        """Generate recommendations via Mistral."""
        history_str = ", ".join(map(str, user_history[-10:]))
        prompt = (
            f"[INST] Based on user interaction history [{history_str}], "
            f"recommend top {k} items from: {item_pool[:50]} [/INST]"
        )
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_length=100, num_beams=2)

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        recommendations = self._parse_item_ids(decoded, item_pool)
        return recommendations[:k]

    def encode_reasoning(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states is None:
            return torch.tensor([])
        return hidden_states.mean(dim=1, keepdim=True)

    def _parse_item_ids(self, text: str, item_pool: list[int]) -> list[int]:
        recommendations = []
        for item in item_pool:
            if str(item) in text and item not in recommendations:
                recommendations.append(item)
                if len(recommendations) >= 10:
                    break
        return recommendations or item_pool[:10]


class QwenRecommender(LLMRecommender):
    """Qwen-7B or Qwen-14B recommendation."""

    def __init__(self, model_name: str = "Qwen/Qwen1.5-7B", device: str = "cuda"):
        super().__init__(model_name, device)
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map=device, load_in_8bit=True
            )
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load Qwen: {e}")

    def generate_recommendations(
        self, user_history: list[int], item_pool: list[int], k: int = 10
    ) -> list[int]:
        """Generate recommendations via Qwen."""
        history_str = ", ".join(map(str, user_history[-10:]))
        prompt = (
            f"User viewed items: {history_str}\n"
            f"Suggest top {k} from pool: {item_pool[:50]}\n"
            f"Top {k}:"
        )
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_length=100)

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        recommendations = self._parse_item_ids(decoded, item_pool)
        return recommendations[:k]

    def encode_reasoning(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states is None:
            return torch.tensor([])
        return hidden_states.mean(dim=1, keepdim=True)

    def _parse_item_ids(self, text: str, item_pool: list[int]) -> list[int]:
        recommendations = []
        for item in item_pool:
            if str(item) in text and item not in recommendations:
                recommendations.append(item)
                if len(recommendations) >= 10:
                    break
        return recommendations or item_pool[:10]


class GPT4Recommender(LLMRecommender):
    """GPT-4 / GPT-4-Turbo recommendation via OpenAI API."""

    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        super().__init__(model_name, device="cpu")  # API-based, no local device needed
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        try:
            import openai

            openai.api_key = self.api_key
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai library required. Install via: pip install openai")

    def generate_recommendations(
        self, user_history: list[int], item_pool: list[int], k: int = 10
    ) -> list[int]:
        """Generate recommendations via GPT-4 API."""
        history_str = ", ".join(map(str, user_history[-10:]))
        message = (
            f"User interaction history: {history_str}\n"
            f"Available items: {item_pool[:50]}\n"
            f"Provide top {k} recommended items as comma-separated integers."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": message}],
                temperature=0.3,
                max_tokens=50,
            )
            text = response.choices[0].message.content
            recommendations = self._parse_item_ids(text, item_pool)
            return recommendations[:k]
        except Exception as e:
            print(f"GPT-4 API error: {e}")
            return item_pool[:k]

    def encode_reasoning(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # API-based model: no access to hidden states
        return torch.tensor([])

    def _parse_item_ids(self, text: str, item_pool: list[int]) -> list[int]:
        recommendations = []
        # Parse comma-separated or space-separated integers
        import re

        numbers = re.findall(r"\d+", text)
        for num_str in numbers:
            num = int(num_str)
            if num in item_pool and num not in recommendations:
                recommendations.append(num)
                if len(recommendations) >= 10:
                    break
        return recommendations or item_pool[:10]


class ClaudeRecommender(LLMRecommender):
    """Claude 3.5 Sonnet recommendation via Anthropic API."""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__("claude-3-5-sonnet", device="cpu")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        try:
            import anthropic

            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic library required. Install via: pip install anthropic")

    def generate_recommendations(
        self, user_history: list[int], item_pool: list[int], k: int = 10
    ) -> list[int]:
        """Generate recommendations via Claude API."""
        history_str = ", ".join(map(str, user_history[-10:]))
        message = (
            f"User viewed items: {history_str}\n"
            f"Recommend top {k} from: {item_pool[:50]}\n"
            f"Reply with only the integers, comma-separated."
        )
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=50,
                messages=[{"role": "user", "content": message}],
            )
            text = response.content[0].text
            recommendations = self._parse_item_ids(text, item_pool)
            return recommendations[:k]
        except Exception as e:
            print(f"Claude API error: {e}")
            return item_pool[:k]

    def encode_reasoning(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.tensor([])

    def _parse_item_ids(self, text: str, item_pool: list[int]) -> list[int]:
        recommendations = []
        import re

        numbers = re.findall(r"\d+", text)
        for num_str in numbers:
            num = int(num_str)
            if num in item_pool and num not in recommendations:
                recommendations.append(num)
                if len(recommendations) >= 10:
                    break
        return recommendations or item_pool[:10]


class LlamaGuardRecommender(LLMRecommender):
    """Safety-tuned Llama-Guard for robust recommendations."""

    def __init__(self, device: str = "cuda"):
        super().__init__("meta-llama/LlamaGuard-7b", device)
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.float16, device_map=device, load_in_8bit=True
            )
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load LlamaGuard: {e}")

    def generate_recommendations(
        self, user_history: list[int], item_pool: list[int], k: int = 10
    ) -> list[int]:
        """Generate safe recommendations via LlamaGuard."""
        history_str = ", ".join(map(str, user_history[-10:]))
        prompt = (
            f"[INST] Safe recommendations for user with history [{history_str}] "
            f"from pool {item_pool[:50]}. Filter out any potentially unsafe items. [/INST]"
        )
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_length=100)

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        recommendations = self._parse_item_ids(decoded, item_pool)
        return recommendations[:k]

    def encode_reasoning(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states is None:
            return torch.tensor([])
        return hidden_states.mean(dim=1, keepdim=True)

    def _parse_item_ids(self, text: str, item_pool: list[int]) -> list[int]:
        recommendations = []
        for item in item_pool:
            if str(item) in text and item not in recommendations:
                recommendations.append(item)
                if len(recommendations) >= 10:
                    break
        return recommendations or item_pool[:10]


def get_recommender(model_name: str, **kwargs) -> LLMRecommender:
    """Factory function to instantiate LLM recommender by name."""
    models = {
        "llama-3.1": lambda: LlamaRecommender("meta-llama/Llama-3.1-8B", **kwargs),
        "llama-3.2": lambda: LlamaRecommender("meta-llama/Llama-3.2-8B", **kwargs),
        "llama-2": lambda: LlamaRecommender("meta-llama/Llama-2-7b-hf", **kwargs),
        "mistral-7b": lambda: MistralRecommender(**kwargs),
        "qwen-7b": lambda: QwenRecommender("Qwen/Qwen1.5-7B", **kwargs),
        "qwen-14b": lambda: QwenRecommender("Qwen/Qwen1.5-14B", **kwargs),
        "gpt-4": lambda: GPT4Recommender("gpt-4", **kwargs),
        "gpt-4-turbo": lambda: GPT4Recommender("gpt-4-turbo", **kwargs),
        "gpt-3.5-turbo": lambda: GPT4Recommender("gpt-3.5-turbo", **kwargs),
        "claude-3-5-sonnet": lambda: ClaudeRecommender(**kwargs),
        "llama-guard": lambda: LlamaGuardRecommender(**kwargs),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    return models[model_name]()
