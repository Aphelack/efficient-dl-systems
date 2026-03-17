import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from dataclasses import dataclass

from edlang.entrypoints.config import EngineConfig


@dataclass
class Request:
    request_id: int
    prompt: str
    max_new_tokens: int
    current_len: int = 0
    sampling_params: Optional[Dict[str, Any]] = None  # Bonus Part

    input_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    past_key_values: Optional[Any] = None
    generated_tokens: Optional[List[int]] = None
    generated_text: Optional[str] = None
    num_generated: int = 0
    is_finished: bool = False


@dataclass
class BatchResult:
    request_ids: List[int]
    new_tokens: List[List[int]]
    finished: List[bool]


class InferenceEngine:
    def __init__(self, engine_config: EngineConfig):
        self.model_config = engine_config.model_config

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            torch_dtype=self.model_config.torch_dtype,
            device_map=self.model_config.device,
        )
        self.model.eval()

    @torch.no_grad()
    def prefill(self, requests: List[Request]) -> BatchResult:
        """
        Prefill phase: tokenize prompts, run through model, generate first token.
        
        Steps:
        1. Tokenize prompts and create batch
        2. Forward pass with use_cache=True to get logits and KV cache
        3. Generate first token for each request (greedy: argmax)
        4. Save request state (input_ids, attention_mask, past_key_values)
        5. Check if finished (EOS token or max_new_tokens reached)
        
        Note: Use attention_mask to get real prompt length (without padding).
        """
        if not requests:
            return BatchResult(request_ids=[], new_tokens=[], finished=[])

        # TODO: Tokenize prompts and create batch (use self.tokenizer with padding=True)
        tokenized_batch = self.tokenizer(
            [r.prompt for r in requests],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # TODO: Forward pass through model
        outputs = self.model(**tokenized_batch, use_cache=True)

        # TODO: For each request:
        #   - Get real prompt length from attention_mask
        #   - Generate next token (greedy: argmax from logits[i, real_prompt_len - 1, :])
        #   - Get past_key_values for the request with self._get_past_for_request
        #   - Save state: current_len, input_ids (real part only), attention_mask, past_key_values
        #   - Set generated_tokens, num_generated, is_finished

        for i, request in enumerate(requests):
            attn = tokenized_batch["attention_mask"][i]  # (seq_len,)
            real_prompt_len = int(attn.sum().item())

            next_token = self._sample(outputs.logits[i, real_prompt_len - 1, :], request)

            request.num_generated = 1
            request.is_finished = (
                next_token == self.tokenizer.eos_token_id
                or request.num_generated >= request.max_new_tokens
            )
            request.generated_tokens = [next_token]

            request.input_ids = tokenized_batch["input_ids"][
                i, :real_prompt_len
            ].unsqueeze(0)
            request.attention_mask = tokenized_batch["attention_mask"][
                i, :real_prompt_len
            ].unsqueeze(0)

            # используем индекс в батче, а не request_id
            request.past_key_values = self._get_past_for_request(
                outputs.past_key_values,
                batch_idx=i,
                real_seq_len=real_prompt_len,
            )
            request.current_len = real_prompt_len

        return BatchResult(
            request_ids=[r.request_id for r in requests],
            new_tokens=[r.generated_tokens for r in requests],
            finished=[r.is_finished for r in requests],
        )

    @torch.no_grad()
    def decode(self, requests: List[Request]) -> BatchResult:
        """
        Decode phase: generate next token for each active request using KV cache.
        
        Steps:
        1. Filter active (non-finished) requests
        2. Prepare batched KV cache with RIGHT padding
        3. Create batch from last generated tokens
        4. Build attention_mask accounting for different sequence lengths
        5. Forward pass with past_key_values and cache_position
        6. Generate next token (greedy: argmax)
        7. Update request state
        
        Note: Use RIGHT padding for KV cache. Handle finished requests separately.
        """
        # TODO: Filter active requests (if none, return empty results for all)
        active_requests = [r for r in requests if not r.is_finished]
        if len(active_requests) == 0:
            return BatchResult(
                request_ids=[r.request_id for r in requests],
                new_tokens=[r.generated_tokens for r in requests],
                finished=[r.is_finished for r in requests],
            )

        # TODO: Prepare batched KV cache using _prepare_past_key_values_batch
        batch_past_key_values = self._prepare_past_key_values_batch(active_requests)

        # TODO: Create batch from last generated tokens [batch_size, 1]
        token_batch = torch.tensor(
            [r.generated_tokens[-1] for r in active_requests],
            device=self.model.device,
        ).unsqueeze(1)  # (B_active, 1)

        # TODO: Build attention_mask for each active request
        max_seq_len = max(r.current_len for r in active_requests)
        attention_masks = torch.zeros(
            len(active_requests),
            max_seq_len + 1,
            dtype=torch.long,
            device=self.model.device,
        )
        for i, r in enumerate(active_requests):
            # история длиной r.current_len + новый токен
            attention_masks[i, : r.current_len + 1] = 1
        attention_mask_batch = attention_masks

        # TODO: Forward pass with past_key_values
        outputs = self.model(
            input_ids=token_batch,
            attention_mask=attention_mask_batch,
            past_key_values=batch_past_key_values,
            use_cache=True,
        )

        # TODO: Get next tokens (greedy: argmax from last logit)
        

        # TODO: Update each request state (generated_tokens, num_generated, past_key_values, etc.)
        for i, request in enumerate(active_requests):
            next_token = self._sample(outputs.logits[i, -1, :], request)
            if request.generated_tokens is None:
                request.generated_tokens = []
            request.generated_tokens.append(next_token)
            request.num_generated += 1
            request.current_len += 1

            request.past_key_values = self._get_past_for_request(
                outputs.past_key_values,
                batch_idx=i,
            )

            if (
                next_token == self.tokenizer.eos_token_id
                or request.num_generated >= request.max_new_tokens
            ):
                request.is_finished = True

        return BatchResult(
            request_ids=[r.request_id for r in requests],
            new_tokens=[r.generated_tokens for r in requests],
            finished=[r.is_finished for r in requests],
        )

    def _get_past_for_request(
        self,
        past_key_values,
        batch_idx: int,
        real_seq_len: Optional[int] = None,
    ):
        if past_key_values is None:
            return None

        new_cache = DynamicCache()
        for layer_idx in range(self.model.config.num_hidden_layers):
            key = past_key_values.key_cache[layer_idx][batch_idx : batch_idx + 1]
            value = past_key_values.value_cache[layer_idx][batch_idx : batch_idx + 1]

            if real_seq_len is not None and key.shape[2] > real_seq_len:
                key = key[:, :, :real_seq_len, :]
                value = value[:, :, :real_seq_len, :]

            new_cache.update(key, value, layer_idx)
        return new_cache

    def _prepare_past_key_values_batch(self, requests: List[Request]):
        """
        Prepare batched KV cache from requests with RIGHT padding.
        
        Combines KV cache from different requests into one batch. Since requests
        may have different sequence lengths, add RIGHT padding to max_seq_len.
        """
        if not requests:
            return None
        max_seq_len = max(r.current_len for r in requests)
        new_cache = DynamicCache()
        for layer_idx in range(self.model.config.num_hidden_layers):
            keys = []
            values = []
            for r in requests:
                if r.past_key_values is not None:
                    key = r.past_key_values.key_cache[layer_idx]
                    value = r.past_key_values.value_cache[layer_idx]
                    # Pad to max_seq_len with zeros on the right
                    pad_len = max_seq_len - key.shape[2]
                    if pad_len > 0:
                        key = torch.nn.functional.pad(key, (0, 0, 0, pad_len))
                        value = torch.nn.functional.pad(value, (0, 0, 0, pad_len))
                    keys.append(key)
                    values.append(value)
                else:
                    # If no cache, add zeros
                    keys.append(
                        torch.zeros(
                            (
                                1,
                                self.model.config.num_attention_heads,
                                max_seq_len,
                                self.model.config.head_size,
                            ),
                            device=self.model.device,
                        )
                    )
                    values.append(
                        torch.zeros(
                            (
                                1,
                                self.model.config.num_attention_heads,
                                max_seq_len,
                                self.model.config.head_size,
                            ),
                            device=self.model.device,
                        )
                    )

            new_cache.update(torch.cat(keys, dim=0), torch.cat(values, dim=0), layer_idx)

        # TODO: Create new DynamicCache for batch
        return new_cache

    def _sample(self, tokens_dist: torch.Tensor, request: Request) -> int:
        # BOUNS PART - Implement sampling logic with sampling_params

        if tokens_dist.dim() == 2:
            logits = tokens_dist[0]
        else:
            logits = tokens_dist

        params = request.sampling_params or {}

        temperature = params.get("temperature", 1.0)
        top_k = params.get("top_k", None)
        top_p = params.get("top_p", None)
        do_sample = params.get("do_sample", False)
        eos_token_id = params.get("eos_token_id", self.tokenizer.eos_token_id)
        ignore_eos_token = params.get("ignore_eos_token", False)

        vocab_size = logits.numel()

        # Greedy
        if (not do_sample) or temperature == 0.0:
            token_id = int(torch.argmax(logits).item())
            if ignore_eos_token and token_id == eos_token_id:
                values, indices = torch.topk(logits, k=min(2, vocab_size))
                token_id = int(indices[-1].item())
            return token_id

        # Temperature
        logits = logits / max(temperature, 1e-6)

        # Top-k
        if top_k is not None and top_k > 0 and top_k < vocab_size:
            values, indices = torch.topk(logits, top_k)

            if top_k == 1 and ignore_eos_token and eos_token_id is not None:
                top_idx = int(indices[0].item())
                if top_idx == eos_token_id:
                    values2, indices2 = torch.topk(logits, k=min(2, vocab_size))
                    return int(indices2[-1].item())

            mask = torch.full_like(logits, float("-inf"))
            mask[indices] = logits[indices]
            logits = mask

        # Top-p
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum > top_p
            mask[..., 0] = False
            sorted_logits[mask] = float("-inf")
            logits = torch.full_like(logits, float("-inf"))
            logits[sorted_indices] = sorted_logits

        probs = F.softmax(logits, dim=-1)

        if (
            ignore_eos_token
            and eos_token_id is not None
            and 0 <= eos_token_id < vocab_size
        ):
            probs[eos_token_id] = 0.0
            total = probs.sum()
            if total > 0:
                probs = probs / total

        if probs.sum() == 0 or torch.isnan(probs).any():
            return int(torch.argmax(logits).item())

        token_id = int(torch.multinomial(probs, num_samples=1).item())
        return token_id



    def get_generated_text(self, request: Request) -> str:
        if not request.generated_tokens:
            return request.prompt

        full_ids = request.input_ids[0].tolist() + request.generated_tokens
        return self.tokenizer.decode(full_ids, skip_special_tokens=True)
