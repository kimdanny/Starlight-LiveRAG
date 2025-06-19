"""
Falcon model connection through local clustered compute
"""

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional
from prompt_template import (
    SYSTEM_PROMPT,
    USER_PROMPT_WITH_INFO,
    USER_PROMPT_WITHOUT_INFO,
    USER_SUBPROMPT_INFO,
)


class FalconLLM:
    def __init__(
        self,
        model_name: str = "tiiuae/Falcon3-10B-Instruct",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 1024,
    ):
        """
        Initialize the Falcon LLM class with a model from HuggingFace.

        Args:
            model_name: The model name/path from HuggingFace
            device_map: Strategy for distributing model across GPUs ('auto' recommended)
            torch_dtype: Data type for model weights ('auto' recommended)
            max_new_tokens: Default maximum number of tokens to generate
            system_prompt: Default system prompt to use with the model
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,  # Required for some models with custom code
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @staticmethod
    def form_user_prompt(question, retrieved_docs: list[str] = None) -> str:
        if retrieved_docs:
            infos = []
            for i, doc in enumerate(retrieved_docs):
                infos.append(USER_SUBPROMPT_INFO.format(DOC_i=i, DOC_i_TEXT=doc))
            info_subprompt = "\n".join(infos)
            user_prompt = USER_PROMPT_WITH_INFO.format(
                QUESTION=question, INFO_PROMPT=info_subprompt
            )
        else:
            user_prompt = USER_PROMPT_WITHOUT_INFO.format(QUESTION=question)

        return user_prompt

    def query(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        generation_kwargs: Optional[Dict] = None,
    ) -> str:
        """
        Send a query to the LLM and get a response.

        Args:
            prompt: The text prompt to send to the model
            system_prompt: System prompt to use (overrides the default)
            max_new_tokens: Maximum number of tokens to generate
            generation_kwargs: Additional kwargs to pass to model.generate()

        Returns:
            The model's response as a string
        """
        # Use class defaults if parameters not provided
        max_new_tokens = max_new_tokens or self.max_new_tokens
        system_prompt = system_prompt or SYSTEM_PROMPT
        generation_kwargs = generation_kwargs or {}

        # Format messages for chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Apply the chat template
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize input
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Generate response
        generated_ids = self.model.generate(
            **model_inputs, max_new_tokens=max_new_tokens, **generation_kwargs
        )

        # Extract only the new tokens
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # Decode the response
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]

        return response

    def batch_query(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        generation_kwargs: Optional[Dict] = None,
        batch_size: int = 1,
    ) -> List[str]:
        """
        Send multiple queries to the LLM and get responses.

        Args:
            prompts: List of text prompts to send to the model
            system_prompt: System prompt to use (overrides the default)
            max_new_tokens: Maximum number of tokens to generate
            generation_kwargs: Additional kwargs to pass to model.generate()
            batch_size: Number of prompts to process at once (need to be careful with GPU memory)

        Returns:
            List of model responses as strings
        """
        # Use class defaults if parameters not provided
        max_new_tokens = max_new_tokens or self.max_new_tokens
        system_prompt = system_prompt or SYSTEM_PROMPT
        generation_kwargs = generation_kwargs or {}

        all_responses = []

        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_messages = [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                for prompt in batch_prompts
            ]

            # Apply the chat template to each message set
            texts = [
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                for messages in batch_messages
            ]

            # Tokenize inputs
            model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(
                self.model.device
            )

            # Generate responses
            generated_ids = self.model.generate(
                **model_inputs, max_new_tokens=max_new_tokens, **generation_kwargs
            )

            # Extract only the new tokens for each input in the batch
            new_tokens = []
            for j, output_ids in enumerate(generated_ids):
                input_length = len(model_inputs.input_ids[j])
                new_tokens.append(output_ids[input_length:])

            # Decode the responses
            batch_responses = self.tokenizer.batch_decode(
                new_tokens, skip_special_tokens=True
            )
            all_responses.extend(batch_responses)

            print(
                f"Processed batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}"
            )

        return all_responses

    @staticmethod
    def extract_tag_from_response(res: str, tag: str = "answer") -> str:
        if tag == "answer":
            matched = re.search(r"<answer>(.*?)</answer>", res, re.DOTALL)
            # TODO: add fallback in case of ill-formated response
        elif tag == "think":
            matched = re.search(r"<think>(.*?)</think>", res, re.DOTALL)
            # TODO: add fallback in case of ill-formated response
        return matched.group(1).strip() if matched else ""

    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """
        Get the current GPU memory usage for each GPU.

        Returns:
            A dictionary mapping GPU IDs to memory usage in GB
        """
        if not torch.cuda.is_available():
            return {"cpu": 0}

        memory_usage = {}
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / (
                1024**3
            )  # Convert to GB
            memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)  # Convert to GB
            memory_usage[f"gpu_{i}"] = {
                "allocated_gb": memory_allocated,
                "reserved_gb": memory_reserved,
            }

        return memory_usage
