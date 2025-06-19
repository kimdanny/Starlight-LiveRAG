"""
Falcon v2 (with fallback prompts) model connection through local clustered compute
"""

import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Any
from generators.prompt_template import (
    SYSTEM_PROMPT,
    USER_SUBPROMPT_INFO,
    USER_PROMPT_WITH_INFO,
    USER_FALLBACK_PROMPT_WITH_INFO,
    USER_PROMPT_WITHOUT_INFO,
    USER_FALLBACK_PROMPT_WITHOUT_INFO,
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
        # Set pad_token_id explicitly to suppress the warning
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @staticmethod
    def form_user_prompt(
        question, retrieved_docs: list[str] = None, fallback=False
    ) -> str:
        if retrieved_docs:
            infos = []
            for i, doc in enumerate(retrieved_docs):
                infos.append(USER_SUBPROMPT_INFO.format(DOC_i=i, DOC_i_TEXT=doc))
            info_subprompt = "\n".join(infos)
            if not fallback:
                user_prompt = USER_PROMPT_WITH_INFO.format(
                    QUESTION=question, INFO_PROMPT=info_subprompt
                )
            else:
                user_prompt = USER_FALLBACK_PROMPT_WITH_INFO.format(
                    QUESTION=question, INFO_PROMPT=info_subprompt
                )
        else:
            if not fallback:
                user_prompt = USER_PROMPT_WITHOUT_INFO.format(QUESTION=question)
            else:
                user_prompt = USER_FALLBACK_PROMPT_WITHOUT_INFO.format(
                    QUESTION=question
                )

        return user_prompt

    def query(
        self,
        question: str,
        retrieved_docs: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        generation_kwargs: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Send a query to the LLM and get a response with fallback handling.

        Args:
            question: The question to ask the model
            retrieved_docs: List of retrieved documents to include with the question
            system_prompt: System prompt to use (overrides the default)
            max_new_tokens: Maximum number of tokens to generate
            generation_kwargs: Additional kwargs to pass to model.generate()

        Returns:
            Dictionary containing response, extracted answer, reasoning, and the final prompt used
        """
        # Use class defaults if parameters not provided
        max_new_tokens = max_new_tokens or self.max_new_tokens
        system_prompt = system_prompt or SYSTEM_PROMPT
        generation_kwargs = generation_kwargs or {}

        def get_response(messages):
            # Apply the chat template
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Tokenize input
            model_inputs = self.tokenizer([text], return_tensors="pt").to(
                self.model.device
            )
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
            response = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            return response

        # Form the initial user prompt
        user_prompt = self.form_user_prompt(
            question=question, retrieved_docs=retrieved_docs, fallback=False
        )

        # Format messages for chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = get_response(messages)

        # Try to extract structured output
        answer = self.extract_tag_from_response(response, tag="answer")
        reasoning = self.extract_tag_from_response(response, tag="think")

        # Check if we need fallback (ill-formatted response)
        needs_fallback = (not answer) or ("<think>" in answer) or ("<answer>" in answer)
        final_prompt = user_prompt

        # If fallback is needed
        if needs_fallback:
            print(
                f"Structured output not found, using fallback for: {question[:30]}...",
                flush=True,
            )
            fallback_prompt = self.form_user_prompt(
                question=question, retrieved_docs=retrieved_docs, fallback=True
            )

            # Format messages for fallback
            fallback_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": fallback_prompt},
            ]

            fallback_response = get_response(fallback_messages)
            final_prompt = fallback_prompt

            # The full fallback response is the answer
            return {
                "response": fallback_response,
                "answer": fallback_response,  # In fallback mode, the whole response is the answer
                "reasoning": "",
                "used_fallback": True,
                "final_prompt": final_prompt,
            }

        return {
            "response": response,
            "answer": answer,
            "reasoning": reasoning,
            "used_fallback": False,
            "final_prompt": final_prompt,
        }

    def batch_query(
        self,
        questions: List[str],
        retrieved_docs_list: Optional[List[List[str]]] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        generation_kwargs: Optional[Dict] = None,
        batch_size: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Send multiple queries to the LLM and get responses with fallback handling.

        Args:
            questions: List of questions to send to the model
            retrieved_docs_list: List of retrieved documents lists for each question
            system_prompt: System prompt to use (overrides the default)
            max_new_tokens: Maximum number of tokens to generate
            generation_kwargs: Additional kwargs to pass to model.generate()
            batch_size: Number of prompts to process at once

        Returns:
            List of dictionaries containing responses, extracted answers, reasoning, and final prompts
        """
        # Use class defaults if parameters not provided
        max_new_tokens = max_new_tokens or self.max_new_tokens
        system_prompt = system_prompt or SYSTEM_PROMPT
        generation_kwargs = generation_kwargs or {}

        # If no retrieved docs provided, use None for each question
        if retrieved_docs_list is None:
            retrieved_docs_list = [None] * len(questions)

        # Create user prompts for each question
        prompts = [
            self.form_user_prompt(question=question, retrieved_docs=docs)
            for question, docs in zip(questions, retrieved_docs_list)
        ]

        all_responses = []

        # Process prompts in batches
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i : i + batch_size]
            batch_questions = questions[i : i + batch_size]
            batch_docs = retrieved_docs_list[i : i + batch_size]

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

            # Process each response to extract structured output and handle fallbacks
            batch_results = []
            fallback_indices = []
            fallback_questions = []
            fallback_docs = []

            for j, response in enumerate(batch_responses):
                # Try to extract structured output
                answer = self.extract_tag_from_response(response, tag="answer")
                answer = self.clean_answer(
                    answer
                )  # hacky way of cleaning from heuristics
                reasoning = self.extract_tag_from_response(response, tag="think")

                if (not answer) or ("<think>" in answer) or ("<answer>" in answer):
                    # Mark for fallback
                    fallback_indices.append(j)
                    fallback_questions.append(batch_questions[j])
                    fallback_docs.append(batch_docs[j])

                    # Add a placeholder to be updated later
                    batch_results.append(
                        {
                            "response": response,  # for debugging purpose
                            "answer": "",
                            "reasoning": "",
                            "used_fallback": False,  # Will be updated after fallback
                            "needs_update": True,  # Marker for later update
                            "final_prompt": batch_prompts[j],  # Initial prompt used
                        }
                    )
                else:
                    # No fallback needed
                    batch_results.append(
                        {
                            "response": response,
                            "answer": answer,
                            "reasoning": reasoning,
                            "used_fallback": False,
                            "needs_update": False,
                            "final_prompt": batch_prompts[j],  # Initial prompt used
                        }
                    )

            # Process fallbacks if needed
            if fallback_indices:
                # Create fallback prompts
                fallback_prompts = [
                    self.form_user_prompt(
                        question=question, retrieved_docs=docs, fallback=True
                    )
                    for question, docs in zip(fallback_questions, fallback_docs)
                ]

                fallback_messages = [
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ]
                    for prompt in fallback_prompts
                ]

                # Apply the chat template to each message set
                fallback_texts = [
                    self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    for messages in fallback_messages
                ]

                # Process fallbacks in smaller sub-batches if needed
                fallback_batch_size = min(batch_size, len(fallback_indices))
                fallback_responses = []

                for k in range(0, len(fallback_texts), fallback_batch_size):
                    sub_batch_texts = fallback_texts[k : k + fallback_batch_size]
                    sub_batch_prompts = fallback_prompts[k : k + fallback_batch_size]

                    # Tokenize inputs
                    fallback_inputs = self.tokenizer(
                        sub_batch_texts, return_tensors="pt", padding=True
                    ).to(self.model.device)

                    # Generate responses
                    fallback_generated_ids = self.model.generate(
                        **fallback_inputs,
                        max_new_tokens=max_new_tokens,
                        **generation_kwargs,
                    )

                    # Extract only the new tokens
                    fallback_new_tokens = []
                    for j, output_ids in enumerate(fallback_generated_ids):
                        input_length = len(fallback_inputs.input_ids[j])
                        fallback_new_tokens.append(output_ids[input_length:])

                    # Decode the responses
                    sub_batch_fallback_responses = self.tokenizer.batch_decode(
                        fallback_new_tokens, skip_special_tokens=True
                    )

                    fallback_responses.extend(sub_batch_fallback_responses)

                # Update the original batch results with fallback responses
                for j, (idx, fallback_response, fallback_prompt) in enumerate(
                    zip(fallback_indices, fallback_responses, fallback_prompts)
                ):
                    batch_results[idx] = {
                        "response": fallback_response,
                        "answer": fallback_response,  # In fallback mode, the whole response is the answer
                        "reasoning": "",
                        "used_fallback": True,
                        "needs_update": False,
                        "final_prompt": fallback_prompt,  # Fallback prompt used
                    }

            # Add processed batch to all responses
            all_responses.extend(batch_results)

            print(
                f"Processed batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}",
                flush=True,
            )

        # Clean up the results by removing temporary fields
        for result in all_responses:
            if "needs_update" in result:
                del result["needs_update"]

        return all_responses

    @staticmethod
    def extract_tag_from_response(res: str, tag: str = "answer") -> str:
        if tag == "answer":
            matched = re.search(r"<answer>(.*?)</answer>", res, re.DOTALL)
        elif tag == "think":
            matched = re.search(r"<think>(.*?)</think>", res, re.DOTALL)
        return matched.group(1).strip() if matched else ""

    @staticmethod
    def clean_answer(text: str) -> str:
        """Remove leading '> ' and trailing '\n>' if they exist"""
        if text.startswith("> "):
            text = text[2:]
        if text.endswith("\n>"):
            text = text[:-2]
        return text

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


# if __name__ == "__main__":
#     # Initialize the LLM
#     llm = FalconLLM(
#         model_name="tiiuae/Falcon3-10B-Instruct", device_map="auto", torch_dtype="auto"
#     )

#     # Print memory usage after loading
#     print("GPU Memory usage after loading model:")
#     print(llm.get_memory_usage())

#     test_queries = [
#         "vacuum excavation vs backhoe utility safety",
#         "What were the performance characteristics of the Sopwith Triplane in terms of speed, service ceiling, and endurance?",
#         "As a history student interested in scientific societies, what was the Lunar Society's purpose, and how did industrialization affect scientific progress?",
#     ]

#     res = llm.batch_query(test_queries)
#     print(res[0]["answer"])

#     # Create some fake documents for testing
#     fake_docs = {
#         # very similar docs retrieved
#         "second_brain": [
#             "A second brain is a personal knowledge management system that allows individuals to store, organize, and retrieve information efficiently. It serves as an extension of one's biological memory.",
#             "The concept of a second brain was popularized by productivity expert Tiago Forte. His methodology, called Building a Second Brain (BASB), focuses on four key activities: collecting valuable information, organizing it in a way that makes sense to you, distilling it to find the essence, and using it to create new work.",
#             "A second brain is a personal knowledge management system that allows individuals to store, organize, and retrieve information efficiently. It serves as an extension of one's biological memory.",
#             "The concept of a second brain was popularized by productivity expert Tiago Forte. His methodology, called Building a Second Brain (BASB), focuses on four key activities: collecting valuable information, organizing it in a way that makes sense to you, distilling it to find the essence, and using it to create new work.",
#             "A second brain is a personal knowledge management system that allows individuals to store, organize, and retrieve information efficiently. It serves as an extension of one's biological memory.",
#         ],
#         # very similar docs retrieved
#         "quantum_computing": [
#             "Quantum computing is a type of computation that harnesses the collective properties of quantum states, such as superposition, interference, and entanglement, to perform calculations.",
#             "Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits or qubits, which can exist in multiple states simultaneously due to superposition.",
#             "Quantum computing is a type of computation that harnesses the collective properties of quantum states, such as superposition, interference, and entanglement, to perform calculations.",
#             "Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits or qubits, which can exist in multiple states simultaneously due to superposition.",
#             "Quantum computing is a type of computation that harnesses the collective properties of quantum states, such as superposition, interference, and entanglement, to perform calculations.",
#         ],
#         # completely irrelevant docs
#         "confidential_computing": [
#             "AI poetry is computer-generated verse created using natural language processing algorithms and machine learning models trained on vast corpora of human-written poetry.",
#             "Modern AI poetry generators can mimic various styles, forms, and even specific poets, producing content that sometimes challenges human ability to distinguish between AI and human-created verse.",
#             "Quantum computing is a type of computation that harnesses the collective properties of quantum states, such as superposition, interference, and entanglement, to perform calculations.",
#             "Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits or qubits, which can exist in multiple states simultaneously due to superposition.",
#             "Quantum computing is a type of computation that harnesses the collective properties of quantum states, such as superposition, interference, and entanglement, to perform calculations.",
#         ],
#         # mixed relevancy
#         "france": [
#             "France is a country located in Western Europe. It shares borders with Belgium, Luxembourg, Germany, Switzerland, Italy, Monaco, Spain, and Andorra.",
#             "Paris is the capital and largest city of France, known for landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.",
#             "The concept of a second brain was popularized by productivity expert Tiago Forte. His methodology, called Building a Second Brain (BASB), focuses on four key activities: collecting valuable information, organizing it in a way that makes sense to you, distilling it to find the essence, and using it to create new work.",
#             "Quantum computing is a type of computation that harnesses the collective properties of quantum states, such as superposition, interference, and entanglement, to perform calculations.",
#             "France is a country located in Western Europe. It shares borders with Belgium, Luxembourg, Germany, Switzerland, Italy, Monaco, Spain, and Andorra."
#         ]
#     }

#     # Test single querying with documents
#     test_question = "What is a second brain?"
#     test_docs = fake_docs["second_brain"]
#     print(f"\nQuestion: {test_question}")
#     print(f"Associated docs: {test_docs}")

#     response_dict = llm.query(
#         question=test_question,
#         retrieved_docs=test_docs
#     )

#     print(f"\nFinal Prompt Used:\n{response_dict['final_prompt']}")
#     print(f"\nResponse:\n{response_dict['response']}")
#     print(f"\nAnswer:\n{response_dict['answer']}")
#     print(f"\nReasoning:\n{response_dict['reasoning']}")
#     if response_dict.get('used_fallback'):
#         print("Used fallback prompt!")

#     # Test batch querying with documents
#     test_questions = [
#         "What is a second brain?",
#         "Explain quantum computing in simple terms",
#         "Explain confidential computing in simple terms",
#         "Where is the capital of France?",
#     ]

#     # Associate documents with each question
#     test_docs_list = [
#         fake_docs["second_brain"],
#         fake_docs["quantum_computing"],
#         fake_docs["confidential_computing"],
#         fake_docs["france"]
#     ]

#     print(f"\nSending batch of {len(test_questions)} queries with associated documents...")

#     # Print out the questions and their associated documents
#     for i, (question, docs) in enumerate(zip(test_questions, test_docs_list)):
#         print(f"\nQuestion {i+1}: {question}")
#         print(f"Associated docs: {docs}")

#     batch_results = llm.batch_query(
#         questions=test_questions,
#         retrieved_docs_list=test_docs_list,
#         batch_size=4  # Process in smaller batches to demonstrate batch processing
#     )

#     print("\nBatch Responses:")
#     for i, (question, result) in enumerate(zip(test_questions, batch_results)):
#         print(f"\nQuestion {i+1}: {question}")
#         if result.get('used_fallback'):
#             print("Used fallback prompt!")
#         print(f"Final Prompt Used: {result['final_prompt']}")
#         print(f"Response {i+1}: {result['response']}")
#         print(f"Answer extracted: {result['answer']}")
