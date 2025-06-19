from datasets import Dataset
from ragas.metrics import answer_correctness, faithfulness
from ragas import evaluate
from ragas.run_config import RunConfig
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
import os


class Metric:
    def __init__(self) -> None:
        pass

    def batch_evaluate(self, data) -> list[float]:
        pass


class AnswerCorrectness(Metric):
    def __init__(self) -> None:
        super().__init__()
        load_dotenv(override=True)
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    def batch_evaluate(self, data: dict, max_workers: int = 16) -> list[float]:
        """
        data_example = {
            "question": ["When was the first super bowl?", "Who won the most super bowls?"],
            "answer": [
                "The first superbowl was held on Jan 15, 1967",
                "The most super bowls have been won by The New England Patriots",
            ],
            "ground_truth": [
                "The first superbowl was held on January 15, 1967",
                "The New England Patriots have won the Super Bowl a record six times",
            ],
        }

        Example df output:
                               user_input                                           response                                          reference  answer_correctness
        0  When was the first super bowl?       The first superbowl was held on Jan 15, 1967   The first superbowl was held on January 15, 1967            0.999097
        1   Who won the most super bowls?  The most super bowls have been won by The New ...  The New England Patriots have won the Super Bo...            0.981072
        """
        assert set(data.keys()) == {"question", "answer", "ground_truth"}
        assert len(data["question"]) == len(data["answer"]) == len(data["ground_truth"])
        dataset = Dataset.from_dict(data)
        # decrease max worker to avoid timeout or rate limit
        score = evaluate(
            dataset,
            metrics=[answer_correctness],
            run_config=RunConfig(max_workers=max_workers),
        )
        df = score.to_pandas()
        return list(df["answer_correctness"])


class BEM(Metric):
    def __init__(self, use_gpu=True) -> None:
        super().__init__()

        # Device selection
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        )
        print(f"Using {self.device} for inference", flush=True)

        # Load tokenizer and model
        print("Loading BEM model and tokenizer...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "kortukov/answer-equivalence-bem"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "kortukov/answer-equivalence-bem"
        )

        # Move model to selected device
        self.model = self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode
        print("BEM model loaded successfully", flush=True)

    def _tokenize_example(self, example):
        """
        Tokenize a single example for the BEM model.

        Args:
            example: Dictionary containing 'question', 'reference', and 'candidate'

        Returns:
            Tokenized inputs for the model
        """
        question = example["question"]
        reference = example["reference"]
        candidate = example["candidate"]

        text = f"[CLS] {candidate} [SEP]"
        text_pair = f"{reference} [SEP] {question} [SEP]"

        return self.tokenizer(
            text=text,
            text_pair=text_pair,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def _tokenize_batch(self, examples, batch_size=16):
        """
        Tokenize a batch of examples.

        Args:
            examples: List of dictionaries with 'question', 'reference', and 'candidate'
            batch_size: Batch size for processing

        Returns:
            List of tokenized inputs
        """
        all_inputs = []

        for i in range(0, len(examples), batch_size):
            batch = examples[i : i + batch_size]
            batch_inputs = {"input_ids": [], "token_type_ids": [], "attention_mask": []}

            for example in batch:
                inputs = self._tokenize_example(example)
                batch_inputs["input_ids"].append(inputs["input_ids"][0])
                batch_inputs["token_type_ids"].append(inputs["token_type_ids"][0])
                batch_inputs["attention_mask"].append(inputs["attention_mask"][0])

            # Stack tensors
            for key in batch_inputs:
                batch_inputs[key] = torch.stack(batch_inputs[key])

            # Move to device
            for key in batch_inputs:
                batch_inputs[key] = batch_inputs[key].to(self.device)

            all_inputs.append(batch_inputs)

        return all_inputs

    def batch_evaluate(self, data, batch_size=16) -> list[float]:
        """
        Evaluate BEM scores for a list of QA triplets in batches.

        Args:
            data: List of dictionaries, each containing 'question', 'reference', and 'candidate'
            batch_size: Number of examples to process in each batch

        Returns:
            List of BEM scores for each example

        Example data format:
        data = [
            {
                'question': 'why is the sky blue',
                'reference': 'light scattering',
                'candidate': 'scattering of light'
            },
            {
                'question': 'how many planets in the solar system',
                'reference': 'eight planets',
                'candidate': 'there are 8 planets'
            }
        ]
        """
        # Prepare inputs in batches
        all_batched_inputs = self._tokenize_batch(data, batch_size)

        # Process all batches and collect scores
        all_scores = []

        with torch.no_grad():  # Disable gradient calculation for inference
            for i, batch_inputs in enumerate(all_batched_inputs):
                # Get model outputs
                outputs = self.model(**batch_inputs)

                # Convert logits to probabilities
                probs = F.softmax(outputs.logits, dim=-1)

                # Extract positive class probability (index 1)
                batch_scores = probs[:, 1].cpu().numpy().tolist()
                all_scores.extend(batch_scores)

                # Print progress
                if i % 50 == 0:
                    start_idx = i * batch_size
                    end_idx = min(start_idx + len(batch_scores), len(data))
                    print(f"Processed {end_idx}/{len(data)} examples", flush=True)

        return all_scores


class Faithfulness(Metric):
    def __init__(self) -> None:
        super().__init__()
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    def batch_evaluate(self, data: dict, max_workers: int = 16) -> list[float]:
        """
        example data format:
        data = {
            "question": ["When was the first super bowl?", "Who won the most super bowls?"],
            "answer": [
                "The first superbowl was held on Jan 15, 1967",
                "The most super bowls have been won by The New England Patriots",
            ],
            "contexts": [
                [
                    "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,"
                ],
                [
                    "The Green Bay Packers...Green Bay, Wisconsin.",
                    "The Packers compete...Football Conference",
                ],
            ],
        }

        example output:
                                    user_input                                 retrieved_contexts                                           response  faithfulness
        0  When was the first super bowl?  [The First AFL–NFL World Championship Game was...       The first superbowl was held on Jan 15, 1967           1.0
        1   Who won the most super bowls?  [The Green Bay Packers...Green Bay, Wisconsin....  The most super bowls have been won by The New ...           0.0

        """
        assert set(data.keys()) == {"question", "answer", "contexts"}
        assert len(data["question"]) == len(data["answer"]) == len(data["contexts"])
        dataset = Dataset.from_dict(data)
        score = evaluate(
            dataset,
            metrics=[faithfulness],
            run_config=RunConfig(max_workers=max_workers),
        )
        df = score.to_pandas()
        return list(df["faithfulness"])
