from typing import List, Literal
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from functools import cache

HF_MODEL = "intfloat/e5-base-v2"


@cache
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    return tokenizer


@cache
def has_cuda():
    return torch.cuda.is_available()


@cache
def get_model():
    model = AutoModel.from_pretrained(HF_MODEL, trust_remote_code=True)
    if has_cuda():
        model = model.to("cuda")
    else:
        model = model.to("cpu")
    return model


def average_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def batch_embed_queries(
    queries: List[str],
    query_prefix: str = "query: ",
    pooling: Literal["cls", "avg"] = "avg",
    normalize: bool = True,
) -> List[List[float]]:
    with_prefixes = [" ".join([query_prefix, query]) for query in queries]
    tokenizer = get_tokenizer()
    model = get_model()
    with torch.no_grad():
        encoded = tokenizer(
            with_prefixes,
            padding=True,
            return_tensors="pt",
            truncation="longest_first",
        )
        encoded = encoded.to(model.device)
        model_out = model(**encoded)
        match pooling:
            case "cls":
                embeddings = model_out.last_hidden_state[:, 0]
            case "avg":
                embeddings = average_pool(
                    model_out.last_hidden_state, encoded["attention_mask"]
                )
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.tolist()


def batch_embed_documents(
    documents: List[str],
    normalize: bool = True,
) -> np.ndarray:
    """
    Embeds a batch of documents using the specified model and returns a NumPy ndarray.
    Uses average pooling for embedding generation.

    Args:
        documents: List of document texts to embed
        normalize: Whether to normalize the embeddings
        model_name: HuggingFace model name to use
        device: Device to run the model on ("cuda", "cpu", etc.)

    Returns:
        k x n NumPy array where k is the number of documents and n is embedding dimension
    """
    # Set device and load tokenizer and model
    device = "cuda" if has_cuda() else "cpu"
    tokenizer = get_tokenizer()
    model = get_model()

    # Tokenize and embed
    with torch.no_grad():
        encoded = tokenizer(
            documents,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        model_out = model(**encoded)

        # Apply average pooling
        attention_mask = encoded["attention_mask"]
        last_hidden = model_out.last_hidden_state

        # Create mask for pooling
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        )

        # Apply mask and calculate average
        sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
        sum_mask = torch.sum(input_mask_expanded, 1)
        embeddings = sum_embeddings / sum_mask

        # Normalize if requested
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Convert to numpy array
        return embeddings.cpu().numpy()
