import numpy as np
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModel

np.random.seed(42)


def multiple_cutoff_rankings(scores, cutoff, invert=True, return_full_rankings=False):
    n_samples = scores.shape[0]
    n_docs = scores.shape[1]
    cutoff = min(n_docs, cutoff)

    ind = np.arange(n_samples)
    partition = np.argpartition(scores, cutoff - 1, axis=1)
    sorted_partition = np.argsort(scores[ind[:, None], partition[:, :cutoff]], axis=1)
    rankings = partition[ind[:, None], sorted_partition]

    if not invert:
        inverted = None
    else:
        inverted = np.full((n_samples, n_docs), cutoff, dtype=rankings.dtype)
        inverted[ind[:, None], rankings] = np.arange(cutoff)[None, :]

    if return_full_rankings:
        partition[:, :cutoff] = rankings
        rankings = partition

    return rankings, inverted


def gumbel_sample_rankings(
    log_scores,
    n_samples,
    cutoff=None,
    inverted=False,
    doc_prob=False,
    prob_per_rank=False,
    return_gumbel=False,
    return_full_rankings=False,
):
    """
    Adapted from
    https://github.com/HarrieO/2022-SIGIR-plackett-luce/blob/main/utils/plackettluce.py
    https://github.com/HarrieO/2022-SIGIR-plackett-luce/blob/main/utils/ranking.py

    Copyright (C) H.R. Oosterhuis 2022.
    Distributed under the MIT License (see the accompanying README.md and LICENSE files).
    """
    n_docs = log_scores.shape[0]
    ind = np.arange(n_samples)

    if cutoff:
        ranking_len = min(n_docs, cutoff)
    else:
        ranking_len = n_docs

    if prob_per_rank:
        rank_prob_matrix = np.empty((ranking_len, n_docs), dtype=np.float64)

    gumbel_samples = np.random.gumbel(size=(n_samples, n_docs))
    # gumbel_samples = np.zeros(shape=(n_samples, n_docs))
    gumbel_scores = log_scores[None, :] + gumbel_samples

    rankings, inv_rankings = multiple_cutoff_rankings(
        -gumbel_scores,
        ranking_len,
        invert=inverted,
        return_full_rankings=return_full_rankings,
    )

    if not doc_prob:
        if not return_gumbel:
            return rankings, inv_rankings, None, None, None
        else:
            return rankings, inv_rankings, None, None, gumbel_scores

    log_scores = np.tile(log_scores[None, :], (n_samples, 1))
    # print(log_scores.shape)  # n_samples x ranking_len
    rankings_prob = np.empty((n_samples, ranking_len), dtype=np.float64)
    for i in range(ranking_len):
        # normalization
        log_scores += 18 - np.amax(log_scores, axis=1)[:, None]
        log_denom = np.log(np.sum(np.exp(log_scores), axis=1))
        probs = np.exp(log_scores - log_denom[:, None])
        if prob_per_rank:
            rank_prob_matrix[i, :] = np.mean(probs, axis=0)
        rankings_prob[:, i] = probs[ind, rankings[:, i]]
        # set to 0 (NINF)
        log_scores[ind, rankings[:, i]] = np.NINF

    if return_gumbel:
        gumbel_return_values = gumbel_scores
    else:
        gumbel_return_values = None

    if prob_per_rank:
        return (
            rankings,
            inv_rankings,
            rankings_prob,
            rank_prob_matrix,
            gumbel_return_values,
        )
    else:
        return rankings, inv_rankings, rankings_prob, None, gumbel_return_values


def batch_embed_documents(
    documents: List[str],
    normalize: bool = True,
    model_name: str = "intfloat/e5-base-v2",
    device: Optional[str] = None,
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
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

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
