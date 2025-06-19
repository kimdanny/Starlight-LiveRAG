"""
Set up of all retrievers (actions of a router)

- Recommended to use at test time (when selecting one action per query)
    - thus not supporting batch querying
- For all retrievers, final result yields top 5 passages
    - For two-stage cascaded retrieval, first stage retrieves 50 passages, and return 5 passage after reranking
"""

from retrievers.OpenSearchConnection import OpenSearchConnection
from retrievers.PineconeConnection import PineconeConnection
from retrievers.ReRankers import StochasticReRanker, ScoreRegularizationReRanker
from retrievers.utils import batch_embed_documents
from typing import Tuple
import numpy as np


# Retriever 1: pure BM25
def bm25(
    query: str,
    oc: OpenSearchConnection,
    first_stage_config: dict = {},
    return_extras=False,
) -> Tuple[list[str], list[str]] | Tuple[list[str], list[str], np.ndarray]:
    # set default top_k to 5
    top_k = first_stage_config.get("top_k", 5)
    res = oc.query_opensearch(query, top_k=top_k)

    if res["timed_out"]:
        raise Exception("Opensearch timed out error")
    # get results
    texts, doc_ids, scores = oc.get_texts_docids_scores(res["hits"]["hits"])

    if return_extras:
        return texts, doc_ids, scores
    return texts, doc_ids


# Retriever 2: BM25 + StochasticReRanking
def bm25_stochastic(
    query: str,
    oc: OpenSearchConnection,
    first_stage_config: dict = {"top_k": 50},
    rerank_config: dict = {},
) -> Tuple[list[str], list[str]]:
    first_stage_texts, first_stage_doc_ids, first_stage_scores = bm25(
        query, oc, first_stage_config, return_extras=True
    )
    top_k = rerank_config.get("top_k", 5)
    rr = StochasticReRanker(
        first_stage_texts, first_stage_doc_ids, first_stage_scores, top_k
    )
    if alpha := rerank_config.get("alpha"):
        rr.ALPHA = alpha
    if n_samples := rerank_config.get("n_samples"):
        rr.N_SAMPLES = n_samples

    texts, doc_ids = rr.rerank()
    return texts, doc_ids


# Retriever 3: BM25 + ScoreRegularizationReRanking
def bm25_regularize(
    query: str,
    oc: OpenSearchConnection,
    first_stage_config: dict = {"top_k": 50},
    rerank_config: dict = {},
) -> Tuple[list[str], list[str]]:
    first_stage_texts, first_stage_doc_ids, first_stage_scores = bm25(
        query, oc, first_stage_config, return_extras=True
    )
    # set up a reranker
    top_k = rerank_config.get("top_k", 5)
    device = rerank_config.get("device", "cuda")
    embeddings = batch_embed_documents(first_stage_texts, device=device)
    rr = ScoreRegularizationReRanker(
        first_stage_texts, first_stage_doc_ids, embeddings, first_stage_scores, top_k
    )
    if top_m := rerank_config.get("top_m"):
        rr.top_m = top_m

    texts, doc_ids = rr.rerank()
    return texts, doc_ids


# Retriever 4: pure E5BaseV2
def e5base(
    query: str,
    pc: PineconeConnection,
    first_stage_config: dict = {},
    return_extras=False,
) -> Tuple[list[str], list[str]] | Tuple[list[str], list[str], np.ndarray, np.ndarray]:
    # set default top_k to 5
    top_k = first_stage_config.get("top_k", 5)
    res = pc.query_pinecone(query, top_k)
    texts, doc_ids, embeddings, scores = pc.get_texts_docids_embeddings_and_scores(
        matches=res["matches"]
    )

    if return_extras:
        return texts, doc_ids, embeddings, scores
    return texts, doc_ids


# Retriever 5: E5BaseV2 + StochasticReRanking
def e5base_stochastic(
    query: str,
    pc: PineconeConnection,
    first_stage_config: dict = {"top_k": 50},
    rerank_config: dict = {},
) -> Tuple[list[str], list[str]]:
    first_stage_texts, first_stage_doc_ids, _, first_stage_scores = e5base(
        query, pc, first_stage_config, return_extras=True
    )
    top_k = rerank_config.get("top_k", 5)
    rr = StochasticReRanker(
        first_stage_texts, first_stage_doc_ids, first_stage_scores, top_k
    )
    if alpha := rerank_config.get("alpha"):
        rr.ALPHA = alpha
    if n_samples := rerank_config.get("n_samples"):
        rr.N_SAMPLES = n_samples

    texts, doc_ids = rr.rerank()
    return texts, doc_ids


# Retriever 6: E5BaseV2 + ScoreRegularizationReRanking
def e5base_regularize(
    query: str,
    pc: PineconeConnection,
    first_stage_config: dict = {"top_k": 50},
    rerank_config: dict = {},
) -> Tuple[list[str], list[str]]:
    (
        first_stage_texts,
        first_stage_doc_ids,
        first_stage_embeddings,
        first_stage_scores,
    ) = e5base(query, pc, first_stage_config, return_extras=True)
    # set up a reranker
    top_k = rerank_config.get("top_k", 5)
    device = rerank_config.get("device", "cuda")
    rr = ScoreRegularizationReRanker(
        first_stage_texts,
        first_stage_doc_ids,
        first_stage_embeddings,
        first_stage_scores,
        top_k,
    )
    if top_m := rerank_config.get("top_m"):
        rr.top_m = top_m

    texts, doc_ids = rr.rerank()
    return texts, doc_ids
