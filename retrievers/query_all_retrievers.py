"""
Send a single query to all retrievers
"""

from retrievers.individual_retrievers import bm25, e5base
from retrievers.utils import batch_embed_documents
from retrievers.OpenSearchConnection import OpenSearchConnection
from retrievers.PineconeConnection import PineconeConnection
from retrievers.ReRankers import StochasticReRanker, ScoreRegularizationReRanker

cache = {
    "bm25": {"texts": ..., "doc_ids": ..., "embeddings": ..., "scores": ...},
    "e5base": {"texts": ..., "doc_ids": ..., "embeddings": ..., "scores": ...},
}


def query_all_retrievers(
    query: str,
    oc: OpenSearchConnection,
    pc: PineconeConnection,
    first_stage_config={"top_k": 50},
    final_top_k: int = 5,
) -> dict:
    """
    Simulation of uncooperative retrieval environment, where we only get retrieved passages.
    No score, corpus, embedding related information is obtained.
    Associated Doc ID is obtained (for LiveRAG challenge purpose).
    """
    results = {
        "bm25": {"texts": [""], "doc_ids": [""]},
        "bm25_stochastic": {"texts": [""], "doc_ids": [""]},
        "bm25_regularize": {"texts": [""], "doc_ids": [""]},
        "e5base": {"texts": [""], "doc_ids": [""]},
        "e5base_stochastic": {"texts": [""], "doc_ids": [""]},
        "e5base_regularize": {"texts": [""], "doc_ids": [""]},
    }
    # Update cache with bm25 and e5base results
    _query_bm25(query, oc, first_stage_config)
    _query_e5base(query, pc, first_stage_config)
    # Retriever 1
    results["bm25"]["texts"] = cache["bm25"]["texts"][:final_top_k]
    results["bm25"]["doc_ids"] = cache["bm25"]["doc_ids"][:final_top_k]

    # Retriever 2
    texts, doc_ids = StochasticReRanker(
        texts=cache["bm25"]["texts"],
        doc_ids=cache["bm25"]["doc_ids"],
        scores=cache["bm25"]["scores"],
        top_k=final_top_k,
    ).rerank()
    results["bm25_stochastic"]["texts"] = texts
    results["bm25_stochastic"]["doc_ids"] = doc_ids

    # Retriever 3
    texts, doc_ids = ScoreRegularizationReRanker(
        texts=cache["bm25"]["texts"],
        doc_ids=cache["bm25"]["doc_ids"],
        embeddings=cache["bm25"]["embeddings"],
        scores=cache["bm25"]["scores"],
        top_k=final_top_k,
    ).rerank()
    results["bm25_regularize"]["texts"] = texts
    results["bm25_regularize"]["doc_ids"] = doc_ids

    # Retriever 4
    results["e5base"]["texts"] = cache["e5base"]["texts"][:final_top_k]
    results["e5base"]["doc_ids"] = cache["e5base"]["doc_ids"][:final_top_k]

    # Retriever 5
    texts, doc_ids = StochasticReRanker(
        texts=cache["e5base"]["texts"],
        doc_ids=cache["e5base"]["doc_ids"],
        scores=cache["e5base"]["scores"],
        top_k=final_top_k,
    ).rerank()
    results["e5base_stochastic"]["texts"] = texts
    results["e5base_stochastic"]["doc_ids"] = doc_ids

    # Retriever 6
    texts, doc_ids = ScoreRegularizationReRanker(
        texts=cache["e5base"]["texts"],
        doc_ids=cache["e5base"]["doc_ids"],
        embeddings=cache["e5base"]["embeddings"],
        scores=cache["e5base"]["scores"],
        top_k=final_top_k,
    ).rerank()

    results["e5base_regularize"]["texts"] = texts
    results["e5base_regularize"]["doc_ids"] = doc_ids

    return results


def _query_bm25(
    query: str, oc: OpenSearchConnection, first_stage_config={"top_k": 50}
) -> None:
    texts, doc_ids, scores = bm25(query, oc, first_stage_config, return_extras=True)
    embeddings = batch_embed_documents(texts, device="cuda")
    # update cache
    cache["bm25"]["texts"] = texts
    cache["bm25"]["doc_ids"] = doc_ids
    cache["bm25"]["embeddings"] = embeddings
    cache["bm25"]["scores"] = scores


def _query_e5base(
    query: str, pc: PineconeConnection, first_stage_config={"top_k": 50}
) -> None:
    texts, doc_ids, embeddings, scores = e5base(
        query, pc, first_stage_config, return_extras=True
    )
    # update cache
    cache["e5base"]["texts"] = texts
    cache["e5base"]["doc_ids"] = doc_ids
    cache["e5base"]["embeddings"] = embeddings
    cache["e5base"]["scores"] = scores
