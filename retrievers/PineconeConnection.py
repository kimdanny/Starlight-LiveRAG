from typing import List, Literal, Tuple
from multiprocessing.pool import ThreadPool
from pinecone import Pinecone
import torch
import numpy as np
from functools import cache
from transformers import AutoModel, AutoTokenizer
from aws.aws_connection import get_ssm_secret


class PineconeConnection:
    """
    Connection to dense retrieval through Pinecone Fineweb Index hosted on LiveRAG AWS
    """

    def __init__(
        self,
        namespace: str = "default",
        index_name: str = "fineweb10bt-512-0w-e5-base-v2",
        embedding_model_name: str = "intfloat/e5-base-v2",
    ) -> None:
        self.namespace = namespace
        self.index_name = index_name
        self.model_name = embedding_model_name

    @cache
    def has_mps(self):
        return torch.backends.mps.is_available()

    @cache
    def has_cuda(self):
        return torch.cuda.is_available()

    @cache
    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return tokenizer

    @cache
    def get_model(self):
        model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        if self.has_mps():
            model = model.to("mps")
        elif self.has_cuda():
            model = model.to("cuda")
        else:
            model = model.to("cpu")
        return model

    @cache
    def get_pinecone_index(self):
        pc = Pinecone(api_key=get_ssm_secret("/pinecone/ro_token"))
        index = pc.Index(name=self.index_name)
        return index

    def average_pool(
        self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def embed_query(
        self,
        query: str,
        query_prefix: str = "query: ",
        pooling: Literal["cls", "avg"] = "avg",
        normalize: bool = True,
    ) -> list[float]:
        return self.batch_embed_queries([query], query_prefix, pooling, normalize)[0]

    def batch_embed_queries(
        self,
        queries: List[str],
        query_prefix: str = "query: ",
        pooling: Literal["cls", "avg"] = "avg",
        normalize: bool = True,
    ) -> List[List[float]]:
        with_prefixes = [" ".join([query_prefix, query]) for query in queries]
        tokenizer = self.get_tokenizer()
        model = self.get_model()
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
                    embeddings = self.average_pool(
                        model_out.last_hidden_state, encoded["attention_mask"]
                    )
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.tolist()

    def query_pinecone(self, query: str, top_k: int = 10) -> dict:
        index = self.get_pinecone_index()
        results = index.query(
            vector=self.embed_query(query),
            top_k=top_k,
            include_values=True,
            namespace=self.namespace,
            include_metadata=True,
        )

        return results

    def batch_query_pinecone(
        self,
        queries: list[str],
        top_k: int = 10,
        n_parallel: int = 10,
    ) -> list[dict]:
        """Batch query a Pinecone index and return the results.

        Internally uses a ThreadPool to parallelize the queries.
        """
        index = self.get_pinecone_index()
        embeds = self.batch_embed_queries(queries)
        pool = ThreadPool(n_parallel)
        results = pool.map(
            lambda x: index.query(
                vector=x,
                top_k=top_k,
                include_values=True,
                namespace=self.namespace,
                include_metadata=True,
            ),
            embeds,
        )
        return results

    def get_texts_and_scores(self, matches: list) -> Tuple[list[str], np.ndarray]:
        """
        Given the matches of the retrieval results,
        returns the following:
            texts: list of retrieved chunks
            scores: (k,) dimensional numpy array of retrieval scores
        """
        texts = [match["metadata"]["text"] for match in matches]
        scores = [match["score"] for match in matches]
        return texts, np.array(scores)

    def get_texts_docids_embeddings_and_scores(
        self, matches: list
    ) -> Tuple[list[str], list[str], np.ndarray, np.ndarray]:
        """
        Given the matches of the retrieval results,
        returns the following:
            texts: list of retrieved chunks
            embeddings: k x n matrix / ndarray (k documents; n dimensional embedding)
            scores: (k,) dimensional numpy array of retrieval scores
        """
        texts = [match["metadata"]["text"] for match in matches]
        doc_ids = [match["metadata"]["doc_id"] for match in matches]
        embeddings = [match["values"] for match in matches]
        scores = [match["score"] for match in matches]
        return texts, doc_ids, np.array(embeddings), np.array(scores)

    def show_pinecone_results(self, results):
        for match in results["matches"]:
            print("chunk:", match["id"], "score:", match["score"])
            print("total_doc_chunks:", match["metadata"]["total_doc_chunks"])
            print(match["metadata"]["text"])
            print()
