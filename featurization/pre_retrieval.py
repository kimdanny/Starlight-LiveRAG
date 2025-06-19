"""
pre-retrieval features

- query_embedding (R^n)
- query_length (R)
- query_type (Boolean)
    If query is a keyword- or natural-language-based.
    https://huggingface.co/shahrukhx01/bert-mini-finetune-question-detection
"""

from featurization.featurizationClass import Featurization
from featurization.utils import batch_embed_queries
from haystack.components.routers import TransformersTextRouter


class PreRetrievalFeaturization(Featurization):
    def __init__(self, feature_names: list[str]) -> None:
        super().__init__(feature_names)
        supported_features = ["query_embedding", "query_length", "query_type"]
        for fn in feature_names:
            assert fn in supported_features

        if "query_type" in feature_names:
            self.ttr = TransformersTextRouter(
                model="shahrukhx01/bert-mini-finetune-question-detection"
            )
            self.ttr.warm_up()

        # initialize per-query result dict
        # this per-query result should be the same
        #   across all retrievers when creating router training dataset
        self.result_dict = {}
        for fn in feature_names:
            self.result_dict[fn] = None

    def featurize(self, query: str) -> dict:
        if "query_embedding" in self.feature_names:
            embedding: list[float] = batch_embed_queries([query])[0]
            self.result_dict["query_embedding"] = embedding

        if "query_length" in self.feature_names:
            query_len: float = self._batch_get_query_length([query])[0]
            self.result_dict["query_length"] = query_len

        if "query_type" in self.feature_names:
            query_type: float = self._batch_classify_query_type([query])[0]
            self.result_dict["query_type"] = query_type

        return self.result_dict

    def batch_featurize(self, queries: list[str]) -> list[dict]:
        if "query_embedding" in self.feature_names:
            embedding: list[list[float]] = batch_embed_queries(queries)
            self.result_dict["query_embedding"] = embedding

        if "query_length" in self.feature_names:
            query_len: list[float] = self._batch_get_query_length(queries)
            self.result_dict["query_length"] = query_len

        if "query_type" in self.feature_names:
            query_type: list[float] = self._batch_classify_query_type(queries)
            self.result_dict["query_type"] = query_type

        return self.result_dict

    # Query length
    def _batch_get_query_length(self, queries: list[str]) -> list[float]:
        """
        Word length
        During training, this should be batch normalized
        """
        return [float(len(q.split())) for q in queries]

    # Query type classification
    def _batch_classify_query_type(self, queries: list[str]) -> list[float]:
        """
        Classify the complexity of a batch of queries.
        Returns a list of labels:
            0.0 for keyword-based queries
            1.0 for natural language queries
        """
        results = []
        for query in queries:
            result = self.ttr.run(text=query)
            label = next(iter(result))
            results.append(
                float(label.split("_")[-1])
            )  # Extract numeric label from 'LABEL_0' or 'LABEL_1'
        return results


