"""
Basic routers that use a post-retrieval feature as a strategy
"""

from routers.RouterClass import Router
from featurization.post_retrieval import PostRetrievalFeaturization
from retrievers.query_all_retrievers import query_all_retrievers
from retrievers.OpenSearchConnection import OpenSearchConnection
from retrievers.PineconeConnection import PineconeConnection

oc = OpenSearchConnection()
pc = PineconeConnection()

ret_name_to_id = {
    "bm25": 1,
    "bm25_stochastic": 2,
    "bm25_regularize": 3,
    "e5base": 4,
    "e5base_stochastic": 5,
    "e5base_regularize": 6,
}


def _query_all_retrievers(question: str) -> dict:
    """
    Query all retrievers and return the results
    """
    res = query_all_retrievers(question, oc, pc)
    # match the format
    all_actions_docs = {}
    for ret in res.keys():
        all_actions_docs[ret] = res[ret]["texts"]
    return all_actions_docs


class OverallSimRouter(Router):
    def __init__(self, M: int = 7) -> None:
        super().__init__(M)
        self.feature_names = ["overall_sim"]
        self.prf = PostRetrievalFeaturization(self.feature_names)

    def batch_route(self, questions: list[str]) -> list[int]:
        routing_results = []
        for question in questions:
            all_actions_docs = _query_all_retrievers(question)
            routing_result = self.prf.featurize(question, all_actions_docs)
            # example routing result:
            # {
            #   "bm25": {"overall_sim": 0.54},
            #   "bm25_stochastic": {"overall_sim": 0.45},
            #     ...
            # }
            max_val = float("-inf")
            max_ret = ""
            for ret, prf_result in routing_result.items():
                if prf_result["overall_sim"] > max_val:
                    max_val = prf_result["overall_sim"]
                    max_ret = ret
            routing_results.append(ret_name_to_id[max_ret])
        return routing_results


class AvgSimRouter(Router):
    def __init__(self, M: int = 7) -> None:
        super().__init__(M)
        self.feature_names = ["avg_sim"]
        self.prf = PostRetrievalFeaturization(self.feature_names)

    def batch_route(self, questions: list[str]) -> list[int]:
        routing_results = []
        for question in questions:
            all_actions_docs = _query_all_retrievers(question)
            routing_result = self.prf.featurize(question, all_actions_docs)
            # example routing result:
            # {
            #   "bm25": {"avg_sim": 0.54},
            #   "bm25_stochastic": {"avg_sim": 0.45},
            #     ...
            # }
            max_val = float("-inf")
            max_ret = ""
            for ret, prf_result in routing_result.items():
                if prf_result["avg_sim"] > max_val:
                    max_val = prf_result["avg_sim"]
                    max_ret = ret
            routing_results.append(ret_name_to_id[max_ret])
        return routing_results


class MaxSimRouter(Router):
    def __init__(self, M: int = 7) -> None:
        super().__init__(M)
        self.feature_names = ["max_sim"]
        self.prf = PostRetrievalFeaturization(self.feature_names)

    def batch_route(self, questions: list[str]) -> list[int]:
        routing_results = []
        for question in questions:
            all_actions_docs = _query_all_retrievers(question)
            routing_result = self.prf.featurize(question, all_actions_docs)
            # example routing result:
            # {
            #   "bm25": {"max_sim": 0.54},
            #   "bm25_stochastic": {"max_sim": 0.45},
            #     ...
            # }
            max_val = float("-inf")
            max_ret = ""
            for ret, prf_result in routing_result.items():
                if prf_result["max_sim"] > max_val:
                    max_val = prf_result["max_sim"]
                    max_ret = ret
            routing_results.append(ret_name_to_id[max_ret])
        return routing_results


class VarSimRouter(Router):
    def __init__(self, M: int = 7) -> None:
        super().__init__(M)
        self.feature_names = ["var_sim"]
        self.prf = PostRetrievalFeaturization(self.feature_names)

    def batch_route(self, questions: list[str]) -> list[int]:
        routing_results = []
        for question in questions:
            all_actions_docs = _query_all_retrievers(question)
            routing_result = self.prf.featurize(question, all_actions_docs)
            # example routing result:
            # {
            #   "bm25": {"var_sim": 0.54},
            #   "bm25_stochastic": {"var_sim": 0.45},
            #     ...
            # }
            min_val = float("inf")
            min_ret = ""
            for ret, prf_result in routing_result.items():
                if prf_result["var_sim"] < min_val:
                    min_val = prf_result["var_sim"]
                    min_ret = ret
            routing_results.append(ret_name_to_id[min_ret])
        return routing_results


class MoranRouter(Router):
    def __init__(self, M: int = 7) -> None:
        super().__init__(M)
        self.feature_names = ["moran"]
        self.prf = PostRetrievalFeaturization(self.feature_names)

    def batch_route(self, questions: list[str]) -> list[int]:
        routing_results = []
        for question in questions:
            all_actions_docs = _query_all_retrievers(question)
            routing_result = self.prf.featurize(question, all_actions_docs)
            # example routing result:
            # {
            #   "bm25": {"moran": 0.54},
            #   "bm25_stochastic": {"moran": 0.45},
            #     ...
            # }
            max_val = float("-inf")
            max_ret = ""
            for ret, prf_result in routing_result.items():
                if prf_result["moran"] > max_val:
                    max_val = prf_result["moran"]
                    max_ret = ret
            routing_results.append(ret_name_to_id[max_ret])
        return routing_results
