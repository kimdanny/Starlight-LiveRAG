"""
post-retrieval features, including cross-retriever features

- overall_sim:
    overall semantic similarity between query and the aggregated embedding of retrieved documents,
    s(q, e(z_i))
- avg_sim:
    average similarity score between query and individual retrieved documents,
    Avg_j s(q, doc_{i,j})
- max_sim:
    maximum similarity score between query and individual retrieved documents,
    Max_j s(q, doc_{i,j})
- var_sim:
    variance of retrieval similarity scores, capturing retrieval confidence dispersion,
    Var_j s(q, \doc_{i,j})
- moran:
    Moran coefficient, which measures semantic autocorrelation among retrieved documents in alignment with the cluster hypothesis,
    (y^T y_tilde) / (l2_norm(y) l2_norm(y_tilde))
- cross_retriever_sim:
    average semantic similarity of the current retriever's result set with those from other retrievers,
    1/(M-1) \sum^{M}_{m; m \ne i} s(e(z_i), e(z_m))

where,
    i           = indexing of retrievers (actions)
    j           = indexing of retrieved documents from retriever i
    k           = top k documents to retrieve
    M           = total number of retrievers
    s(. , .)    = cosine similarity
    z_i         = [\doc_{i,1}, \doc_{i,2}, ..., \doc_{i,k}]
    e(z_i)      = 1/k \sum^{k}_{j=1} embed(\doc_{i,j})
    y           = kx1 retrieval score matrix
    y_tilde     = P^T y
    P           = row stochastic W matrix
    W           = kxk document similarity matrix
"""

from featurization.featurizationClass import Featurization
from featurization.utils import batch_embed_queries, batch_embed_documents
import numpy as np


class PostRetrievalFeaturization(Featurization):
    def __init__(self, feature_names: list[str]) -> None:
        super().__init__(feature_names)
        supported_features = [
            "overall_sim",
            "avg_sim",
            "max_sim",
            "var_sim",
            "moran",
            "cross_retriever_sim",
        ]
        for fn in feature_names:
            assert fn in supported_features

        # initialize per-retriever result dict
        self.result_dict = {}
        # example self.result_dict:
        # {
        #     "bm25": {
        #         "overall_sim": None,
        #         "avg_sim": None,
        #         "max_sim": None,
        #         "var_sim": None,
        #         "moran": None,
        #         "cross_retriever_sim": None,
        #     },
        #     "bm25_stochastic": {
        #         ...
        #     },
        #     ...
        # }
        for ret in [
            "bm25",
            "bm25_stochastic",
            "bm25_regularize",
            "e5base",
            "e5base_stochastic",
            "e5base_regularize",
        ]:
            self.result_dict[ret] = {fn: None for fn in supported_features}

    def featurize(self, query: str, all_actions_docs: dict) -> dict:
        """
        Params
            query: a query string
            all_actions_docs: list of retrieved docs for all actions in order
                example of all_actions_docs:
                    {
                        "bm25": ["doc1", "doc2"],
                        "bm25_stochastic": ["doc1", "doc2"],
                    }
        """
        q_embedding = np.array(batch_embed_queries([query])[0])
        ret_avg_embedding = {}

        for ret, docs in all_actions_docs.items():
            docs_embeddings = batch_embed_documents(docs)  # k x n
            # aggregated representation of retrieved docs from a retriever i
            averaged_repre = np.mean(docs_embeddings, axis=0)  # n-dim vector
            if "cross_retriever_sim" in self.feature_names:
                ret_avg_embedding[ret] = averaged_repre

            similarity_scores_list = [
                self.cosine_similarity(q_embedding, docs_embeddings[j])
                for j in range(len(docs_embeddings))
            ]  # k-dim vector

            # s(q, e(z_i))
            if "overall_sim" in self.feature_names:
                overall_sim = self.cosine_similarity(q_embedding, averaged_repre)
                self.result_dict[ret]["overall_sim"] = float(overall_sim)

            # Avg_j s(q, doc_{i,j})
            if "avg_sim" in self.feature_names:
                avg_sim = np.mean(similarity_scores_list)
                self.result_dict[ret]["avg_sim"] = float(avg_sim)

            # Max_j s(q, doc_{i,j})
            if "max_sim" in self.feature_names:
                max_sim = np.max(similarity_scores_list)
                self.result_dict[ret]["max_sim"] = float(max_sim)

            # Var_j s(q, \doc_{i,j})
            if "var_sim" in self.feature_names:
                var_sim = np.var(similarity_scores_list)
                self.result_dict[ret]["var_sim"] = float(var_sim)

            # (y^T y_tilde) / (l2_norm(y) l2_norm(y_tilde))
            if "moran" in self.feature_names:
                # calculate y and P
                y = np.array(similarity_scores_list).reshape(-1, 1)  # k x 1
                W = docs_embeddings @ docs_embeddings.T  # k x k
                P = np.zeros_like(W)  # k x k
                for i in range(W.shape[0]):
                    top_indices = np.argsort(W[i])[-int(docs_embeddings.shape[0] * 1) :]
                    P[i, top_indices] = W[i, top_indices]
                    P[i] /= P[i].sum()  # row stochastic
                y_tilde = P @ y  # k x 1
                moran = (y_tilde.T @ y) / (
                    np.linalg.norm(y) * np.linalg.norm(y_tilde)
                )  # scalar 1x1
                self.result_dict[ret]["moran"] = moran[0][
                    0
                ]  # getting the scalar value out of the 1x1 matrix

        # 1/(M-1) \sum^{M}_{m; m \ne i} s(e(z_i), e(z_m))
        if "cross_retriever_sim" in self.feature_names:
            for ret in all_actions_docs.keys():
                sim_sum = 0
                for other_ret in all_actions_docs.keys():
                    if ret == other_ret:
                        continue
                    sim_sum += self.cosine_similarity(
                        ret_avg_embedding[ret], ret_avg_embedding[other_ret]
                    )
                cross_retr_avg_sim = sim_sum / (len(all_actions_docs) - 1)
                self.result_dict[ret]["cross_retriever_sim"] = float(cross_retr_avg_sim)

        return self.result_dict

    @staticmethod
    def cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

