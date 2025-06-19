"""
Implementation of
1. Stochastic Reranking
    @inproceedings{diaz2020evaluating,
    title={Evaluating stochastic rankings with expected exposure},
    author={Diaz, Fernando and Mitra, Bhaskar and Ekstrand, Michael D and Biega, Asia J and Carterette, Ben},
    booktitle={Proceedings of the 29th ACM international conference on information and knowledge management},
    pages={275--284},
    year={2020}
    }
    @inproceedings{oosterhuis2022learning,
    title={Learning-to-rank at the speed of sampling: Plackett-luce gradient estimation with minimal computational complexity},
    author={Oosterhuis, Harrie},
    booktitle={Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
    pages={2266--2271},
    year={2022}
    }
2. Score Regularization
    @inproceedings{diaz2005regularizing,
    title={Regularizing ad hoc retrieval scores},
    author={Diaz, Fernando},
    booktitle={Proceedings of the 14th ACM international conference on Information and knowledge management},
    pages={672--679},
    year={2005}
    }
"""

from retrievers.utils import gumbel_sample_rankings
import numpy as np
import random
from typing import Tuple

random.seed(42)


class ReRanker:
    def __init__(
        self,
        texts: list[str],
        doc_ids: list[str],
        embeddings: np.ndarray | None,
        scores: np.ndarray,
        top_k: int,
    ) -> None:
        self.texts = texts
        self.doc_ids = doc_ids
        self.embeddings = embeddings
        self.scores = scores
        self.top_k = top_k

    def rerank(self) -> Tuple[list[str], list[str]]:
        """
        Returns reranked texts and associated doc_ids
        """
        pass


#######
# Stochastic Reranking
#######


class StochasticReRanker(ReRanker):
    def __init__(
        self, texts: list[str], doc_ids: list[str], scores: np.ndarray, top_k: int
    ) -> None:
        super().__init__(texts, doc_ids, None, scores, top_k)
        self.ALPHA = 2
        self.N_SAMPLES = 50

    def rerank(self) -> Tuple[list[str], list[str]]:
        min_value = self.scores.min()
        max_value = self.scores.max()

        if min_value < 0:
            # rescale the scores to have minimum value of 0
            self.scores = self.scores - min_value
        # Min-Max Normalization, followed by scaling to [1, 2]
        self.scores = (self.scores - min_value) / (max_value - min_value)
        self.scores = self.scores + 1  # rescale to [1, 2] to make ALPHA effect bigger

        # Apply ALPHA as a temperature parameter
        self.scores = self.scores**self.ALPHA

        # Perform PL sampling with top_k
        pl_result = gumbel_sample_rankings(
            self.scores, n_samples=self.N_SAMPLES, cutoff=self.top_k, doc_prob=False
        )
        sampled_rankings = pl_result[0]
        # randomly select one ranking from the sampled rankings
        chosen_ranking = sampled_rankings[np.random.choice(sampled_rankings.shape[0])]
        # return reranked passages and document ids
        return [self.texts[rank] for rank in chosen_ranking], [
            self.doc_ids[rank] for rank in chosen_ranking
        ]


#######
# Score Regularization
#######


class ScoreRegularizationReRanker(ReRanker):
    def __init__(
        self,
        texts: list[str],
        doc_ids: list[str],
        embeddings: np.ndarray,
        scores: np.ndarray,
        top_k: int,
    ) -> None:
        super().__init__(texts, doc_ids, embeddings, scores, top_k)
        # TODO: configure t and apply
        self.top_m = int(self.embeddings.shape[0] * 1)

    def rerank(self) -> Tuple[list[str], list[str]]:
        # k x n documents matrix (k documents; n dimensional embedding)
        D = self.embeddings
        # k x k similarity matrix W = DD^T
        W = D @ D.T
        # k x k Row stochastic matrix P:
        #   for each row of W, keep the top-m similarities, then normalize to sum to 1
        P = np.zeros_like(W)
        for i in range(W.shape[0]):
            top_indices = np.argsort(W[i])[
                -self.top_m :
            ]  # Get indices of top-m similarities
            P[i, top_indices] = W[i, top_indices]  # Keep only top-m similarities
            P[i] /= P[i].sum()  # Normalize to sum to 1
        # shape the given original (k,) score vector to (kx1) yielding y
        y = self.scores.reshape(-1, 1)
        # get a new score vector y_tilde by P y
        y_tilde = P @ y
        # rerank the texts based on the y_tilde
        reranked_texts = [self.texts[i] for i in np.argsort(y_tilde.flatten())[::-1]]
        reranked_doc_ids = [
            self.doc_ids[i] for i in np.argsort(y_tilde.flatten())[::-1]
        ]

        # return top_k texts
        return reranked_texts[: self.top_k], reranked_doc_ids[: self.top_k]
