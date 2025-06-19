# LTRR: Learning To Rank Retrievers for LLMs

This is an official code repository of the paper, 
[LTRR: Learning To Rank Retrievers for LLMs](https://www.arxiv.org/abs/2506.13743), presented at SIGIR 2025 LiveRAG workshop as a spotlight presentation.

**Abstract**  
Retrieval-Augmented Generation (RAG) systems typically rely on a single fixed retriever, despite growing evidence that no single retriever performs optimally across all query types. In this paper, we explore a query routing approach that dynamically selects from a pool of retrievers based on the query, using both train-free heuristics and learned routing models. We frame routing as a learning-to-rank (LTR) problem and introduce LTRR, a framework that learns to rank retrievers by their expected utility gain to downstream LLM performance. Our experiments, conducted on synthetic QA data with controlled query type variations, show that routing-based RAG systems can outperform the best single-retriever-based systems. Performance gains are especially pronounced in models trained with the Answer Correctness (AC) metric and with pairwise learning approaches, especially with XGBoost. We also observe improvements in generalization to out-of-distribution queries. As part of the SIGIR 2025 LiveRAG challenge, our submitted system demonstrated the practical viability of our approach, achieving competitive performance in both answer correctness and faithfulness. These findings highlight the importance of both training methodology and metric selection in query routing for RAG systems.




## Reference
If you find our research valuable, please consider citing it as follows:
```
@misc{kim2025ltrrlearningrankretrievers,
      title={LTRR: Learning To Rank Retrievers for LLMs}, 
      author={To Eun Kim and Fernando Diaz},
      year={2025},
      eprint={2506.13743},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.13743}, 
}
```
