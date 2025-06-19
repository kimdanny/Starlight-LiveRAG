"""
Implementations of "Learning to Rank Retrievers (LTRR) for LLMs"
Load trained routers and only peforform inference here
Routers should be able to perform batch inference
"""


class Router:
    def __init__(self, M: int = 7) -> None:
        self.M = M  # number of total actions including no-retrieval
        pass

    def batch_route(self, questions: list[str]) -> list[int]:
        """
        returns a list of action IDs in int
        """
        pass
