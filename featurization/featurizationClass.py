"""
Abstract class (implementation)
"""


class Featurization:
    def __init__(self, feature_names: list[str]) -> None:
        """
        feature_names: list of feature names to be featurized
        """
        self.feature_names = feature_names

    def featurize(self, query: str) -> dict:
        """Returns featurized results in a dictionary"""
        pass

    def batch_featurize(self, queries: list[str]) -> list[dict]:
        """Returns featurized results for queries in a list"""
        pass
