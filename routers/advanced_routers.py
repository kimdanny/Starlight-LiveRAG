"""
Load routers from trained-routers directory
"""

from routers.RouterClass import Router
import os
import pandas as pd
import numpy as np
from typing import Dict
import xgboost as xgb
from sklearn.impute import SimpleImputer


class PairwiseXGBoostRouter(Router):
    def __init__(self, model_dir: str, router_test_fp: str, M: int = 7) -> None:
        super().__init__(M)
        self.ltrr_model = PairwiseTreeLTRR(model_dir=model_dir)
        self.router_test_fp = router_test_fp

    def batch_route(self) -> list[int]:
        test_data = pd.read_csv(self.router_test_fp)
        self.ltrr_model.load_model()
        predicted_rankings = self.ltrr_model.get_rankings(test_data)
        routing_results = [
            int(list(action_ranking)[0])
            for qid, action_ranking in predicted_rankings.items()
        ]
        return routing_results


class BaseLTRR:
    """Base class for Learning to Rank Retriever models"""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def load_model(self, load_path: str):
        raise NotImplementedError

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def get_rankings(self, test_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get rankings in pairwise fashion
        Get rankings of actions for each query.
        Returns a dictionary mapping qid to ranked array of actions.
        """
        predictions = self.predict(test_data)

        # Pairwise: predictions are preference scores (continuous values, not binary)
        rankings = {}
        for qid in test_data["qid"].unique():
            qid_data = test_data[test_data["qid"] == qid]
            qid_preds = predictions[qid_data.index]

            # Get all unique actions from both action_A and action_B
            all_actions = sorted(
                set(qid_data["action_A"].unique()) | set(qid_data["action_B"].unique())
            )

            # Initialize weighted Borda scores for each action
            borda_scores = {action: 0.0 for action in all_actions}

            # Keep track of how many comparisons each action is involved in
            comparison_counts = {action: 0 for action in all_actions}

            # Count weighted wins for each action based on confidence
            for i, ((a, b), pred) in enumerate(
                zip(zip(qid_data["action_A"], qid_data["action_B"]), qid_preds)
            ):
                # Get confidence score (normalized to [0, 1])
                confidence = (pred + 1) / 2  # Convert from [-1, 1] to [0, 1]

                # Add weighted scores
                if confidence > 0.5:  # A is preferred
                    borda_scores[a] += confidence
                    borda_scores[b] += 1 - confidence
                else:  # B is preferred
                    borda_scores[a] += confidence
                    borda_scores[b] += 1 - confidence

                # Track comparison counts
                comparison_counts[a] += 1
                comparison_counts[b] += 1

            # Normalize scores by number of comparisons
            for action in all_actions:
                if comparison_counts[action] > 0:
                    borda_scores[action] /= comparison_counts[action]

            # For this query, print the scores for debugging
            print(
                f"QID {qid} scores: {' '.join([f'{a}:{borda_scores[a]:.2f}' for a in sorted(all_actions)])}"
            )

            # Sort actions by their Borda scores
            sorted_actions = sorted(
                all_actions, key=lambda x: borda_scores[x], reverse=True
            )
            rankings[qid] = np.array(sorted_actions)

        # Track and report ranking diversity
        unique_rankings = {}
        for qid, ranking in rankings.items():
            ranking_tuple = tuple(ranking)
            if ranking_tuple not in unique_rankings:
                unique_rankings[ranking_tuple] = []
            unique_rankings[ranking_tuple].append(qid)

        print(f"\nRanking diversity statistics:")
        print(f"  Total number of queries: {len(rankings)}")
        print(f"  Number of unique rankings: {len(unique_rankings)}")

        # Show the most common rankings
        print("\nMost common rankings:")
        most_common = sorted(
            unique_rankings.items(), key=lambda x: len(x[1]), reverse=True
        )
        for i, (ranking, qids) in enumerate(most_common[:5]):  # Show top 5
            print(
                f"  Ranking {i+1}: {ranking} - {len(qids)} queries ({len(qids)/len(rankings)*100:.1f}%)"
            )

        return rankings


class PairwiseTreeLTRR(BaseLTRR):
    """Pairwise Learning to Rank using XGBoost"""

    def __init__(self, model_dir: str, **kwargs):
        super().__init__(model_dir)
        self.model = xgb.XGBClassifier(**kwargs)
        self.imputer = SimpleImputer(strategy="median")

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract features from pairwise dataset with action-specific columns.
        We need to:
        1. Correctly identify A and B feature columns
        2. Create meaningful feature differences
        """
        # Drop the question column if it exists
        data = data.copy()
        if "question" in data.columns:
            data = data.drop(columns=["question"])

        # Find all action-specific feature columns (that end with _A or _B)
        a_cols = [
            col for col in data.columns if col.endswith("_A") and col != "action_A"
        ]
        b_cols = [
            col for col in data.columns if col.endswith("_B") and col != "action_B"
        ]

        # Map A columns to corresponding B columns
        features = []
        for a_col in a_cols:
            base_name = a_col[:-2]  # Remove "_A" suffix
            b_col = f"{base_name}_B"
            if b_col in b_cols:
                # Feature difference (A-B)
                diff_values = data[a_col].values - data[b_col].values
                features.append(diff_values)

        # Add shared features that don't have A/B variants
        base_cols = [
            col
            for col in data.columns
            if col not in ["qid", "action_A", "action_B", "preference_label_bool"]
            and not col.endswith("_A")
            and not col.endswith("_B")
        ]

        for col in base_cols:
            features.append(data[col].values)

        # Stack features horizontally into a matrix
        X = np.column_stack(features)

        # Print feature stats
        print(f"Feature matrix shape: {X.shape}")
        print(
            f"Feature statistics - min: {X.min():.4f}, max: {X.max():.4f}, mean: {X.mean():.4f}, std: {X.std():.4f}"
        )

        # Check for NaN values and impute
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            print(f"Found {nan_count} NaN values in feature matrix, imputing...")
            X = self.imputer.fit_transform(X)

        return X

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        X = self._prepare_features(test_data)

        # Get raw probabilities for more nuanced predictions
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            # Use class 1 probability as a confidence score (higher = more confident that A > B)
            preds_confidence = probs[:, 1]

            # Print diagnostics about prediction distribution
            print(
                f"Prediction confidence - min: {preds_confidence.min():.4f}, max: {preds_confidence.max():.4f}"
            )
            print(
                f"Prediction confidence - mean: {preds_confidence.mean():.4f}, std: {preds_confidence.std():.4f}"
            )

            # Check if predictions are mostly extreme (close to 0 or 1)
            extreme_preds = np.sum((preds_confidence < 0.1) | (preds_confidence > 0.9))
            print(
                f"Extreme predictions (< 0.1 or > 0.9): {extreme_preds} ({extreme_preds/len(preds_confidence)*100:.1f}%)"
            )

            # Use confidence scores to create more nuanced predictions
            # Instead of just -1 or 1, use confidence level for more granular Borda counting
            return 2 * preds_confidence - 1  # Scale from [0,1] to [-1,1]
        else:
            # Fallback to binary predictions if predict_proba is not available
            print(
                "WARNING: Model does not support probability estimates, using binary predictions"
            )
            preds = self.model.predict(X)

            # Print distribution of binary predictions
            unique_preds, counts = np.unique(preds, return_counts=True)
            print(f"Binary prediction distribution: {dict(zip(unique_preds, counts))}")

            return 2 * preds - 1

    def load_model(self):
        model_path = os.path.join(self.model_dir, f"xgboost.json")
        self.model.load_model(model_path)
