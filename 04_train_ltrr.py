import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.impute import SimpleImputer
import xgboost as xgb
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F
import pickle
import argparse
from scipy.sparse.linalg import eigsh
from scipy.stats import kendalltau
from tqdm import tqdm
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import GridSearchCV, StratifiedKFold

torch.manual_seed(42)


class BaseLTRR:
    """Base class for Learning to Rank Retriever models"""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def save_model(self, save_path: str):
        raise NotImplementedError

    def load_model(self, load_path: str):
        raise NotImplementedError

    def train(self, train_data: pd.DataFrame):
        raise NotImplementedError

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def get_rankings(self, test_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get rankings of actions for each query.
        Returns a dictionary mapping qid to ranked array of actions.
        """
        predictions = self.predict(test_data)

        if isinstance(
            self,
            (
                PointwiseTreeLTRR,
                PointwiseSVMLTRR,
                PointwiseNeuralLTRR,
                PointwiseDeBERTaLTRR,
            ),
        ):
            # Pointwise: predictions are delta values
            rankings = {}
            for qid in test_data["qid"].unique():
                qid_data = test_data[test_data["qid"] == qid]
                qid_preds = predictions[qid_data.index]
                rankings[qid] = qid_data["action"].iloc[np.argsort(-qid_preds)].values

        elif isinstance(
            self,
            (
                PairwiseTreeLTRR,
                PairwiseSVMLTRR,
                PairwiseNeuralLTRR,
                PairwiseDeBERTaLTRR,
            ),
        ):
            # Pairwise: predictions are preference scores (continuous values, not binary)
            rankings = {}
            for qid in test_data["qid"].unique():
                qid_data = test_data[test_data["qid"] == qid]
                qid_preds = predictions[qid_data.index]

                # Get all unique actions from both action_A and action_B
                all_actions = sorted(
                    set(qid_data["action_A"].unique())
                    | set(qid_data["action_B"].unique())
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

        else:  # Listwise models
            # Listwise: predictions are scores for all actions
            rankings = {}
            i = 0
            for qid in test_data["qid"].unique():
                # For LambdaMART with action scorers, add a tiny query-specific noise
                if isinstance(self, ListwiseLambdaMARTLTRR):
                    # Set a query-specific seed for reproducibility
                    qid_numeric = (
                        int(qid.split("-")[-1]) if "-" in str(qid) else hash(str(qid))
                    )
                    np.random.seed(qid_numeric % 1000000)
                    # Add small random noise (0.1%) to break ties differently for each query
                    pred = predictions[i].copy()
                    noise = np.random.normal(0, 0.001, size=pred.shape)
                    pred += noise
                    rankings[qid] = np.argsort(-pred)
                else:
                    # Regular ListNet model
                    rankings[qid] = np.argsort(-predictions[i])
                i += 1

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


class PointwiseTreeLTRR(BaseLTRR):
    """Pointwise Learning to Rank using XGBoost"""

    def __init__(self, model_dir: str, **kwargs):
        super().__init__(model_dir)
        self.model = xgb.XGBRegressor(**kwargs)
        self.imputer = SimpleImputer(strategy="median")

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        # Drop the question column if it exists
        data = data.copy()
        if "question" in data.columns:
            data = data.drop(columns=["question"])

        # Get all feature columns except qid, action, and delta_label
        feature_cols = [
            col for col in data.columns if col not in ["qid", "action", "delta_label"]
        ]
        X = data[feature_cols].values
        # Impute NaN values with median
        X = self.imputer.fit_transform(X)
        return X

    def train(self, train_data: pd.DataFrame):
        X = self._prepare_features(train_data)
        y = train_data["delta_label"].values
        self.model.fit(X, y)

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        X = self._prepare_features(test_data)
        return self.model.predict(X)

    def save_model(self, save_path: str):
        model_path = os.path.join(self.model_dir, f"{save_path}.json")
        self.model.save_model(model_path)

    def load_model(self, load_path: str):
        model_path = os.path.join(self.model_dir, f"{load_path}.json")
        self.model.load_model(model_path)


class PointwiseSVMLTRR(BaseLTRR):
    """Pointwise Learning to Rank using SVM"""

    def __init__(self, model_dir: str, **kwargs):
        super().__init__(model_dir)
        self.model = SVR(**kwargs)
        self.imputer = SimpleImputer(strategy="median")

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        # Drop the question column if it exists
        data = data.copy()
        if "question" in data.columns:
            data = data.drop(columns=["question"])

        feature_cols = [
            col for col in data.columns if col not in ["qid", "action", "delta_label"]
        ]
        X = data[feature_cols].values
        X = self.imputer.fit_transform(X)
        return X

    def train(self, train_data: pd.DataFrame):
        X = self._prepare_features(train_data)
        y = train_data["delta_label"].values
        self.model.fit(X, y)

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        X = self._prepare_features(test_data)
        return self.model.predict(X)

    def save_model(self, save_path: str):
        model_path = os.path.join(self.model_dir, f"{save_path}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, load_path: str):
        model_path = os.path.join(self.model_dir, f"{load_path}.pkl")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)


class PointwiseNeuralLTRR(BaseLTRR):
    """Pointwise Learning to Rank using Neural Network"""

    class PointwiseDataset(Dataset):
        def __init__(self, data: pd.DataFrame):
            # Drop the question column if it exists
            data = data.copy()
            if "question" in data.columns:
                data = data.drop(columns=["question"])

            self.features = torch.FloatTensor(
                data.drop(["qid", "action", "delta_label"], axis=1).values
            )
            self.labels = torch.FloatTensor(data["delta_label"].values)
            # Create mask for NaN values
            self.mask = torch.isnan(self.features)
            # Initialize learnable parameters for NaN values
            self.nan_params = nn.Parameter(torch.randn(self.features.shape[1]))

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            features = self.features[idx].clone()
            # Replace NaN values with learnable parameters
            features[self.mask[idx]] = self.nan_params[self.mask[idx]]
            return features, self.labels[idx]

    class PointwiseModel(nn.Module):
        def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64]):
            super().__init__()
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.extend(
                    [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)]
                )
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x).squeeze()

    def __init__(
        self,
        model_dir: str,
        hidden_dims: List[int] = [256, 128, 64],
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ):
        super().__init__(model_dir)
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, train_data: pd.DataFrame, num_epochs: int = 100):
        dataset = self.PointwiseDataset(train_data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = self.PointwiseModel(
            input_dim=len(train_data.columns) - 3,  # exclude qid, action, delta_label
            hidden_dims=self.hidden_dims,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        for epoch in tqdm(range(num_epochs), desc="Training epochs"):
            total_loss = 0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
            for features, labels in pbar:
                features, labels = features.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(features)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}"
                )

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        dataset = self.PointwiseDataset(test_data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for features, _ in dataloader:
                features = features.to(self.device)
                preds = self.model(features)
                predictions.extend(preds.cpu().numpy())
        return np.array(predictions)

    def save_model(self, save_path: str):
        model_path = os.path.join(self.model_dir, f"{save_path}.pt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "hidden_dims": self.hidden_dims,
            },
            model_path,
        )

    def load_model(self, load_path: str):
        model_path = os.path.join(self.model_dir, f"{load_path}.pt")
        checkpoint = torch.load(model_path, weights_only=True)

        # Extract dimensions from the checkpoint state dict
        state_dict = checkpoint["model_state_dict"]
        input_dim = state_dict["network.0.weight"].shape[1]
        hidden_dims = checkpoint["hidden_dims"]

        # Create the model with correct dimensions first
        self.model = self.PointwiseModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)

        # Then load the state dict
        self.model.load_state_dict(state_dict)


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

    def train(self, train_data: pd.DataFrame, do_hyperparameter_tuning: bool = True):
        X = self._prepare_features(train_data)
        # Convert preference labels from [-1, 1] to [0, 1]
        y = (train_data["preference_label_bool"].values + 1) // 2

        # Print class distribution
        unique_labels, counts = np.unique(y, return_counts=True)
        print(f"Class distribution: {dict(zip(unique_labels, counts))}")

        # Define base hyperparameters for ranking problems
        base_params = {
            "objective": "binary:logistic",  # For better probability calibration
            "eval_metric": "auc",  # Area under ROC curve
            "learning_rate": 0.05,  # Lower learning rate for better generalization
            "max_depth": 5,  # Control model complexity
            "min_child_weight": 2,  # Help prevent overfitting
            "gamma": 0.1,  # Minimum loss reduction for partition
            "subsample": 0.8,  # Use 80% of data per tree (prevents overfitting)
            "colsample_bytree": 0.8,  # Use 80% of features per tree (feature diversity)
            "scale_pos_weight": 1.0,  # Balance class weights if needed
            "random_state": 42,  # For reproducibility
        }

        if do_hyperparameter_tuning:
            print("Starting hyperparameter tuning...")
            # Create a base model with fixed parameters
            base_model = xgb.XGBClassifier(
                objective="binary:logistic", eval_metric="auc", random_state=42
            )

            # Define parameter grid for tuning
            param_grid = {
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
                "min_child_weight": [1, 2, 3],
                "gamma": [0, 0.1, 0.2],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.9],
                "scale_pos_weight": [1.0],
            }

            # Create a smaller subset for tuning if dataset is large
            if len(X) > 10000:
                from sklearn.model_selection import train_test_split

                X_sample, _, y_sample, _ = train_test_split(
                    X, y, test_size=0.7, random_state=42, stratify=y
                )
                print(f"Using {len(X_sample)} samples for hyperparameter tuning")
            else:
                X_sample, y_sample = X, y

            # Define cross-validation strategy
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

            # Create GridSearchCV object
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
                verbose=1,
            )

            # Fit the grid search
            grid_search.fit(X_sample, y_sample)

            # Get best parameters
            best_params = grid_search.best_params_
            print(f"Best hyperparameters: {best_params}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

            # Update base parameters with best parameters
            base_params.update(best_params)

        # Create a new model with the optimized parameters
        self.model = xgb.XGBClassifier(**base_params)

        # Train the model on the full dataset
        print(f"Training XGBoost model with {X.shape[1]} features...")
        self.model.fit(X, y, verbose=True)

        # Get feature importance
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            print("\nTop 10 most important features:")
            # Get indices of top features
            top_indices = np.argsort(importances)[-10:]
            for idx in reversed(top_indices):
                print(f"  Feature {idx}: {importances[idx]:.6f}")

        # Evaluate on training set
        train_preds = self.model.predict(X)
        train_acc = np.mean(train_preds == y)
        print(f"Training accuracy: {train_acc:.4f}")

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

    def save_model(self, save_path: str):
        model_path = os.path.join(self.model_dir, f"{save_path}.json")
        self.model.save_model(model_path)

    def load_model(self, load_path: str):
        model_path = os.path.join(self.model_dir, f"{load_path}.json")
        self.model.load_model(model_path)


class PairwiseSVMLTRR(BaseLTRR):
    """Pairwise Learning to Rank using SVM"""

    def __init__(self, model_dir: str, **kwargs):
        super().__init__(model_dir)
        # Default parameters optimized for ranking
        default_params = {
            "C": 1.0,  # Higher C to give more weight to training examples
            "loss": "squared_hinge",  # Smooth loss function for better gradients
            "max_iter": 10000,  # Increase max iterations significantly
            "class_weight": None,  # We'll handle this manually
            "dual": True,  # Dual formulation is better for n_samples > n_features
            "tol": 1e-5,  # Tighter tolerance
            "random_state": 42,  # For reproducibility
            "fit_intercept": True,  # Include intercept term
        }

        # Update defaults with any provided kwargs
        default_params.update(kwargs)

        # Use LinearSVC instead of SVC for faster training
        self.model = LinearSVC(**default_params)
        self.imputer = SimpleImputer(strategy="median")

        # Add a scaler for feature normalization
        self.scaler = None
        self.feature_cols = None

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract features from pairwise dataset with action-specific columns.
        For each feature, we'll calculate differences (A-B) and also include raw features.
        """
        # Drop the question column if it exists
        data = data.copy()
        if "question" in data.columns:
            data = data.drop(columns=["question"])

        # Find all action-specific feature columns (that end with _A or _B)
        a_cols = [
            col
            for col in data.columns
            if col.endswith("_A") and col not in ["action_A"]
        ]
        b_cols = [
            col
            for col in data.columns
            if col.endswith("_B") and col not in ["action_B"]
        ]

        # Map A columns to corresponding B columns
        ab_col_pairs = []
        for a_col in a_cols:
            base_name = a_col[:-2]  # Remove "_A" suffix
            b_col = f"{base_name}_B"
            if b_col in b_cols:
                ab_col_pairs.append((a_col, b_col))

        # Identify base feature columns (shared between actions)
        base_cols = [
            col
            for col in data.columns
            if col not in ["qid", "action_A", "action_B", "preference_label_bool"]
            and not col.endswith("_A")
            and not col.endswith("_B")
        ]

        print(
            f"Found {len(base_cols)} base features and {len(ab_col_pairs)} action-specific feature pairs"
        )

        # Create feature matrix
        features = []

        # Add differences between A and B features (captures preference information)
        for a_col, b_col in ab_col_pairs:
            # Feature difference (A-B)
            diff_values = data[a_col].values - data[b_col].values
            features.append(diff_values)

            # Also add raw features to help model learn individual feature importance
            features.append(data[a_col].values)
            features.append(data[b_col].values)

        # Add base features that are shared
        for col in base_cols:
            features.append(data[col].values)

        # Stack features horizontally into a matrix
        X = np.column_stack(features)

        # Store feature column names for later reference
        diff_col_names = [f"diff_{a_col[:-2]}" for a_col, _ in ab_col_pairs]
        a_col_names = [f"raw_{a_col}" for a_col, _ in ab_col_pairs]
        b_col_names = [f"raw_{b_col}" for _, b_col in ab_col_pairs]
        self.feature_cols = diff_col_names + a_col_names + b_col_names + base_cols

        # Print feature stats
        print(f"Feature matrix shape: {X.shape}")
        print(
            f"Feature statistics - min: {X.min():.4f}, max: {X.max():.4f}, mean: {X.mean():.4f}, std: {X.std():.4f}"
        )

        # Check for NaN values
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            print(f"WARNING: Found {nan_count} NaN values in feature matrix")
            # Impute NaN values with median
            X = self.imputer.fit_transform(X)

        # Check for zero variance features
        variances = np.var(X, axis=0)
        zero_var_indices = np.where(variances < 1e-10)[0]
        if len(zero_var_indices) > 0:
            print(f"WARNING: {len(zero_var_indices)} features have near-zero variance")

            # Remove zero variance features
            if (
                len(zero_var_indices) < X.shape[1]
            ):  # If not all features have zero variance
                print(f"Removing {len(zero_var_indices)} zero variance features")
                non_zero_var_indices = np.where(variances >= 1e-10)[0]
                X = X[:, non_zero_var_indices]
                self.feature_cols = [self.feature_cols[i] for i in non_zero_var_indices]
                print(f"Remaining features: {len(self.feature_cols)}")

        return X

    def train(self, train_data: pd.DataFrame):
        print("Preparing features...")
        X = self._prepare_features(train_data)
        y = train_data["preference_label_bool"].values

        # Check class distribution
        unique_labels, counts = np.unique(y, return_counts=True)
        print(f"Class distribution: {dict(zip(unique_labels, counts))}")

        # Create a robust scaler for feature normalization that preserves relative distances
        from sklearn.preprocessing import RobustScaler

        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Try different C values to find the best one
        best_c = None
        best_accuracy = 0

        c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
        print("Finding optimal C parameter...")

        for c in c_values:
            model = LinearSVC(
                C=c,
                loss="squared_hinge",
                dual=True,
                max_iter=5000,
                random_state=42,
                tol=1e-5,
            )

            try:
                model.fit(X_scaled, y)
                train_preds = model.predict(X_scaled)
                accuracy = np.mean(train_preds == y)
                print(f"C={c}, accuracy={accuracy:.4f}")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_c = c
            except Exception as e:
                print(f"Error with C={c}: {str(e)}")

        if best_c is not None:
            print(f"Best C value: {best_c} with accuracy {best_accuracy:.4f}")
            self.model.C = best_c

        print(
            f"Training final SVM on {len(X_scaled)} samples with {X_scaled.shape[1]} features..."
        )

        # Train the final model
        try:
            self.model.fit(X_scaled, y)

            # Calculate and print training metrics
            train_preds = self.model.predict(X_scaled)
            train_accuracy = np.mean(train_preds == y)
            print(f"Training completed! Training accuracy: {train_accuracy:.4f}")

            # Print feature importance
            if hasattr(self.model, "coef_"):
                coef = self.model.coef_[0]

                # Get some statistics on coefficients
                coef_abs = np.abs(coef)
                print(
                    f"Coefficient statistics - min: {coef.min():.6f}, max: {coef.max():.6f}"
                )
                print(
                    f"Abs coefficient statistics - min: {coef_abs.min():.6f}, max: {coef_abs.max():.6f}, mean: {coef_abs.mean():.6f}"
                )

                # Check if all coefficients are near zero
                if np.all(np.abs(coef) < 1e-5):
                    print(
                        "WARNING: All feature coefficients are near zero. Model may not have converged properly."
                    )

                # Print largest absolute coefficients
                print("Top 10 important features by weight magnitude:")
                top_indices = np.argsort(np.abs(coef))[-10:]
                for idx in reversed(top_indices):
                    feature_name = f"Feature {idx}"
                    if self.feature_cols and idx < len(self.feature_cols):
                        feature_name = self.feature_cols[idx]
                    print(f"{feature_name}: {coef[idx]:.6f}")

                # Also print the distribution of coefficient signs
                pos_coef = np.sum(coef > 0)
                neg_coef = np.sum(coef < 0)
                zero_coef = np.sum(np.abs(coef) < 1e-5)
                print(
                    f"Coefficient sign distribution: {pos_coef} positive, {neg_coef} negative, {zero_coef} near-zero"
                )

            else:
                print(
                    "Model doesn't have coefficients attribute. Cannot show feature importance."
                )

        except Exception as e:
            print(f"Error during final model training: {str(e)}")
            # Fall back to a more robust model
            print("Falling back to more robust SVM configuration...")
            self.model = LinearSVC(C=1.0, max_iter=5000, random_state=42)
            self.model.fit(X_scaled, y)

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        X = self._prepare_features(test_data)

        # Apply the same normalization as in training
        if self.scaler is not None:
            X = self.scaler.transform(X)

        # Use raw decision values for more granular ranking
        try:
            scores = self.model.decision_function(X)

            # Print score statistics
            print(
                f"Prediction scores - min: {scores.min():.6f}, max: {scores.max():.6f}, mean: {scores.mean():.6f}, std: {scores.std():.6f}"
            )

            # Check if all scores are the same
            if np.std(scores) < 1e-5:
                print("WARNING: All prediction scores are nearly identical")

            # Check if predictions are mostly extreme
            extreme_threshold = np.max(np.abs(scores)) * 0.9
            extreme_preds = np.sum(np.abs(scores) > extreme_threshold)
            print(
                f"Extreme predictions (> {extreme_threshold:.2f}): {extreme_preds} ({extreme_preds/len(scores)*100:.1f}%)"
            )

            return scores
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            # Fallback to binary prediction
            preds = self.model.predict(X)

            # Print distribution of binary predictions
            unique_preds, counts = np.unique(preds, return_counts=True)
            print(f"Binary prediction distribution: {dict(zip(unique_preds, counts))}")

            return preds.astype(float)

    def save_model(self, save_path: str):
        model_path = os.path.join(self.model_dir, f"{save_path}.pkl")
        with open(model_path, "wb") as f:
            # Save both the model and the scaler
            pickle.dump(
                {
                    "model": self.model,
                    "scaler": self.scaler,
                    "feature_cols": self.feature_cols,
                },
                f,
            )

    def load_model(self, load_path: str):
        model_path = os.path.join(self.model_dir, f"{load_path}.pkl")
        with open(model_path, "rb") as f:
            # Load both the model and the scaler
            saved_data = pickle.load(f)
            self.model = saved_data["model"]
            self.scaler = saved_data.get(
                "scaler"
            )  # Handle older saved models without scaler
            self.feature_cols = saved_data.get(
                "feature_cols"
            )  # Handle older saved models


class PairwiseNeuralLTRR(BaseLTRR):
    """Pairwise Learning to Rank using Neural Network"""

    class PairwiseDataset(Dataset):
        def __init__(self, data: pd.DataFrame):
            # Drop the question column if it exists
            data = data.copy()
            if "question" in data.columns:
                data = data.drop(columns=["question"])

            # Extract columns for each action
            base_cols = [
                col
                for col in data.columns
                if col not in ["qid", "action_A", "action_B", "preference_label_bool"]
                and not col.endswith("_A")
                and not col.endswith("_B")
            ]

            # Find all action-specific feature columns, excluding action_A and action_B identifiers
            self.a_cols = [
                col for col in data.columns if col.endswith("_A") and col != "action_A"
            ]
            self.b_cols = [
                col for col in data.columns if col.endswith("_B") and col != "action_B"
            ]

            # Validate that we have matching pairs
            self.ab_col_pairs = []
            for a_col in self.a_cols:
                base_name = a_col[:-2]  # Remove "_A" suffix
                b_col = f"{base_name}_B"
                if b_col in self.b_cols:
                    self.ab_col_pairs.append((a_col, b_col))

            # Extract features
            feature_lists = []

            # Add feature differences (A-B)
            for a_col, b_col in self.ab_col_pairs:
                feature_lists.append(data[a_col].values - data[b_col].values)

            # Add raw features for each action
            for a_col, _ in self.ab_col_pairs:
                feature_lists.append(data[a_col].values)

            for _, b_col in self.ab_col_pairs:
                feature_lists.append(data[b_col].values)

            # Add base features
            for col in base_cols:
                feature_lists.append(data[col].values)

            # Combine all features
            self.features = torch.FloatTensor(np.column_stack(feature_lists))

            # Store preference labels
            self.labels = torch.FloatTensor(data["preference_label_bool"].values)

            # Create mask for NaN values
            self.mask = torch.isnan(self.features)

            # Initialize learnable parameters for NaN values
            self.nan_params = nn.Parameter(torch.randn(self.features.shape[1]))

            # Print some diagnostics
            print(
                f"PairwiseDataset - features: {self.features.shape}, labels: {self.labels.shape}"
            )
            print(
                f"Feature statistics - min: {self.features.min().item():.4f}, max: {self.features.max().item():.4f}"
            )
            print(f"NaN values: {self.mask.sum().item()}")

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            feature = self.features[idx].clone()
            # Replace NaN values with learnable parameters
            feature[self.mask[idx]] = self.nan_params[self.mask[idx]]
            return feature, self.labels[idx]

    class PairwiseModel(nn.Module):
        def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64]):
            super().__init__()
            layers = []
            prev_dim = input_dim

            # Add batch normalization and dropout
            for hidden_dim in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(prev_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                    ]
                )
                prev_dim = hidden_dim

            # Final layer without activation
            layers.append(nn.Linear(prev_dim, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x).squeeze()

    def __init__(
        self,
        model_dir: str,
        hidden_dims: List[int] = [256, 128, 64],
        batch_size: int = 32,
        learning_rate: float = 0.0001,  # Reduced learning rate
    ):
        super().__init__(model_dir)
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def pairwise_loss(self, pred, target):
        """Compute pairwise ranking loss using a modified hinge loss formulation"""
        # Apply tanh to normalize predictions to [-1, 1] range
        pred = torch.tanh(pred)

        # Compute margin
        margin = 0.1  # Smaller margin for easier learning

        # Compute loss with margin (using target directly)
        # target is already in {-1, 1}, so we multiply by target
        loss = torch.relu(margin - pred * target).mean()

        # Add L2 regularization on the predictions
        l2_reg = 0.01 * torch.mean(pred**2)

        return loss + l2_reg

    def train(self, train_data: pd.DataFrame, num_epochs: int = 100):
        dataset = self.PairwiseDataset(train_data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Get input dimension from the dataset features (after preprocessing)
        input_dim = dataset.features.shape[1]
        print(f"Model input dimension: {input_dim}")

        self.model = self.PairwiseModel(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
        ).to(self.device)

        # Use AdamW optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=0.01
        )

        # Use learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        best_loss = float("inf")
        patience = 10
        patience_counter = 0

        for epoch in tqdm(range(num_epochs), desc="Training epochs"):
            self.model.train()
            total_loss = 0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

            for features, labels in pbar:
                features = features.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                # Get predictions
                preds = self.model(features)

                # Compute pairwise ranking loss
                loss = self.pairwise_loss(preds, labels)

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / len(dataloader)
            scheduler.step(avg_loss)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

                # Evaluate on the training set
                self.model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for features, labels in dataloader:
                        features = features.to(self.device)
                        labels = labels.to(self.device)
                        preds = self.model(features)
                        # Calculate binary accuracy
                        pred_signs = torch.sign(preds)
                        correct += (pred_signs == labels).sum().item()
                        total += len(labels)

                    accuracy = correct / total
                    print(f"Training accuracy: {accuracy:.4f}")

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        dataset = self.PairwiseDataset(test_data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for features, _ in dataloader:
                features = features.to(self.device)

                # Get predictions
                preds = self.model(features)
                predictions.extend(preds.cpu().numpy())

        # Print prediction statistics
        predictions_array = np.array(predictions)
        print(
            f"Prediction statistics - min: {predictions_array.min():.6f}, max: {predictions_array.max():.6f}"
        )
        print(
            f"Mean: {predictions_array.mean():.6f}, std: {predictions_array.std():.6f}"
        )

        return predictions_array

    def save_model(self, save_path: str):
        model_path = os.path.join(self.model_dir, f"{save_path}.pt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "hidden_dims": self.hidden_dims,
            },
            model_path,
        )

    def load_model(self, load_path: str):
        model_path = os.path.join(self.model_dir, f"{load_path}.pt")
        checkpoint = torch.load(model_path, weights_only=True)

        # Extract dimensions from the checkpoint state dict
        state_dict = checkpoint["model_state_dict"]
        input_dim = state_dict["network.0.weight"].shape[1]
        hidden_dims = checkpoint["hidden_dims"]

        # Create the model with correct dimensions first
        self.model = self.PairwiseModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)

        # Then load the state dict
        self.model.load_state_dict(state_dict)


class ListwiseNeuralLTRR(BaseLTRR):
    """Listwise Learning to Rank using Neural Network (ListNet style)"""

    class ListwiseDataset(Dataset):
        def __init__(self, data: pd.DataFrame):
            # Drop the question column if it exists
            data = data.copy()
            if "question" in data.columns:
                data = data.drop(columns=["question"])

            # Get query features (same for all actions)
            query_cols = ["qid"] + [
                col
                for col in data.columns
                if col.startswith("emb_") or col in ["query_length", "query_type"]
            ]
            self.query_features = torch.FloatTensor(
                data[query_cols].drop_duplicates().drop("qid", axis=1).values
            )

            # Get action features and labels
            action_cols = [
                col
                for col in data.columns
                if any(
                    col.startswith(prefix)
                    for prefix in [
                        "overall_sim_",
                        "avg_sim_",
                        "max_sim_",
                        "var_sim_",
                        "moran_",
                        "cross_retriever_sim_",
                    ]
                )
            ]
            self.action_features = torch.FloatTensor(data[action_cols].values)

            label_cols = [col for col in data.columns if col.startswith("delta_label_")]
            self.labels = torch.FloatTensor(data[label_cols].values)

            # Create mask for NaN values
            self.mask = torch.isnan(self.action_features)
            # Initialize learnable parameters for NaN values
            self.nan_params = nn.Parameter(torch.randn(self.action_features.shape[1]))

        def __len__(self):
            return len(self.query_features)

        def __getitem__(self, idx):
            query_feat = self.query_features[idx]
            action_feat = self.action_features[idx].clone()
            # Replace NaN values with learnable parameters
            action_feat[self.mask[idx]] = self.nan_params[self.mask[idx]]
            label = self.labels[idx]
            return query_feat, action_feat, label

    class ListwiseModel(nn.Module):
        def __init__(
            self,
            query_dim: int,
            action_dim: int,
            num_actions: int = 7,  # Number of actions (0-6)
            hidden_dims: List[int] = [256, 128, 64],
        ):
            super().__init__()
            # Query encoder
            query_layers = []
            prev_dim = query_dim
            for hidden_dim in hidden_dims:
                query_layers.extend(
                    [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)]
                )
                prev_dim = hidden_dim
            self.query_encoder = nn.Sequential(*query_layers)

            # Action encoder
            action_layers = []
            prev_dim = action_dim
            for hidden_dim in hidden_dims:
                action_layers.extend(
                    [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)]
                )
                prev_dim = hidden_dim
            self.action_encoder = nn.Sequential(*action_layers)

            # Final scoring layer - output scores for all actions
            # doing times two as the model first encodes the query features and action features separately,
            # each resulting in a vector of size hidden_dims[-1]
            self.scoring_layer = nn.Linear(hidden_dims[-1] * 2, num_actions)

        def forward(self, query_feat, action_feat):
            query_encoded = self.query_encoder(query_feat)
            action_encoded = self.action_encoder(action_feat)
            combined = torch.cat([query_encoded, action_encoded], dim=1)
            return self.scoring_layer(combined)  # [batch_size, num_actions]

    def __init__(
        self,
        model_dir: str,
        hidden_dims: List[int] = [256, 128, 64],
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ):
        super().__init__(model_dir)
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, train_data: pd.DataFrame, num_epochs: int = 100):
        dataset = self.ListwiseDataset(train_data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        query_dim = dataset.query_features.shape[1]
        action_dim = dataset.action_features.shape[1]
        num_actions = dataset.labels.shape[1]  # Number of actions from labels

        self.model = self.ListwiseModel(
            query_dim=query_dim,
            action_dim=action_dim,
            num_actions=num_actions,
            hidden_dims=self.hidden_dims,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in tqdm(range(num_epochs), desc="Training epochs"):
            total_loss = 0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
            for query_feat, action_feat, labels in pbar:
                query_feat = query_feat.to(self.device)
                action_feat = action_feat.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                scores = self.model(
                    query_feat, action_feat
                )  # [batch_size, num_actions]

                # ListNet loss
                pred_probs = F.softmax(
                    scores, dim=1
                )  # Apply softmax along action dimension
                target_probs = F.softmax(labels, dim=1)

                # Compute cross entropy loss
                loss = (
                    -torch.sum(target_probs * torch.log(pred_probs + 1e-10))
                    / self.batch_size
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}"
                )

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        dataset = self.ListwiseDataset(test_data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for query_feat, action_feat, _ in dataloader:
                query_feat = query_feat.to(self.device)
                action_feat = action_feat.to(self.device)
                scores = self.model(query_feat, action_feat)
                predictions.extend(scores.cpu().numpy())
        return np.array(predictions)

    def save_model(self, save_path: str):
        model_path = os.path.join(self.model_dir, f"{save_path}.pt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "hidden_dims": self.hidden_dims,
            },
            model_path,
        )

    def load_model(self, load_path: str):
        model_path = os.path.join(self.model_dir, f"{load_path}.pt")
        checkpoint = torch.load(model_path, weights_only=True)

        # Extract dimensions from the checkpoint state dict
        state_dict = checkpoint["model_state_dict"]
        query_dim = state_dict["query_encoder.0.weight"].shape[1]
        action_dim = state_dict["action_encoder.0.weight"].shape[1]
        num_actions = state_dict["scoring_layer.weight"].shape[0]
        hidden_dims = checkpoint["hidden_dims"]

        # Create the model with correct dimensions first
        self.model = self.ListwiseModel(
            query_dim=query_dim,
            action_dim=action_dim,
            num_actions=num_actions,
            hidden_dims=hidden_dims,
        ).to(self.device)

        # Then load the state dict
        self.model.load_state_dict(state_dict)


class ListwiseLambdaMARTLTRR(BaseLTRR):
    """Listwise Learning to Rank using Neural Network (LambdaMART style)
    with action-specific scoring heads
    """

    class ListwiseDataset(Dataset):
        def __init__(self, data: pd.DataFrame):
            # Drop the question column if it exists
            data = data.copy()
            if "question" in data.columns:
                data = data.drop(columns=["question"])

            # Get query features (same for all actions)
            query_cols = ["qid"] + [
                col
                for col in data.columns
                if col.startswith("emb_") or col in ["query_length", "query_type"]
            ]
            self.query_features = torch.FloatTensor(
                data[query_cols].drop_duplicates().drop("qid", axis=1).values
            )

            # Store query IDs for reference
            self.qids = data[query_cols].drop_duplicates()["qid"].values

            # Get action features and labels
            action_cols = [
                col
                for col in data.columns
                if any(
                    col.startswith(prefix)
                    for prefix in [
                        "overall_sim_",
                        "avg_sim_",
                        "max_sim_",
                        "var_sim_",
                        "moran_",
                        "cross_retriever_sim_",
                    ]
                )
            ]
            self.action_features = torch.FloatTensor(data[action_cols].values)

            label_cols = [col for col in data.columns if col.startswith("delta_label_")]
            self.labels = torch.FloatTensor(data[label_cols].values)

            # Create mask for NaN values
            self.mask = torch.isnan(self.action_features)
            # Initialize learnable parameters for NaN values
            self.nan_params = nn.Parameter(torch.randn(self.action_features.shape[1]))

            # Print dataset statistics
            print(
                f"ListwiseDataset - query features: {self.query_features.shape}, action features: {self.action_features.shape}, labels: {self.labels.shape}"
            )
            print(
                f"Label statistics - min: {self.labels.min().item():.4f}, max: {self.labels.max().item():.4f}, mean: {self.labels.mean().item():.4f}"
            )
            print(f"NaN values in features: {self.mask.sum().item()}")
            print(f"Number of unique queries: {len(self.qids)}")

        def __len__(self):
            return len(self.query_features)

        def __getitem__(self, idx):
            query_feat = self.query_features[idx]
            action_feat = self.action_features[idx].clone()
            # Replace NaN values with learnable parameters
            action_feat[self.mask[idx]] = self.nan_params[self.mask[idx]]
            label = self.labels[idx]
            return query_feat, action_feat, label

    class ListwiseModel(nn.Module):
        def __init__(
            self,
            query_dim: int,
            action_dim: int,
            num_actions: int = 7,  # Number of actions (0-6)
            hidden_dims: List[int] = [128, 64, 32],  # Smaller hidden dimensions
        ):
            super().__init__()
            # Query-specific scoring branch
            self.query_encoder = nn.Sequential(
                nn.Linear(query_dim, hidden_dims[0]),
                nn.LayerNorm(hidden_dims[0]),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.LayerNorm(hidden_dims[1]),
                nn.GELU(),
                nn.Dropout(0.3),
            )

            # Action-specific scoring branch
            self.action_encoder = nn.Sequential(
                nn.Linear(action_dim, hidden_dims[0]),
                nn.LayerNorm(hidden_dims[0]),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.LayerNorm(hidden_dims[1]),
                nn.GELU(),
                nn.Dropout(0.3),
            )

            # Pairwise interaction layers to capture query-action relationships
            self.interaction_layer = nn.Sequential(
                nn.Linear(hidden_dims[1] * 2, hidden_dims[1]),
                nn.LayerNorm(hidden_dims[1]),
                nn.GELU(),
                nn.Dropout(0.3),
            )

            # Each action gets its own scoring head to ensure diversity
            self.action_scorers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dims[1], hidden_dims[2]),
                        nn.LayerNorm(hidden_dims[2]),
                        nn.GELU(),
                        nn.Dropout(0.3),
                        nn.Linear(hidden_dims[2], 1),
                    )
                    for _ in range(num_actions)
                ]
            )

            # Initialize with different random seeds to ensure diversity
            for i, scorer in enumerate(self.action_scorers):
                # Use different initialization for each action scorer
                torch.manual_seed(42 + i)
                for layer in scorer:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_normal_(layer.weight, gain=0.001)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)

        def forward(self, query_feat, action_feat):
            # Encode query and action features
            query_encoded = self.query_encoder(query_feat)
            action_encoded = self.action_encoder(action_feat)

            # Combine features to capture interactions
            combined = torch.cat([query_encoded, action_encoded], dim=1)
            interaction = self.interaction_layer(combined)

            # Score each action separately
            scores = []
            for scorer in self.action_scorers:
                scores.append(scorer(interaction))

            # Combine scores and ensure they're in [-1, 1] range with tanh
            return torch.tanh(torch.cat(scores, dim=1))

    def __init__(
        self,
        model_dir: str,
        hidden_dims: List[int] = [128, 64, 32],  # Smaller model
        batch_size: int = 16,  # Smaller batch size for more updates
        learning_rate: float = 0.0003,  # Slightly higher learning rate
    ):
        super().__init__(model_dir)
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Track validation metrics
        self.val_ndcg_history = []
        # For calculating diversity penalty
        self.similarity_penalty = 0.1

    def _compute_lambda(
        self, scores: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute LambdaMART gradients with improved normalization"""
        batch_size = scores.size(0)
        num_actions = scores.size(1)
        lambdas = torch.zeros_like(scores).to(self.device)

        # For numerical stability
        MAX_EXP = 10.0  # Cap for exponential

        for i in range(batch_size):
            query_scores = scores[i]  # [num_actions]
            query_labels = labels[i]  # [num_actions]

            # Pre-compute relevance weights and DCG discount factors
            rel_weights = (2**query_labels - 1).clamp(min=0)
            discount_factors = 1.0 / torch.log2(
                torch.arange(2, num_actions + 2, device=self.device).float()
            )

            # Calculate ideal DCG
            sorted_labels, _ = torch.sort(query_labels, descending=True)
            ideal_dcg = torch.sum(
                (2**sorted_labels - 1).clamp(min=0) * discount_factors
            )
            # Avoid division by zero
            ideal_dcg = ideal_dcg.clamp(min=1e-10)

            # Only compute pairs where the relevance differs
            lambda_updates = []  # Track lambda updates to normalize them

            for j in range(num_actions):
                for k in range(j + 1, num_actions):  # Only compute upper triangle
                    if query_labels[j] != query_labels[k]:
                        # Get positions in current ranking
                        j_pos = torch.where(
                            torch.argsort(query_scores, descending=True) == j
                        )[0].item()
                        k_pos = torch.where(
                            torch.argsort(query_scores, descending=True) == k
                        )[0].item()

                        # Compute DCG difference
                        dcg_diff = abs(
                            rel_weights[j]
                            * (discount_factors[j_pos] - discount_factors[k_pos])
                            + rel_weights[k]
                            * (discount_factors[k_pos] - discount_factors[j_pos])
                        )

                        # Normalize by ideal DCG
                        delta_ndcg = dcg_diff / ideal_dcg

                        # Determine which item should be ranked higher
                        sign = 1 if query_labels[j] > query_labels[k] else -1

                        # Compute lambda update with moderated exponential
                        rho = query_scores[j] - query_scores[k]
                        exp_term = torch.exp(
                            torch.clamp(sign * rho, min=-MAX_EXP, max=MAX_EXP)
                        )
                        lambda_val = sign * delta_ndcg * exp_term / (1.0 + exp_term)

                        # Store lambda update for normalization
                        lambda_updates.append((j, k, lambda_val))

            # Normalize lambda updates to prevent exploding gradients
            if lambda_updates:
                lambda_vals = torch.tensor([x[2] for x in lambda_updates])
                # Compute scaling factor - scale to max value of 0.1
                scale_factor = min(0.1 / (lambda_vals.abs().max() + 1e-10), 1.0)

                # Apply normalized lambda updates
                for j, k, lambda_val in lambda_updates:
                    normalized_val = lambda_val * scale_factor
                    lambdas[i, j] += normalized_val
                    lambdas[i, k] -= normalized_val

        return lambdas

    def _calculate_ndcg(self, scores, labels):
        """Calculate NDCG for monitoring"""
        batch_size = scores.size(0)
        batch_ndcg = 0

        for i in range(batch_size):
            # Sort scores and get corresponding labels
            _, score_indices = torch.sort(scores[i], descending=True)
            sorted_labels = labels[i][score_indices]

            # Compute DCG
            rel_weights = (2**sorted_labels - 1).clamp(min=0)
            discount = 1.0 / torch.log2(
                torch.arange(2, len(sorted_labels) + 2, device=self.device).float()
            )
            dcg = torch.sum(rel_weights * discount)

            # Compute ideal DCG
            sorted_ideal_labels, _ = torch.sort(labels[i], descending=True)
            ideal_weights = (2**sorted_ideal_labels - 1).clamp(min=0)
            idcg = torch.sum(ideal_weights * discount)

            # Compute NDCG
            ndcg = dcg / (idcg + 1e-10)
            batch_ndcg += ndcg.item()

        return batch_ndcg / batch_size

    def train(self, train_data: pd.DataFrame, num_epochs: int = 30):
        # Split data into train/validation
        train_indices = np.random.choice(
            len(train_data), size=int(0.8 * len(train_data)), replace=False
        )
        train_mask = np.zeros(len(train_data), dtype=bool)
        train_mask[train_indices] = True

        train_df = train_data[train_mask].reset_index(drop=True)
        val_df = train_data[~train_mask].reset_index(drop=True)

        print(
            f"Training on {len(train_df)} samples, validating on {len(val_df)} samples"
        )

        train_dataset = self.ListwiseDataset(train_df)
        val_dataset = self.ListwiseDataset(val_df)

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

        query_dim = train_dataset.query_features.shape[1]
        action_dim = train_dataset.action_features.shape[1]
        num_actions = train_dataset.labels.shape[1]  # Number of actions from labels

        self.model = self.ListwiseModel(
            query_dim=query_dim,
            action_dim=action_dim,
            num_actions=num_actions,
            hidden_dims=self.hidden_dims,
        ).to(self.device)

        # Use two optimizers:
        # 1. One for the main network (encoders and interaction layer)
        main_params = (
            list(self.model.query_encoder.parameters())
            + list(self.model.action_encoder.parameters())
            + list(self.model.interaction_layer.parameters())
        )

        # 2. One for the action-specific scoring heads (with higher learning rate)
        action_params = list(self.model.action_scorers.parameters())

        optimizer_main = torch.optim.AdamW(
            main_params, lr=self.learning_rate, weight_decay=0.01
        )

        optimizer_action = torch.optim.AdamW(
            action_params,
            lr=self.learning_rate * 5,  # Higher learning rate for diversity
            weight_decay=0.001,  # Lower weight decay to allow more diversity
        )

        # Learning rate schedulers
        scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_main, T_max=num_epochs, eta_min=1e-6
        )
        scheduler_action = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_action, T_max=num_epochs, eta_min=1e-5
        )

        best_val_ndcg = 0
        best_diversity = 0
        patience = 7
        patience_counter = 0

        # Store all query IDs for better tracking
        all_qids = train_df["qid"].unique()

        for epoch in tqdm(range(num_epochs), desc="Training epochs"):
            # Training phase
            self.model.train()
            total_loss = 0
            total_ndcg = 0

            # Add diversity loss scaling that increases over epochs
            diversity_weight = min(0.5, 0.05 * (1 + epoch))  # Gradually increase

            pbar = tqdm(
                train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
            )
            for batch_idx, (query_feat, action_feat, labels) in enumerate(pbar):
                query_feat = query_feat.to(self.device)
                action_feat = action_feat.to(self.device)
                labels = labels.to(self.device)

                # Reset gradients
                optimizer_main.zero_grad()
                optimizer_action.zero_grad()

                # Forward pass
                scores = self.model(query_feat, action_feat)

                # Add significant noise early in training, especially for action scorers
                # This helps break symmetry between different action scorers
                if epoch < num_epochs // 2:
                    noise_scale = 0.1 * (1.0 - epoch / (num_epochs // 2))
                    scores = scores + torch.randn_like(scores) * noise_scale

                # Compute lambdaMART gradients
                lambdas = self._compute_lambda(scores, labels)

                # Main ranking loss (scaled down)
                ranking_loss = -torch.sum(scores * lambdas) / (self.batch_size * 10)

                # Add diversity-promoting loss component
                if epoch > 0:  # Start diversity loss after first epoch
                    # Compute pairwise cosine similarity between score vectors
                    norm_scores = F.normalize(scores, p=2, dim=1)
                    similarity_matrix = torch.mm(
                        norm_scores, norm_scores.transpose(0, 1)
                    )

                    # We want to minimize off-diagonal similarity (maximize diversity)
                    # Create a mask to zero out the diagonal
                    mask = torch.eye(similarity_matrix.size(0), device=self.device) == 0

                    # Compute diversity loss (mean of off-diagonal similarities)
                    diversity_loss = torch.sum(torch.abs(similarity_matrix * mask)) / (
                        mask.sum() + 1e-8
                    )

                    # Combined loss with diversity component
                    loss = ranking_loss + diversity_weight * diversity_loss
                else:
                    diversity_loss = torch.tensor(0.0, device=self.device)
                    loss = ranking_loss

                # Add L2 regularization on scores to prevent extreme values
                score_l2 = 0.01 * torch.mean(scores**2)

                # Final loss
                total_loss_val = loss + score_l2

                # Backward pass and optimization
                total_loss_val.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Update parameters
                optimizer_main.step()
                optimizer_action.step()

                # Track metrics
                total_loss += ranking_loss.item()

                # Calculate NDCG
                batch_ndcg = self._calculate_ndcg(scores.detach(), labels)
                total_ndcg += batch_ndcg

                pbar.set_postfix(
                    {
                        "ranking_loss": f"{ranking_loss.item():.4f}",
                        "diversity_loss": f"{diversity_loss.item():.4f}",
                        "ndcg": f"{batch_ndcg:.4f}",
                    }
                )

            # End of epoch training metrics
            train_loss = total_loss / len(train_dataloader)
            train_ndcg = total_ndcg / len(train_dataloader)

            # Validation phase
            self.model.eval()
            val_loss = 0
            val_ndcg = 0
            val_predictions = []

            with torch.no_grad():
                for query_feat, action_feat, labels in val_dataloader:
                    query_feat = query_feat.to(self.device)
                    action_feat = action_feat.to(self.device)
                    labels = labels.to(self.device)

                    scores = self.model(query_feat, action_feat)
                    lambdas = self._compute_lambda(scores, labels)

                    loss = -torch.sum(scores * lambdas) / (self.batch_size * 10)
                    val_loss += loss.item()

                    # Calculate NDCG
                    batch_ndcg = self._calculate_ndcg(scores, labels)
                    val_ndcg += batch_ndcg

                    # Store predictions for diversity analysis
                    val_predictions.extend(scores.cpu().numpy())

            val_loss = val_loss / len(val_dataloader)
            val_ndcg = val_ndcg / len(val_dataloader)
            self.val_ndcg_history.append(val_ndcg)

            # Check ranking diversity
            unique_rankings = set()
            for pred in val_predictions:
                ranking = tuple(np.argsort(-pred))
                unique_rankings.add(ranking)

            diversity_ratio = len(unique_rankings) / len(val_predictions)

            # Update schedulers
            scheduler_main.step()
            scheduler_action.step()

            # Print metrics
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train NDCG: {train_ndcg:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val NDCG: {val_ndcg:.4f}, Diversity: {diversity_ratio:.4f}"
            )

            # Check diversity every epoch
            print(
                f"Validation ranking diversity: {len(unique_rankings)} unique rankings out of {len(val_predictions)} samples ({diversity_ratio*100:.2f}%)"
            )

            # Detailed view of top rankings
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                ranking_counts = {}
                for pred in val_predictions:
                    ranking = tuple(np.argsort(-pred))
                    if ranking not in ranking_counts:
                        ranking_counts[ranking] = 0
                    ranking_counts[ranking] += 1

                top_rankings = sorted(
                    ranking_counts.items(), key=lambda x: x[1], reverse=True
                )[:3]
                print("Top 3 rankings:")
                for i, (ranking, count) in enumerate(top_rankings):
                    print(
                        f"  #{i+1}: {ranking} - {count} samples ({count/len(val_predictions)*100:.2f}%)"
                    )

            # Early stopping considers both NDCG and diversity
            combined_metric = val_ndcg * (
                0.5 + 0.5 * diversity_ratio
            )  # Balance NDCG and diversity

            if (combined_metric > best_val_ndcg * (0.5 + 0.5 * best_diversity)) or (
                diversity_ratio > 0.5 and val_ndcg >= 0.7
            ):
                best_val_ndcg = val_ndcg
                best_diversity = diversity_ratio
                patience_counter = 0

                # Save best model
                best_model_path = os.path.join(self.model_dir, "best_lambdamart.pt")
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "hidden_dims": self.hidden_dims,
                        "epoch": epoch,
                        "val_ndcg": val_ndcg,
                        "diversity": diversity_ratio,
                    },
                    best_model_path,
                )
                print(
                    f"Saved best model at epoch {epoch+1} with val NDCG {val_ndcg:.4f} and diversity {diversity_ratio:.4f}"
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping at epoch {epoch+1}, no improvement for {patience} epochs"
                    )
                    break

        # Load the best model
        best_model_path = os.path.join(self.model_dir, "best_lambdamart.pt")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(
                f"Loaded best model from epoch {checkpoint['epoch']+1} with val NDCG {checkpoint['val_ndcg']:.4f} and diversity {checkpoint['diversity']:.4f}"
            )

        # Plot NDCG history
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.plot(self.val_ndcg_history)
            plt.title("Validation NDCG over Epochs")
            plt.xlabel("Epochs")
            plt.ylabel("NDCG")
            plt.grid(True)
            plt.savefig(os.path.join(self.model_dir, "ndcg_history.png"))
            plt.close()
            print(
                f"Saved NDCG history plot to {os.path.join(self.model_dir, 'ndcg_history.png')}"
            )
        except:
            print("Could not create NDCG history plot")

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        dataset = self.ListwiseDataset(test_data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        predictions = []
        ranking_diversity = {}
        all_qids = []

        with torch.no_grad():
            for query_feat, action_feat, _ in dataloader:
                query_feat = query_feat.to(self.device)
                action_feat = action_feat.to(self.device)

                # Forward pass to get scores
                base_scores = self.model(query_feat, action_feat)

                # Add slight randomness to predictions to increase diversity
                # Using a very small fixed random seed for reproducibility
                # but different seed for each query to ensure diversity
                for i in range(len(base_scores)):
                    # Get the current global index
                    global_idx = len(predictions) + i

                    # Safely get the qid - make sure we don't exceed the dataset size
                    if hasattr(dataset, "qids") and global_idx < len(dataset.qids):
                        qid = dataset.qids[global_idx]
                    else:
                        qid = str(global_idx)  # Use index as a fallback

                    all_qids.append(qid)

                    # Set a query-specific seed for randomness
                    qid_numeric = (
                        int(qid.split("-")[-1]) if "-" in str(qid) else hash(str(qid))
                    )
                    np.random.seed(qid_numeric % 10000)

                    # Add tiny random noise (0.1%) to break ties differently for each query
                    noise = torch.FloatTensor(
                        np.random.normal(0, 0.001, size=base_scores[i].shape)
                    ).to(self.device)
                    scores_with_noise = base_scores[i] + noise

                    # Track ranking diversity
                    ranking = tuple(
                        torch.argsort(scores_with_noise, descending=True).cpu().numpy()
                    )
                    if ranking not in ranking_diversity:
                        ranking_diversity[ranking] = []
                    ranking_diversity[ranking].append(qid)

                    # Store prediction
                    predictions.append(scores_with_noise.cpu().numpy())

                # Print example predictions for the first batch
                if len(predictions) <= len(base_scores):
                    print(f"First batch prediction statistics:")
                    for i in range(min(5, len(base_scores))):
                        print(
                            f"  Query {i} (ID: {all_qids[i]}): scores = {base_scores[i].cpu().numpy()}"
                        )
                        print(
                            f"    Ranking: {torch.argsort(base_scores[i], descending=True).cpu().numpy()}"
                        )
                        print(
                            f"    With noise: {torch.argsort(base_scores[i] + noise, descending=True).cpu().numpy()}"
                        )

        predictions_array = np.array(predictions)

        # Print prediction statistics
        print(f"Prediction statistics - shape: {predictions_array.shape}")
        print(
            f"  min: {predictions_array.min():.6f}, max: {predictions_array.max():.6f}"
        )
        print(
            f"  mean: {predictions_array.mean():.6f}, std: {predictions_array.std():.6f}"
        )

        # Analyze ranking diversity
        print(f"Number of unique rankings in predictions: {len(ranking_diversity)}")
        print(
            f"Diversity ratio: {len(ranking_diversity) / len(predictions_array) * 100:.2f}%"
        )

        # Show most common rankings
        most_common = sorted(
            [(r, len(qids)) for r, qids in ranking_diversity.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        print("\nMost common rankings:")
        for i, (ranking, count) in enumerate(most_common[:5]):
            print(
                f"  #{i+1}: {ranking} - {count} instances ({count/len(predictions_array)*100:.2f}%)"
            )

            # Print some example query IDs for this ranking
            if i == 0:  # For most common ranking
                example_qids = (
                    ranking_diversity[ranking][:5]
                    if len(ranking_diversity[ranking]) > 5
                    else ranking_diversity[ranking]
                )
                print(f"    Example query IDs: {example_qids}")

        return predictions_array

    def save_model(self, save_path: str):
        model_path = os.path.join(self.model_dir, f"{save_path}.pt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "hidden_dims": self.hidden_dims,
                "val_ndcg_history": self.val_ndcg_history,
                "similarity_penalty": self.similarity_penalty,
            },
            model_path,
        )
        print(f"Model saved to {model_path}")

    def load_model(self, load_path: str):
        model_path = os.path.join(self.model_dir, f"{load_path}.pt")
        checkpoint = torch.load(model_path, weights_only=True)

        # Extract dimensions from the checkpoint state dict
        state_dict = checkpoint["model_state_dict"]
        query_dim = state_dict["query_encoder.0.weight"].shape[1]
        action_dim = state_dict["action_encoder.0.weight"].shape[1]

        # Check if it's the new architecture with action_scorers
        if any("action_scorers" in k for k in state_dict.keys()):
            # Find number of actions based on number of scorers
            num_actions = 0
            for k in state_dict.keys():
                if k.startswith("action_scorers.") and k.split(".")[1].isdigit():
                    num_actions = max(num_actions, int(k.split(".")[1]) + 1)
        else:
            # Legacy model: determine from scoring layer
            num_actions = state_dict["scoring_layer.weight"].shape[0]

        hidden_dims = checkpoint.get("hidden_dims", self.hidden_dims)
        self.val_ndcg_history = checkpoint.get("val_ndcg_history", [])
        self.similarity_penalty = checkpoint.get(
            "similarity_penalty", self.similarity_penalty
        )

        # Create the model with correct dimensions
        self.model = self.ListwiseModel(
            query_dim=query_dim,
            action_dim=action_dim,
            num_actions=num_actions,
            hidden_dims=hidden_dims,
        ).to(self.device)

        # Handle potential mismatches in state dict keys
        model_dict = self.model.state_dict()
        # Filter out keys in checkpoint that don't exist in model
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        # Update model state dict
        model_dict.update(state_dict)

        # Load the filtered state dict
        self.model.load_state_dict(model_dict, strict=False)
        print(f"Model loaded from {model_path}")
        # If there were missing keys, print a warning
        missing_keys = set(model_dict.keys()) - set(state_dict.keys())
        if missing_keys:
            print(
                f"Warning: {len(missing_keys)} keys missing from checkpoint. Model initialized with random weights for these layers."
            )


# ================
"""
DeBERTa (microsoft/deberta-v3-base)-base rankers
    - takes in question string (not the PCA'ed embeddings)
        - in the loaded dataset, there will be "question" column
    - takes in post-retrieval features
"""


class PointwiseDeBERTaLTRR(BaseLTRR):
    """Pointwise Learning to Rank using DeBERTa"""

    class PointwiseDataset(Dataset):
        def __init__(self, data: pd.DataFrame):
            # Extract data
            self.questions = data["question"].values
            self.post_features = data[
                [
                    "query_length",
                    "query_type",
                    "overall_sim",
                    "avg_sim",
                    "max_sim",
                    "var_sim",
                    "moran",
                    "cross_retriever_sim",
                ]
            ].values
            self.labels = torch.FloatTensor(data["delta_label"].values)

            # Create masks for NaN values in post features
            self.mask = torch.isnan(torch.FloatTensor(self.post_features))
            # Initialize learnable parameters for NaN values
            self.nan_params = nn.Parameter(torch.randn(self.post_features.shape[1]))

            # Store mapping from index to qid and action for evaluation
            self.qids = data["qid"].values
            self.actions = data["action"].values

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            # Get text and features
            question = self.questions[idx]
            post_feat = torch.FloatTensor(self.post_features[idx].copy())

            # Replace NaN values with learnable parameters
            post_feat[self.mask[idx]] = self.nan_params[self.mask[idx]]

            return {
                "question": question,
                "features": post_feat,
                "label": self.labels[idx],
                "qid": self.qids[idx],
                "action": self.actions[idx],
            }

    def __init__(
        self,
        model_dir: str,
        model_name: str = "microsoft/deberta-v3-base",
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        dropout: float = 0.1,
        max_length: int = 128,
    ):
        super().__init__(model_dir)
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # These will be initialized in the train method
        self.tokenizer = None
        self.model = None

    def _init_model(self):
        """Initialize the model, tokenizer and optimizer"""
        from transformers import DebertaV2Tokenizer, DebertaV2Model, AutoTokenizer
        import torch.nn as nn

        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except:
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_name)

        # Create model
        self.model = PointwiseDeBERTaModel(
            model_name=self.model_name,
            num_post_features=8,  # query_length, query_type, and 6 post-retrieval features
            dropout=self.dropout,
        ).to(self.device)

    class TextDataCollator:
        """Collates batches for the text model"""

        def __init__(self, tokenizer, max_length=128):
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __call__(self, batch):
            # Extract components
            questions = [item["question"] for item in batch]
            features = torch.stack([item["features"] for item in batch])
            labels = torch.stack([item["label"] for item in batch])
            qids = [item["qid"] for item in batch]
            actions = [item["action"] for item in batch]

            # Tokenize the questions
            encoded_questions = self.tokenizer(
                questions,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            return {
                "input_ids": encoded_questions["input_ids"],
                "attention_mask": encoded_questions["attention_mask"],
                "token_type_ids": encoded_questions.get("token_type_ids", None),
                "features": features,
                "labels": labels,
                "qids": qids,
                "actions": actions,
            }

    def train(self, train_data: pd.DataFrame, num_epochs: int = 5):
        self._init_model()

        dataset = self.PointwiseDataset(train_data)
        collator = self.TextDataCollator(self.tokenizer, self.max_length)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collator
        )

        # Optimizer with weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.learning_rate
        )

        # Learning rate scheduler
        total_steps = len(dataloader) * num_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )

        # Loss function (MSE for regression)
        criterion = nn.MSELoss()

        # Training loop
        self.model.train()
        for epoch in tqdm(range(num_epochs), desc="Training epochs"):
            total_loss = 0
            for batch in tqdm(
                dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
            ):
                # Transfer batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = (
                    batch["token_type_ids"].to(self.device)
                    if batch["token_type_ids"] is not None
                    else None
                )
                features = batch["features"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    features=features,
                )

                # Compute loss and update weights
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            # Log epoch metrics
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

            # Evaluate on a few examples
            if epoch % 2 == 0 or epoch == num_epochs - 1:
                self.model.eval()
                with torch.no_grad():
                    sample_batch = next(iter(dataloader))
                    input_ids = sample_batch["input_ids"].to(self.device)
                    attention_mask = sample_batch["attention_mask"].to(self.device)
                    token_type_ids = (
                        sample_batch["token_type_ids"].to(self.device)
                        if sample_batch["token_type_ids"] is not None
                        else None
                    )
                    features = sample_batch["features"].to(self.device)
                    labels = sample_batch["labels"].to(self.device)

                    predictions = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        features=features,
                    ).squeeze()

                    # Show a few examples
                    for i in range(min(5, len(predictions))):
                        print(
                            f"  Example {i}: Pred = {predictions[i].item():.4f}, Label = {labels[i].item():.4f}"
                        )
                self.model.train()

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")

        # Create test dataset and dataloader
        dataset = self.PointwiseDataset(test_data)
        collator = self.TextDataCollator(self.tokenizer, self.max_length)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collator
        )

        # Make predictions
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                # Transfer batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = (
                    batch["token_type_ids"].to(self.device)
                    if batch["token_type_ids"] is not None
                    else None
                )
                features = batch["features"].to(self.device)

                # Get predictions
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    features=features,
                ).squeeze()

                predictions.extend(outputs.cpu().numpy())

        return np.array(predictions)

    def save_model(self, save_path: str):
        model_dir = os.path.join(self.model_dir, save_path)
        os.makedirs(model_dir, exist_ok=True)

        # Save the model
        self.model.save_pretrained(model_dir)

        # Save the tokenizer
        self.tokenizer.save_pretrained(model_dir)

        # Save model configuration
        config = {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "dropout": self.dropout,
            "max_length": self.max_length,
        }

        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(config, f)

    def load_model(self, load_path: str):
        model_dir = os.path.join(self.model_dir, load_path)

        # Load configuration
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            config = json.load(f)

        # Update attributes
        self.model_name = config["model_name"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.dropout = config["dropout"]
        self.max_length = config["max_length"]

        # Initialize tokenizer and model
        from transformers import DebertaV2Tokenizer, AutoTokenizer

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        except:
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_dir)

        self.model = PointwiseDeBERTaModel.from_pretrained(model_dir).to(self.device)
        self.model.eval()


class PointwiseDeBERTaModel(nn.Module):
    """Custom DeBERTa model for pointwise ranking"""

    def __init__(
        self, model_name="microsoft/deberta-v3-base", num_post_features=8, dropout=0.1
    ):
        super().__init__()
        from transformers import DebertaV2Model

        # Load pre-trained DeBERTa model
        self.deberta = DebertaV2Model.from_pretrained(model_name)
        # Freeze the bottom layers to prevent overfitting
        for param in list(self.deberta.parameters())[
            :-4
        ]:  # Freeze all but last 4 layers
            param.requires_grad = False

        self.dropout = nn.Dropout(dropout)

        # Get the hidden size from the model config
        hidden_size = self.deberta.config.hidden_size

        # Feature projection layer
        self.feature_projection = nn.Linear(num_post_features, hidden_size)

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None, features=None):
        # Process text with DeBERTa
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Get [CLS] token representation for the text
        text_embedding = outputs.last_hidden_state[:, 0, :]
        text_embedding = self.dropout(text_embedding)

        # Process features
        feature_embedding = self.feature_projection(features)
        feature_embedding = self.dropout(feature_embedding)

        # Combine text and features
        combined_embedding = torch.cat([text_embedding, feature_embedding], dim=1)

        # Compute regression score
        score = self.regression_head(combined_embedding)

        return score

    def save_pretrained(self, save_directory):
        # Save DeBERTa model
        self.deberta.save_pretrained(save_directory)

        # Save rest of the model
        torch.save(
            {
                "feature_projection.weight": self.feature_projection.weight,
                "feature_projection.bias": self.feature_projection.bias,
                "regression_head.0.weight": self.regression_head[0].weight,
                "regression_head.0.bias": self.regression_head[0].bias,
                "regression_head.1.weight": self.regression_head[1].weight,
                "regression_head.1.bias": self.regression_head[1].bias,
                "regression_head.4.weight": self.regression_head[4].weight,
                "regression_head.4.bias": self.regression_head[4].bias,
            },
            os.path.join(save_directory, "model_weights.pt"),
        )

    @classmethod
    def from_pretrained(cls, load_directory, num_post_features=8, dropout=0.1):
        from transformers import DebertaV2Model

        # Create a new instance of the model first
        model = cls.__new__(cls)
        nn.Module.__init__(model)

        # Load the DeBERTa model from the original pre-trained checkpoint
        model.deberta = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")
        model.dropout = nn.Dropout(dropout)

        # Get hidden size from the model config
        hidden_size = model.deberta.config.hidden_size

        # Initialize the feature projection layer
        model.feature_projection = nn.Linear(num_post_features, hidden_size)

        # Initialize the regression head
        model.regression_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        # Load saved weights
        weights = torch.load(
            os.path.join(load_directory, "model_weights.pt"),
            map_location=torch.device("cpu"),
        )

        # Load weights
        model.feature_projection.weight.data = weights["feature_projection.weight"]
        model.feature_projection.bias.data = weights["feature_projection.bias"]
        model.regression_head[0].weight.data = weights["regression_head.0.weight"]
        model.regression_head[0].bias.data = weights["regression_head.0.bias"]
        model.regression_head[1].weight.data = weights["regression_head.1.weight"]
        model.regression_head[1].bias.data = weights["regression_head.1.bias"]
        model.regression_head[4].weight.data = weights["regression_head.4.weight"]
        model.regression_head[4].bias.data = weights["regression_head.4.bias"]

        return model


class PairwiseDeBERTaLTRR(BaseLTRR):
    """Pairwise Learning to Rank using DeBERTa"""

    class PairwiseDataset(Dataset):
        def __init__(self, data: pd.DataFrame):
            # Extract data
            self.questions = data["question"].values

            # Extract features for action A and action B
            feature_cols = [
                "query_length",
                "query_type",
                "overall_sim_A",
                "avg_sim_A",
                "max_sim_A",
                "var_sim_A",
                "moran_A",
                "cross_retriever_sim_A",
                "overall_sim_B",
                "avg_sim_B",
                "max_sim_B",
                "var_sim_B",
                "moran_B",
                "cross_retriever_sim_B",
            ]
            self.features = data[feature_cols].values

            self.action_A = data["action_A"].values
            self.action_B = data["action_B"].values

            # Convert labels from [-1, 1] to [0, 1] for BCE loss
            self.labels = torch.FloatTensor(
                (data["preference_label_bool"].values + 1) / 2
            )

            # Create masks for NaN values in features
            self.mask = torch.isnan(torch.FloatTensor(self.features))
            # Initialize learnable parameters for NaN values
            self.nan_params = nn.Parameter(torch.randn(self.features.shape[1]))

            # Store mapping from index to qid for evaluation
            self.qids = data["qid"].values

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            # Get text and features
            question = self.questions[idx]
            features = torch.FloatTensor(self.features[idx].copy())

            # Replace NaN values with learnable parameters
            features[self.mask[idx]] = self.nan_params[self.mask[idx]]

            return {
                "question": question,
                "action_A": self.action_A[idx],
                "action_B": self.action_B[idx],
                "features": features,
                "label": self.labels[idx],
                "qid": self.qids[idx],
            }

    def __init__(
        self,
        model_dir: str,
        model_name: str = "microsoft/deberta-v3-base",
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        dropout: float = 0.1,
        max_length: int = 128,
    ):
        super().__init__(model_dir)
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # These will be initialized in the train method
        self.tokenizer = None
        self.model = None

    def _init_model(self):
        """Initialize the model, tokenizer and optimizer"""
        from transformers import DebertaV2Tokenizer, AutoTokenizer

        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except:
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_name)

        # Create model
        self.model = PairwiseDeBERTaModel(
            model_name=self.model_name,
            num_post_features=14,  # 2 query features + 6 post-features for each action
            dropout=self.dropout,
        ).to(self.device)

    class TextDataCollator:
        """Collates batches for the text model"""

        def __init__(self, tokenizer, max_length=128):
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __call__(self, batch):
            # Extract components
            questions = [item["question"] for item in batch]
            actions_A = [str(item["action_A"]) for item in batch]
            actions_B = [str(item["action_B"]) for item in batch]
            features = torch.stack([item["features"] for item in batch])
            labels = torch.stack([item["label"] for item in batch])
            qids = [item["qid"] for item in batch]

            # Create text pairs for both actions
            text_pairs_A = [
                f"Question: {q}, Action: {a}" for q, a in zip(questions, actions_A)
            ]
            text_pairs_B = [
                f"Question: {q}, Action: {a}" for q, a in zip(questions, actions_B)
            ]

            # Tokenize the inputs
            encoded_A = self.tokenizer(
                text_pairs_A,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            encoded_B = self.tokenizer(
                text_pairs_B,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            return {
                "input_ids_A": encoded_A["input_ids"],
                "attention_mask_A": encoded_A["attention_mask"],
                "token_type_ids_A": encoded_A.get("token_type_ids", None),
                "input_ids_B": encoded_B["input_ids"],
                "attention_mask_B": encoded_B["attention_mask"],
                "token_type_ids_B": encoded_B.get("token_type_ids", None),
                "features": features,
                "labels": labels,
                "qids": qids,
            }

    def train(self, train_data: pd.DataFrame, num_epochs: int = 5):
        self._init_model()

        dataset = self.PairwiseDataset(train_data)
        collator = self.TextDataCollator(self.tokenizer, self.max_length)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collator
        )

        # Optimizer with weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.learning_rate
        )

        # Learning rate scheduler
        total_steps = len(dataloader) * num_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )

        # Loss function (BCE for binary preference)
        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        self.model.train()
        for epoch in tqdm(range(num_epochs), desc="Training epochs"):
            total_loss = 0
            total_accuracy = 0

            for batch in tqdm(
                dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
            ):
                # Transfer batch to device
                input_ids_A = batch["input_ids_A"].to(self.device)
                attention_mask_A = batch["attention_mask_A"].to(self.device)
                token_type_ids_A = (
                    batch["token_type_ids_A"].to(self.device)
                    if batch["token_type_ids_A"] is not None
                    else None
                )

                input_ids_B = batch["input_ids_B"].to(self.device)
                attention_mask_B = batch["attention_mask_B"].to(self.device)
                token_type_ids_B = (
                    batch["token_type_ids_B"].to(self.device)
                    if batch["token_type_ids_B"] is not None
                    else None
                )

                features = batch["features"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids_A=input_ids_A,
                    attention_mask_A=attention_mask_A,
                    token_type_ids_A=token_type_ids_A,
                    input_ids_B=input_ids_B,
                    attention_mask_B=attention_mask_B,
                    token_type_ids_B=token_type_ids_B,
                    features=features,
                )

                # Compute loss and update weights
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

                # Calculate accuracy
                pred_probs = torch.sigmoid(outputs.squeeze())
                pred_labels = (pred_probs >= 0.5).float()
                accuracy = (pred_labels == labels).float().mean().item()
                total_accuracy += accuracy

            # Log epoch metrics
            avg_loss = total_loss / len(dataloader)
            avg_accuracy = total_accuracy / len(dataloader)
            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}"
            )

            # Evaluate on a few examples
            if epoch % 2 == 0 or epoch == num_epochs - 1:
                self.model.eval()
                with torch.no_grad():
                    sample_batch = next(iter(dataloader))
                    input_ids_A = sample_batch["input_ids_A"].to(self.device)
                    attention_mask_A = sample_batch["attention_mask_A"].to(self.device)
                    token_type_ids_A = (
                        sample_batch["token_type_ids_A"].to(self.device)
                        if sample_batch["token_type_ids_A"] is not None
                        else None
                    )

                    input_ids_B = sample_batch["input_ids_B"].to(self.device)
                    attention_mask_B = sample_batch["attention_mask_B"].to(self.device)
                    token_type_ids_B = (
                        sample_batch["token_type_ids_B"].to(self.device)
                        if sample_batch["token_type_ids_B"] is not None
                        else None
                    )

                    features = sample_batch["features"].to(self.device)
                    labels = sample_batch["labels"].to(self.device)

                    outputs = self.model(
                        input_ids_A=input_ids_A,
                        attention_mask_A=attention_mask_A,
                        token_type_ids_A=token_type_ids_A,
                        input_ids_B=input_ids_B,
                        attention_mask_B=attention_mask_B,
                        token_type_ids_B=token_type_ids_B,
                        features=features,
                    )

                    pred_probs = torch.sigmoid(outputs.squeeze())

                    # Show a few examples
                    for i in range(min(5, len(pred_probs))):
                        print(
                            f"  Example {i}: Pred = {pred_probs[i].item():.4f}, Label = {labels[i].item():.4f}"
                        )
                self.model.train()

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")

        # Create test dataset and dataloader
        dataset = self.PairwiseDataset(test_data)
        collator = self.TextDataCollator(self.tokenizer, self.max_length)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collator
        )

        # Make predictions
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                # Transfer batch to device
                input_ids_A = batch["input_ids_A"].to(self.device)
                attention_mask_A = batch["attention_mask_A"].to(self.device)
                token_type_ids_A = (
                    batch["token_type_ids_A"].to(self.device)
                    if batch["token_type_ids_A"] is not None
                    else None
                )

                input_ids_B = batch["input_ids_B"].to(self.device)
                attention_mask_B = batch["attention_mask_B"].to(self.device)
                token_type_ids_B = (
                    batch["token_type_ids_B"].to(self.device)
                    if batch["token_type_ids_B"] is not None
                    else None
                )

                features = batch["features"].to(self.device)

                # Get predictions
                outputs = self.model(
                    input_ids_A=input_ids_A,
                    attention_mask_A=attention_mask_A,
                    token_type_ids_A=token_type_ids_A,
                    input_ids_B=input_ids_B,
                    attention_mask_B=attention_mask_B,
                    token_type_ids_B=token_type_ids_B,
                    features=features,
                ).squeeze()

                # Convert logits to probabilities and scale to [-1, 1]
                probs = torch.sigmoid(outputs).cpu().numpy()
                scaled_probs = 2 * probs - 1  # Convert from [0, 1] to [-1, 1]

                predictions.extend(scaled_probs)

        return np.array(predictions)

    def save_model(self, save_path: str):
        model_dir = os.path.join(self.model_dir, save_path)
        os.makedirs(model_dir, exist_ok=True)

        # Save the model
        self.model.save_pretrained(model_dir)

        # Save the tokenizer
        self.tokenizer.save_pretrained(model_dir)

        # Save model configuration
        config = {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "dropout": self.dropout,
            "max_length": self.max_length,
        }

        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(config, f)

    def load_model(self, load_path: str):
        model_dir = os.path.join(self.model_dir, load_path)

        # Load configuration
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            config = json.load(f)

        # Update attributes
        self.model_name = config["model_name"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.dropout = config["dropout"]
        self.max_length = config["max_length"]

        # Initialize tokenizer and model
        from transformers import DebertaV2Tokenizer, AutoTokenizer

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        except:
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_dir)

        self.model = PairwiseDeBERTaModel.from_pretrained(model_dir).to(self.device)
        self.model.eval()


class PairwiseDeBERTaModel(nn.Module):
    """Custom DeBERTa model for pairwise ranking"""

    def __init__(
        self, model_name="microsoft/deberta-v3-base", num_post_features=14, dropout=0.1
    ):
        super().__init__()
        from transformers import DebertaV2Model

        # Load pre-trained DeBERTa model with weight sharing between A and B
        self.deberta = DebertaV2Model.from_pretrained(model_name)

        # Freeze some of the layers to prevent overfitting
        for param in list(self.deberta.parameters())[
            :-4
        ]:  # Freeze all but last 4 layers
            param.requires_grad = False

        self.dropout = nn.Dropout(dropout)

        # Get the hidden size from the model config
        hidden_size = self.deberta.config.hidden_size

        # Feature projection layer
        self.feature_projection = nn.Linear(num_post_features, hidden_size)

        # Action scoring layer - produces a single score for preference A > B
        self.interaction_layer = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),  # Features + A + B
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        input_ids_A,
        attention_mask_A,
        input_ids_B,
        attention_mask_B,
        features,
        token_type_ids_A=None,
        token_type_ids_B=None,
    ):
        # Process action A with DeBERTa
        outputs_A = self.deberta(
            input_ids=input_ids_A,
            attention_mask=attention_mask_A,
            token_type_ids=token_type_ids_A,
        )

        # Process action B with DeBERTa (weight sharing)
        outputs_B = self.deberta(
            input_ids=input_ids_B,
            attention_mask=attention_mask_B,
            token_type_ids=token_type_ids_B,
        )

        # Get [CLS] token representations
        text_embedding_A = outputs_A.last_hidden_state[:, 0, :]
        text_embedding_B = outputs_B.last_hidden_state[:, 0, :]

        text_embedding_A = self.dropout(text_embedding_A)
        text_embedding_B = self.dropout(text_embedding_B)

        # Process features
        feature_embedding = self.feature_projection(features)
        feature_embedding = self.dropout(feature_embedding)

        # Combine text and features for pairwise comparison
        combined_embedding = torch.cat(
            [feature_embedding, text_embedding_A, text_embedding_B], dim=1
        )

        # Compute preference score (preference of A over B)
        pref_score = self.interaction_layer(combined_embedding)

        return pref_score

    def save_pretrained(self, save_directory):
        # Save DeBERTa model
        self.deberta.save_pretrained(save_directory)

        # Save rest of the model
        torch.save(
            {
                "feature_projection.weight": self.feature_projection.weight,
                "feature_projection.bias": self.feature_projection.bias,
                "interaction_layer.0.weight": self.interaction_layer[0].weight,
                "interaction_layer.0.bias": self.interaction_layer[0].bias,
                "interaction_layer.1.weight": self.interaction_layer[1].weight,
                "interaction_layer.1.bias": self.interaction_layer[1].bias,
                "interaction_layer.4.weight": self.interaction_layer[4].weight,
                "interaction_layer.4.bias": self.interaction_layer[4].bias,
            },
            os.path.join(save_directory, "model_weights.pt"),
        )

    @classmethod
    def from_pretrained(cls, load_directory, num_post_features=14, dropout=0.1):
        from transformers import DebertaV2Model

        # Create a new instance of the model first
        model = cls.__new__(cls)
        nn.Module.__init__(model)

        # Load the DeBERTa model from the original pre-trained checkpoint
        model.deberta = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")
        model.dropout = nn.Dropout(dropout)

        # Get hidden size from the model config
        hidden_size = model.deberta.config.hidden_size

        # Initialize the feature projection layer
        model.feature_projection = nn.Linear(num_post_features, hidden_size)

        # Initialize the interaction layer with appropriate dimensions
        model.interaction_layer = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        # Load saved weights
        weights = torch.load(
            os.path.join(load_directory, "model_weights.pt"),
            map_location=torch.device("cpu"),
        )

        # Load weights into the model
        model.feature_projection.weight.data = weights["feature_projection.weight"]
        model.feature_projection.bias.data = weights["feature_projection.bias"]
        model.interaction_layer[0].weight.data = weights["interaction_layer.0.weight"]
        model.interaction_layer[0].bias.data = weights["interaction_layer.0.bias"]
        model.interaction_layer[1].weight.data = weights["interaction_layer.1.weight"]
        model.interaction_layer[1].bias.data = weights["interaction_layer.1.bias"]
        model.interaction_layer[4].weight.data = weights["interaction_layer.4.weight"]
        model.interaction_layer[4].bias.data = weights["interaction_layer.4.bias"]

        return model


class ListwiseDeBERTaLTRR(BaseLTRR):
    """Listwise Learning to Rank using DeBERTa"""

    class ListwiseDataset(Dataset):
        def __init__(self, data: pd.DataFrame, num_actions=7):
            self.num_actions = num_actions

            # Extract data
            self.questions = data["question"].values
            self.qids = data["qid"].values

            # Extract post-retrieval features for all actions
            post_retrieval_features = [
                "overall_sim",
                "avg_sim",
                "max_sim",
                "var_sim",
                "moran",
                "cross_retriever_sim",
            ]

            # Initialize arrays for features and labels
            self.features = np.zeros(
                (len(data), 2 + num_actions * len(post_retrieval_features))
            )
            self.labels = np.zeros((len(data), num_actions))

            # Fill in query-level features
            self.features[:, 0] = data["query_length"].values
            self.features[:, 1] = data["query_type"].values

            # Fill in action-specific features and labels
            feature_idx = 2
            for i in range(num_actions):
                # Add post-retrieval features for this action
                for feature in post_retrieval_features:
                    col_name = f"{feature}_{i}"
                    if col_name in data.columns:
                        self.features[:, feature_idx] = data[col_name].values
                    feature_idx += 1

                # Add delta label for this action
                label_col = f"delta_label_{i}"
                if label_col in data.columns:
                    self.labels[:, i] = data[label_col].values

            # Convert to tensors
            self.features = torch.FloatTensor(self.features)
            self.labels = torch.FloatTensor(self.labels)

            # Create masks for NaN values in features
            self.mask = torch.isnan(self.features)
            # Initialize learnable parameters for NaN values
            self.nan_params = nn.Parameter(torch.randn(self.features.shape[1]))

        def __len__(self):
            return len(self.questions)

        def __getitem__(self, idx):
            # Get text, features and labels
            question = self.questions[idx]
            features = self.features[idx].clone()

            # Replace NaN values with learnable parameters
            features[self.mask[idx]] = self.nan_params[self.mask[idx]]

            labels = self.labels[idx]

            return {
                "question": question,
                "features": features,
                "labels": labels,
                "qid": self.qids[idx],
            }

    def __init__(
        self,
        model_dir: str,
        model_name: str = "microsoft/deberta-v3-base",
        batch_size: int = 8,  # Smaller batch size due to larger model
        learning_rate: float = 1e-5,
        dropout: float = 0.1,
        max_length: int = 128,
        num_actions: int = 7,
    ):
        super().__init__(model_dir)
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.max_length = max_length
        self.num_actions = num_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Track validation metrics
        self.val_ndcg_history = []
        # For calculating diversity penalty
        self.similarity_penalty = 0.1

        # These will be initialized in the train method
        self.tokenizer = None
        self.model = None

    def _init_model(self):
        """Initialize the model, tokenizer and optimizer"""
        from transformers import DebertaV2Tokenizer, AutoTokenizer

        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except:
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_name)

        # Create model
        self.model = ListwiseDeBERTaModel(
            model_name=self.model_name,
            num_post_features=2
            + self.num_actions * 6,  # 2 query features + 6 post-features per action
            num_actions=self.num_actions,
            dropout=self.dropout,
        ).to(self.device)

    class TextDataCollator:
        """Collates batches for the text model"""

        def __init__(self, tokenizer, max_length=128):
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __call__(self, batch):
            # Extract components
            questions = [item["question"] for item in batch]
            features = torch.stack([item["features"] for item in batch])
            labels = torch.stack([item["labels"] for item in batch])
            qids = [item["qid"] for item in batch]

            # Tokenize the questions
            encoded_questions = self.tokenizer(
                questions,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            return {
                "input_ids": encoded_questions["input_ids"],
                "attention_mask": encoded_questions["attention_mask"],
                "token_type_ids": encoded_questions.get("token_type_ids", None),
                "features": features,
                "labels": labels,
                "qids": qids,
            }

    def listwise_loss(self, scores, labels):
        """
        Compute ListNet loss (top-1 approximation)
        """
        # Apply softmax to scores and labels
        scores_softmax = F.softmax(scores, dim=1)

        # Normalize labels (in case they're not already)
        label_max, _ = torch.max(labels, dim=1, keepdim=True)
        label_min, _ = torch.min(labels, dim=1, keepdim=True)

        # Handle case where all labels are the same
        denom = label_max - label_min
        denom[denom == 0] = 1.0

        # Min-max normalize the labels
        labels_normalized = (labels - label_min) / denom
        labels_softmax = F.softmax(labels_normalized, dim=1)

        # Cross-entropy loss between distributions
        loss = -torch.sum(
            labels_softmax * torch.log(scores_softmax + 1e-10), dim=1
        ).mean()
        return loss

    def ndcg_loss(self, scores, labels, k=None):
        """
        Compute NDCG-based loss using LambdaRank principles
        """
        if k is None:
            k = scores.shape[1]  # Use all actions by default

        batch_size = scores.shape[0]

        # Compute all pairwise differences for scores and labels
        score_diffs = scores.unsqueeze(2) - scores.unsqueeze(
            1
        )  # [batch, action, action]
        label_diffs = labels.unsqueeze(2) - labels.unsqueeze(
            1
        )  # [batch, action, action]

        # Only consider pairs where the label difference is non-zero
        label_mask = (label_diffs != 0).float()

        # Get rank positions based on scores
        _, score_indices = torch.sort(scores, dim=1, descending=True)
        ranks = torch.zeros_like(scores)

        # Compute 1/log2(1+rank) discount
        for i in range(batch_size):
            # Create a tensor of positions (1-indexed)
            positions = torch.arange(
                1, scores.shape[1] + 1, device=scores.device
            ).float()
            # Assign positions to the corresponding indices
            for j, idx in enumerate(score_indices[i]):
                ranks[i, idx] = positions[j]

        rank_discounts = 1.0 / torch.log2(1 + ranks)

        # Only consider top k positions for NDCG
        k_mask = (ranks <= k).float()

        # Calculate Delta NDCG for all pairs
        with torch.no_grad():
            # Compute position differences for all pairs
            rank_diff = rank_discounts.unsqueeze(2) - rank_discounts.unsqueeze(
                1
            )  # [batch, action, action]

            # Weight by label difference
            delta_ndcg = (
                torch.abs(rank_diff)
                * torch.sign(label_diffs)
                * label_mask
                * k_mask.unsqueeze(2)
            )

        # Compute pairwise lambdas
        lambda_ij = torch.sigmoid(-score_diffs) * delta_ndcg

        # Sum lambda values for each action
        lambdas = lambda_ij.sum(dim=2) - lambda_ij.sum(dim=1)

        # Compute loss as dot product of lambdas and scores
        loss = (lambdas * scores).sum(dim=1).mean()

        return loss

    def _calculate_ndcg(self, scores, labels):
        """Calculate NDCG for monitoring"""
        batch_size = scores.size(0)
        batch_ndcg = 0

        for i in range(batch_size):
            # Sort scores and get corresponding labels
            _, score_indices = torch.sort(scores[i], descending=True)
            sorted_labels = labels[i][score_indices]

            # Compute DCG
            rel_weights = (2**sorted_labels - 1).clamp(min=0)
            discount = 1.0 / torch.log2(
                torch.arange(2, len(sorted_labels) + 2, device=self.device).float()
            )
            dcg = torch.sum(rel_weights * discount)

            # Compute ideal DCG
            sorted_ideal_labels, _ = torch.sort(labels[i], descending=True)
            ideal_weights = (2**sorted_ideal_labels - 1).clamp(min=0)
            idcg = torch.sum(ideal_weights * discount)

            # Compute NDCG
            ndcg = dcg / (idcg + 1e-10)
            batch_ndcg += ndcg.item()

        return batch_ndcg / batch_size

    def train(self, train_data: pd.DataFrame, num_epochs: int = 5):
        self._init_model()

        # Split data into train/validation
        train_indices = np.random.choice(
            len(train_data), size=int(0.8 * len(train_data)), replace=False
        )
        train_mask = np.zeros(len(train_data), dtype=bool)
        train_mask[train_indices] = True

        train_df = train_data[train_mask].reset_index(drop=True)
        val_df = train_data[~train_mask].reset_index(drop=True)

        print(
            f"Training on {len(train_df)} samples, validating on {len(val_df)} samples"
        )

        train_dataset = self.ListwiseDataset(train_df, num_actions=self.num_actions)
        val_dataset = self.ListwiseDataset(val_df, num_actions=self.num_actions)

        # Create data collator
        collator = self.TextDataCollator(self.tokenizer, self.max_length)

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collator
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collator
        )

        # Optimizer with weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.learning_rate
        )

        # Learning rate scheduler
        total_steps = len(train_dataloader) * num_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )

        best_val_ndcg = 0
        best_diversity = 0
        patience = 7
        patience_counter = 0

        # Training loop
        for epoch in tqdm(range(num_epochs), desc="Training epochs"):
            # Training phase
            self.model.train()
            total_loss = 0
            total_ndcg = 0

            # Add diversity loss scaling that increases over epochs
            diversity_weight = min(0.5, 0.05 * (1 + epoch))  # Gradually increase

            for batch in tqdm(
                train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
            ):
                # Transfer batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = (
                    batch["token_type_ids"].to(self.device)
                    if batch["token_type_ids"] is not None
                    else None
                )
                features = batch["features"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                optimizer.zero_grad()
                scores = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    features=features,
                )

                # Add noise early in training to break symmetry
                if epoch < num_epochs // 2:
                    noise_scale = 0.1 * (1.0 - epoch / (num_epochs // 2))
                    scores = scores + torch.randn_like(scores) * noise_scale

                # Compute loss - alternate between ListNet and LambdaRank loss
                if epoch % 2 == 0:
                    # ListNet loss for initial training
                    ranking_loss = self.listwise_loss(scores, labels)
                else:
                    # NDCG-based loss for fine-tuning
                    ranking_loss = self.ndcg_loss(scores, labels, k=3)

                # Add diversity-promoting loss component
                if epoch > 0:  # Start diversity loss after first epoch
                    # Compute pairwise cosine similarity between score vectors
                    norm_scores = F.normalize(scores, p=2, dim=1)
                    similarity_matrix = torch.mm(
                        norm_scores, norm_scores.transpose(0, 1)
                    )

                    # We want to minimize off-diagonal similarity (maximize diversity)
                    # Create a mask to zero out the diagonal
                    mask = torch.eye(similarity_matrix.size(0), device=self.device) == 0

                    # Compute diversity loss (mean of off-diagonal similarities)
                    diversity_loss = torch.sum(torch.abs(similarity_matrix * mask)) / (
                        mask.sum() + 1e-8
                    )

                    # Combined loss with diversity component
                    loss = ranking_loss + diversity_weight * diversity_loss
                else:
                    diversity_loss = torch.tensor(0.0, device=self.device)
                    loss = ranking_loss

                # Add L2 regularization on scores to prevent extreme values
                score_l2 = 0.01 * torch.mean(scores**2)
                total_loss_val = loss + score_l2

                # Update weights
                total_loss_val.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += ranking_loss.item()

                # Calculate NDCG
                batch_ndcg = self._calculate_ndcg(scores.detach(), labels)
                total_ndcg += batch_ndcg

            # Log epoch metrics
            avg_loss = total_loss / len(train_dataloader)
            avg_ndcg = total_ndcg / len(train_dataloader)
            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, NDCG: {avg_ndcg:.4f}"
            )

            # Validation phase
            self.model.eval()
            val_loss = 0
            val_ndcg = 0
            val_predictions = []

            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    token_type_ids = (
                        batch["token_type_ids"].to(self.device)
                        if batch["token_type_ids"] is not None
                        else None
                    )
                    features = batch["features"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    scores = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        features=features,
                    )

                    # Compute loss
                    loss = self.listwise_loss(scores, labels)
                    val_loss += loss.item()

                    # Calculate NDCG
                    batch_ndcg = self._calculate_ndcg(scores, labels)
                    val_ndcg += batch_ndcg

                    # Store predictions for diversity analysis
                    val_predictions.extend(scores.cpu().numpy())

            val_loss = val_loss / len(val_dataloader)
            val_ndcg = val_ndcg / len(val_dataloader)
            self.val_ndcg_history.append(val_ndcg)

            # Check ranking diversity
            unique_rankings = set()
            for pred in val_predictions:
                ranking = tuple(np.argsort(-pred))
                unique_rankings.add(ranking)

            diversity_ratio = len(unique_rankings) / len(val_predictions)

            # Print metrics
            print(
                f"Validation Loss: {val_loss:.4f}, Validation NDCG: {val_ndcg:.4f}, Diversity: {diversity_ratio:.4f}"
            )
            print(
                f"Validation ranking diversity: {len(unique_rankings)} unique rankings out of {len(val_predictions)} samples ({diversity_ratio*100:.2f}%)"
            )

            # Early stopping considers both NDCG and diversity
            combined_metric = val_ndcg * (
                0.5 + 0.5 * diversity_ratio
            )  # Balance NDCG and diversity

            if (combined_metric > best_val_ndcg * (0.5 + 0.5 * best_diversity)) or (
                diversity_ratio > 0.5 and val_ndcg >= 0.7
            ):
                best_val_ndcg = val_ndcg
                best_diversity = diversity_ratio
                patience_counter = 0

                # Save best model
                best_model_path = os.path.join(
                    self.model_dir, "best_deberta_listwise.pt"
                )
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "hidden_dims": self.model.hidden_dims,
                        "epoch": epoch,
                        "val_ndcg": val_ndcg,
                        "diversity": diversity_ratio,
                    },
                    best_model_path,
                )
                print(
                    f"Saved best model at epoch {epoch+1} with val NDCG {val_ndcg:.4f} and diversity {diversity_ratio:.4f}"
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping at epoch {epoch+1}, no improvement for {patience} epochs"
                    )
                    break

        # Load the best model
        best_model_path = os.path.join(self.model_dir, "best_deberta_listwise.pt")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(
                f"Loaded best model from epoch {checkpoint['epoch']+1} with val NDCG {checkpoint['val_ndcg']:.4f} and diversity {checkpoint['diversity']:.4f}"
            )

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")

        # Create test dataset and dataloader
        dataset = self.ListwiseDataset(test_data, num_actions=self.num_actions)
        collator = self.TextDataCollator(self.tokenizer, self.max_length)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collator
        )

        # Make predictions
        self.model.eval()
        predictions = []
        ranking_diversity = {}
        all_qids = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                # Transfer batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = (
                    batch["token_type_ids"].to(self.device)
                    if batch["token_type_ids"] is not None
                    else None
                )
                features = batch["features"].to(self.device)
                qids = batch["qids"]

                # Get base scores
                base_scores = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    features=features,
                )

                # Add slight randomness to predictions to increase diversity
                for i in range(len(base_scores)):
                    # Get the query ID
                    qid = qids[i]
                    all_qids.append(qid)

                    # Set a query-specific seed for randomness
                    qid_numeric = (
                        int(qid.split("-")[-1]) if "-" in str(qid) else hash(str(qid))
                    )
                    np.random.seed(qid_numeric % 10000)

                    # Add tiny random noise (0.1%) to break ties differently for each query
                    noise = torch.FloatTensor(
                        np.random.normal(0, 0.001, size=base_scores[i].shape)
                    ).to(self.device)
                    scores_with_noise = base_scores[i] + noise

                    # Track ranking diversity
                    ranking = tuple(
                        torch.argsort(scores_with_noise, descending=True).cpu().numpy()
                    )
                    if ranking not in ranking_diversity:
                        ranking_diversity[ranking] = []
                    ranking_diversity[ranking].append(qid)

                    # Store prediction
                    predictions.append(scores_with_noise.cpu().numpy())

        predictions_array = np.array(predictions)

        # Print prediction statistics
        print(f"Prediction statistics - shape: {predictions_array.shape}")
        print(
            f"  min: {predictions_array.min():.6f}, max: {predictions_array.max():.6f}"
        )
        print(
            f"  mean: {predictions_array.mean():.6f}, std: {predictions_array.std():.6f}"
        )

        # Analyze ranking diversity
        print(f"Number of unique rankings in predictions: {len(ranking_diversity)}")
        print(
            f"Diversity ratio: {len(ranking_diversity) / len(predictions_array) * 100:.2f}%"
        )

        # Show most common rankings
        most_common = sorted(
            [(r, len(qids)) for r, qids in ranking_diversity.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        print("\nMost common rankings:")
        for i, (ranking, count) in enumerate(most_common[:5]):
            print(
                f"  #{i+1}: {ranking} - {count} instances ({count/len(predictions_array)*100:.2f}%)"
            )

        return predictions_array

    def save_model(self, save_path: str):
        model_dir = os.path.join(self.model_dir, save_path)
        os.makedirs(model_dir, exist_ok=True)

        # Save the model
        self.model.save_pretrained(model_dir)

        # Save the tokenizer
        self.tokenizer.save_pretrained(model_dir)

        # Save model configuration
        config = {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "dropout": self.dropout,
            "max_length": self.max_length,
            "num_actions": self.num_actions,
            "val_ndcg_history": self.val_ndcg_history,
            "similarity_penalty": self.similarity_penalty,
        }

        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(config, f)

    def load_model(self, load_path: str):
        model_dir = os.path.join(self.model_dir, load_path)

        # Load configuration
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            config = json.load(f)

        # Update attributes
        self.model_name = config["model_name"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.dropout = config["dropout"]
        self.max_length = config["max_length"]
        self.num_actions = config["num_actions"]
        self.val_ndcg_history = config.get("val_ndcg_history", [])
        self.similarity_penalty = config.get("similarity_penalty", 0.1)

        # Initialize tokenizer and model
        from transformers import DebertaV2Tokenizer, AutoTokenizer

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        except:
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_dir)

        self.model = ListwiseDeBERTaModel.from_pretrained(
            model_dir, num_actions=self.num_actions
        ).to(self.device)
        self.model.eval()


class ListwiseDeBERTaModel(nn.Module):
    """Custom DeBERTa model for listwise ranking"""

    def __init__(
        self,
        model_name="microsoft/deberta-v3-base",
        num_post_features=44,  # 2 + 7 actions * 6 post-retrieval features
        num_actions=7,
        dropout=0.1,
    ):
        super().__init__()
        from transformers import DebertaV2Model

        # Load pre-trained DeBERTa model for question encoding
        self.deberta = DebertaV2Model.from_pretrained(model_name)

        # Freeze some of the layers to prevent overfitting
        for param in list(self.deberta.parameters())[
            :-2
        ]:  # Freeze all but last 2 layers
            param.requires_grad = False

        self.dropout = nn.Dropout(dropout)

        # Get the hidden size from the model config
        hidden_size = self.deberta.config.hidden_size

        # Store hidden dimensions for saving/loading
        self.hidden_dims = [hidden_size, hidden_size // 2]

        # Feature projection layer
        self.feature_projection = nn.Linear(num_post_features, hidden_size)

        # Question projection layer
        self.question_projection = nn.Linear(hidden_size, hidden_size)

        # Interaction layer
        self.interaction_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Action scoring layer - produces scores for all actions
        self.scoring_layer = nn.Linear(hidden_size, num_actions)

        # Action-specific diversity heads for better position-wise scores
        self.action_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 2, 1),
                )
                for _ in range(num_actions)
            ]
        )

        # Initialize with different weights to break symmetry
        for i, head in enumerate(self.action_heads):
            torch.manual_seed(42 + i)
            nn.init.xavier_normal_(head[0].weight, gain=0.01)
            nn.init.xavier_normal_(head[-1].weight, gain=0.01)

    def forward(self, input_ids, attention_mask, token_type_ids=None, features=None):
        # Process question with DeBERTa
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Get [CLS] token representation for the question
        question_embedding = outputs.last_hidden_state[:, 0, :]
        question_embedding = self.dropout(question_embedding)
        question_embedding = self.question_projection(question_embedding)

        # Process features
        feature_embedding = self.feature_projection(features)
        feature_embedding = self.dropout(feature_embedding)

        # Combine question and features
        combined_embedding = torch.cat([question_embedding, feature_embedding], dim=1)
        interaction_embedding = self.interaction_layer(combined_embedding)

        # Generate scores for all actions
        main_scores = self.scoring_layer(interaction_embedding)

        # Generate additional scores using action-specific heads
        action_specific_scores = torch.cat(
            [head(interaction_embedding) for head in self.action_heads], dim=1
        )

        # Combine scores (with more weight on the main scoring layer)
        scores = main_scores + 0.2 * action_specific_scores

        return scores

    def save_pretrained(self, save_directory):
        # Save DeBERTa model
        self.deberta.save_pretrained(save_directory)

        # Save rest of the model weights
        model_weights = {
            "feature_projection.weight": self.feature_projection.weight,
            "feature_projection.bias": self.feature_projection.bias,
            "question_projection.weight": self.question_projection.weight,
            "question_projection.bias": self.question_projection.bias,
            "interaction_layer.0.weight": self.interaction_layer[0].weight,
            "interaction_layer.0.bias": self.interaction_layer[0].bias,
            "interaction_layer.1.weight": self.interaction_layer[1].weight,
            "interaction_layer.1.bias": self.interaction_layer[1].bias,
            "scoring_layer.weight": self.scoring_layer.weight,
            "scoring_layer.bias": self.scoring_layer.bias,
        }

        # Add action head weights
        for i, head in enumerate(self.action_heads):
            model_weights[f"action_heads.{i}.0.weight"] = head[0].weight
            model_weights[f"action_heads.{i}.0.bias"] = head[0].bias
            model_weights[f"action_heads.{i}.3.weight"] = head[3].weight
            model_weights[f"action_heads.{i}.3.bias"] = head[3].bias

        torch.save(model_weights, os.path.join(save_directory, "model_weights.pt"))

    @classmethod
    def from_pretrained(
        cls, load_directory, num_post_features=44, num_actions=7, dropout=0.1
    ):
        from transformers import DebertaV2Model, AutoConfig

        # Create config first
        config = AutoConfig.from_pretrained("microsoft/deberta-v3-base")

        # Create model instance
        model = cls(
            model_name="microsoft/deberta-v3-base",  # Use actual model name
            num_post_features=num_post_features,
            num_actions=num_actions,
            dropout=dropout,
        )

        # Load saved weights
        weights = torch.load(
            os.path.join(load_directory, "model_weights.pt"),
            map_location=torch.device("cpu"),
        )

        # Load weights
        model.feature_projection.weight.data = weights["feature_projection.weight"]
        model.feature_projection.bias.data = weights["feature_projection.bias"]
        model.question_projection.weight.data = weights["question_projection.weight"]
        model.question_projection.bias.data = weights["question_projection.bias"]
        model.interaction_layer[0].weight.data = weights["interaction_layer.0.weight"]
        model.interaction_layer[0].bias.data = weights["interaction_layer.0.bias"]
        model.interaction_layer[1].weight.data = weights["interaction_layer.1.weight"]
        model.interaction_layer[1].bias.data = weights["interaction_layer.1.bias"]
        model.scoring_layer.weight.data = weights["scoring_layer.weight"]
        model.scoring_layer.bias.data = weights["scoring_layer.bias"]

        # Load action head weights
        for i, head in enumerate(model.action_heads):
            head[0].weight.data = weights[f"action_heads.{i}.0.weight"]
            head[0].bias.data = weights[f"action_heads.{i}.0.bias"]
            head[3].weight.data = weights[f"action_heads.{i}.3.weight"]
            head[3].bias.data = weights[f"action_heads.{i}.3.bias"]

        return model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_metric", type=str, default="bem", choices=["bem", "ac"])
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="balanced",
        choices=["balanced", "multi-aspect", "comparison", "complex", "open-ended"],
    )
    parser.add_argument(
        "--ltrr_algo",
        type=str,
        required=True,
        choices=[
            "pointwise-xgboost",
            "pointwise-svm",
            "pointwise-neural",
            "pointwise-deberta",
            "pairwise-xgboost",
            "pairwise-svm",
            "pairwise-neural",
            "pairwise-deberta",
            "listwise-listnet",
            "listwise-lambdamart",
            "listwise-deberta",
        ],
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    BASE_METRIC = str(args.base_metric)
    DATASET_TYPE = str(args.dataset_type)
    ltrr_algo = str(args.ltrr_algo)
    LEARNING_METHOD = str(ltrr_algo.split("-")[0])

    CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    DATASET_DIR = os.path.join(
        CUR_DIR_PATH,
        "data",
        "router-training",
        f"{BASE_METRIC}-based",
        f"{DATASET_TYPE}",
        f"{LEARNING_METHOD}",
    )
    TRAIN_DATA_FP = os.path.join(DATASET_DIR, "train.csv")
    TEST_DATA_FP = os.path.join(DATASET_DIR, "test.csv")

    MODEL_SAVE_DIR = os.path.join(
        CUR_DIR_PATH,
        "trained-ltrr-models",
        f"{BASE_METRIC}-based",
        f"{DATASET_TYPE}",
        f"{LEARNING_METHOD}",
    )
    MODEL_NAME = str(ltrr_algo.split("-")[1])  # use when saving and loading
    REPORT_SAVE_FP = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME}-report.json")

    if ltrr_algo == "pointwise-xgboost":
        ltrr_model = PointwiseTreeLTRR(model_dir=MODEL_SAVE_DIR)
    elif ltrr_algo == "pointwise-svm":
        ltrr_model = PointwiseSVMLTRR(model_dir=MODEL_SAVE_DIR)
    elif ltrr_algo == "pointwise-neural":
        ltrr_model = PointwiseNeuralLTRR(model_dir=MODEL_SAVE_DIR)
    elif ltrr_algo == "pointwise-deberta":
        ltrr_model = PointwiseDeBERTaLTRR(model_dir=MODEL_SAVE_DIR)
    elif ltrr_algo == "pairwise-xgboost":
        ltrr_model = PairwiseTreeLTRR(model_dir=MODEL_SAVE_DIR)
    elif ltrr_algo == "pairwise-svm":
        ltrr_model = PairwiseSVMLTRR(model_dir=MODEL_SAVE_DIR)
    elif ltrr_algo == "pairwise-neural":
        ltrr_model = PairwiseNeuralLTRR(model_dir=MODEL_SAVE_DIR)
    elif ltrr_algo == "pairwise-deberta":
        ltrr_model = PairwiseDeBERTaLTRR(model_dir=MODEL_SAVE_DIR)
    elif ltrr_algo == "listwise-listnet":
        ltrr_model = ListwiseNeuralLTRR(model_dir=MODEL_SAVE_DIR)
    elif ltrr_algo == "listwise-lambdamart":
        ltrr_model = ListwiseLambdaMARTLTRR(model_dir=MODEL_SAVE_DIR)
    elif ltrr_algo == "listwise-deberta":
        ltrr_model = ListwiseDeBERTaLTRR(model_dir=MODEL_SAVE_DIR)

    # 1. load train and test data
    train_data = pd.read_csv(TRAIN_DATA_FP)
    test_data = pd.read_csv(TEST_DATA_FP)

    # 2. train ltrr model
    print(f"Training {ltrr_algo} model...")
    ltrr_model.train(train_data)

    # 3. save ltrr model
    print(f"Saving model to {MODEL_SAVE_DIR}...")
    ltrr_model.save_model(MODEL_NAME)

    # 4. load ltrr model
    print(f"Loading model from {MODEL_SAVE_DIR}...")
    ltrr_model.load_model(MODEL_NAME)

    # 5. test ltrr model
    print("Testing model...")
    test_predictions = ltrr_model.predict(test_data)

    # 6. for each test query, get predicted rankings and gold rankings
    print("Computing rankings...")
    predicted_rankings = ltrr_model.get_rankings(test_data)

    # Get gold rankings based on learning method
    gold_rankings = {}
    if LEARNING_METHOD == "pointwise":
        # For pointwise: sort by delta_label
        for qid in test_data["qid"].unique():
            qid_data = test_data[test_data["qid"] == qid]
            gold_rankings[qid] = (
                qid_data["action"]
                .iloc[np.argsort(-qid_data["delta_label"].values)]
                .values
            )

    elif LEARNING_METHOD == "pairwise":
        # For pairwise: use Borda count
        for qid in test_data["qid"].unique():
            qid_data = test_data[test_data["qid"] == qid]

            # Get all unique actions from both action_A and action_B
            all_actions = sorted(
                set(qid_data["action_A"].unique()) | set(qid_data["action_B"].unique())
            )

            # Initialize Borda scores for each action
            borda_scores = {action: 0 for action in all_actions}

            # Count wins for each action based on preference_label_bool
            for _, row in qid_data.iterrows():
                a, b = row["action_A"], row["action_B"]
                if row["preference_label_bool"] == 1:  # A is preferred
                    borda_scores[a] += 1
                else:  # B is preferred
                    borda_scores[b] += 1

            # Sort actions by their Borda scores
            sorted_actions = sorted(
                all_actions, key=lambda x: borda_scores[x], reverse=True
            )
            gold_rankings[qid] = np.array(sorted_actions)

    else:  # listwise
        # For listwise: sort by delta_label_0, delta_label_1, etc.
        for qid in test_data["qid"].unique():
            qid_data = test_data[test_data["qid"] == qid]
            # Get all delta labels for this query
            delta_cols = [
                col for col in qid_data.columns if col.startswith("delta_label_")
            ]
            delta_values = (
                qid_data[delta_cols].iloc[0].values
            )  # All actions for a query are in one row
            gold_rankings[qid] = np.argsort(-delta_values)

    # 7. calculate kendall tau between predicted rankings and gold rankings for all test queries
    tau_scores = []
    for qid in test_data["qid"].unique():
        pred_rank = predicted_rankings[qid]
        gold_rank = gold_rankings[qid]
        tau, _ = kendalltau(pred_rank, gold_rank)
        tau_scores.append(tau)

    mean_tau = np.mean(tau_scores)
    std_tau = np.std(tau_scores)
    min_tau = np.min(tau_scores)
    max_tau = np.max(tau_scores)

    # 7-1. calculate precision@1: correct action in top 1 of the predicted rankings
    precision_at_1 = []
    for qid in test_data["qid"].unique():
        pred_rank = predicted_rankings[qid]
        gold_rank = gold_rankings[qid]
        precision_at_1.append(int(pred_rank[0] == gold_rank[0]))
    mean_precision_at_1 = np.mean(precision_at_1)

    # 8. make the result report and save in the REPORT_SAVE_FP
    report = {
        "model": ltrr_algo,
        "base_metric": BASE_METRIC,
        "dataset_type": DATASET_TYPE,
        "learning_method": LEARNING_METHOD,
        "metrics": {
            "mean_precision_at_1": float(mean_precision_at_1),
            "mean_kendall_tau": float(mean_tau),
            "max_kendall_tau": float(max_tau),
            "min_kendall_tau": float(min_tau),
            "std_kendall_tau": float(std_tau),
        },
        "per_query_predicted_rankings": {
            qid: {
                "predicted_rankings": str(predicted_rankings[qid].tolist()),
                "gold_rankings": str(gold_rankings[qid].tolist()),
            }
            for qid in test_data["qid"].unique()
        },
        # "per_query_metrics": {
        #     qid: {
        #         "kendall_tau": float(tau)
        #     }
        #     for qid, tau in zip(test_data['qid'].unique(), tau_scores)
        # }
    }

    print(f"Saving report to {REPORT_SAVE_FP}...")
    with open(REPORT_SAVE_FP, "w") as f:
        json.dump(report, f, indent=2)
