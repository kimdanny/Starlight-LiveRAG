"""
Make a pandas dataframe-based dataset for feature construction.
For basic pointwise version, the dataset has the following columns:
    -> qid (total: 1 col)       (this is not used in feature construction)
    -> question (total: 1 col)  (this is not used in feature construction)
    -> action (total: 1 col)    (this is not used in feature construction)
    - all features (total: 40 cols)
        -> query_embedding (32 col): original 768-dim embedding fit_trasformed to 32-dim (therefore, fitted pca model should be saved with dataset)
        -> query_length (1 col)
        -> query_type (1 col)
        -> overall_sim (1 col)
        -> avg_sim (1 col)
        -> max_sim (1 col)
        -> var_sim (1 col)
        -> moran (1 col)
        -> cross_retriever_sim (1 col)
    - label (total: 1 col)
        either bem_delta (1 col) or ac_delta (1 col)

30 types of datasets:
    - BEM-based
        - balanced
            - pointwise, pairwise, listwise, llm
        - multi-aspect
            - pointwise, pairwise, listwise, llm
        - comparison
            - pointwise, pairwise, listwise, llm
        - complex
            - pointwise, pairwise, listwise, llm
        - open-ended
            - pointwise, pairwise, listwise, llm
    - AC-based
        - balanced
            - pointwise, pairwise, listwise, llm
        - unseen balanced
            - pointwise, pairwise, listwise, llm
        - unseen multi-aspect
            - pointwise, pairwise, listwise, llm
        - unseen comparison
            - pointwise, pairwise, listwise, llm
        - unseen complex
            - pointwise, pairwise, listwise, llm
        - unseen open-ended
            - pointwise, pairwise, listwise, llm
"""

import os
import json
import pandas as pd
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_metric", type=str, default="bem", choices=["bem", "ac"])
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="balanced",
        # If non-balanced, specified answer-type is removed from training set
        # check under qid["question_categories"] and check where "categorization_name" is "answer-type"
        # and then check "category_name" is the specified answer-type
        choices=["balanced", "multi-aspect", "comparison", "complex", "open-ended"],
    )
    parser.add_argument(
        "--learning_method",
        type=str,
        default="pointwise",
        choices=["pointwise", "pairwise", "listwise"],
    )
    parser.add_argument(
        "--reduced_embedding_size",
        type=int,
        default=32,
        help="The size of the reduced embedding",
    )
    args = parser.parse_args()
    return args


args = get_args()
BASE_METRIC = args.base_metric
DATASET_TYPE = args.dataset_type
LEARNING_METHOD = args.learning_method
REDUCED_EMBEDDING_SIZE = int(args.reduced_embedding_size)

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
FULL_RECALL_TRAIN_FP = os.path.join(
    CUR_DIR_PATH,
    "data",
    "full-recall",
    "train.json",
)
FULL_RECALL_TEST_FP = os.path.join(CUR_DIR_PATH, "data", "full-recall", "test.json")
DATASET_SAVE_DIR = os.path.join(
    CUR_DIR_PATH,
    "data",
    "router-training",
    f"{BASE_METRIC}-based",
    f"{DATASET_TYPE}",
    f"{LEARNING_METHOD}",
)
os.makedirs(DATASET_SAVE_DIR, exist_ok=True)
TRAIN_DATA_SAVE_FP = os.path.join(DATASET_SAVE_DIR, "train.csv")
TEST_DATA_SAVE_FP = os.path.join(DATASET_SAVE_DIR, "test.csv")


def filter_data(fr_data, dataset_type: str, split: str) -> dict:
    if dataset_type == "balanced":
        # no filtering
        return fr_data
    elif dataset_type == "multi-aspect":
        if split == "train":
            # filter out qids that do have "multi-aspect" in question_categories
            fr_data = {
                qid: q_data
                for qid, q_data in fr_data.items()
                if "multi-aspect"
                not in [
                    cat["category_name"]
                    for cat in q_data["question_categories"]
                    if cat["categorization_name"] == "answer-type"
                ]
            }
        elif split == "test":
            # only include qids that do have "multi-aspect" in question_categories
            fr_data = {
                qid: q_data
                for qid, q_data in fr_data.items()
                if "multi-aspect"
                in [
                    cat["category_name"]
                    for cat in q_data["question_categories"]
                    if cat["categorization_name"] == "answer-type"
                ]
            }
    elif dataset_type == "comparison":
        if split == "train":
            # filter out qids that do have "comparison" in question_categories
            fr_data = {
                qid: q_data
                for qid, q_data in fr_data.items()
                if "comparison"
                not in [
                    cat["category_name"]
                    for cat in q_data["question_categories"]
                    if cat["categorization_name"] == "answer-type"
                ]
            }
        elif split == "test":
            # only include qids that do have "comparison" in question_categories
            fr_data = {
                qid: q_data
                for qid, q_data in fr_data.items()
                if "comparison"
                in [
                    cat["category_name"]
                    for cat in q_data["question_categories"]
                    if cat["categorization_name"] == "answer-type"
                ]
            }
    elif dataset_type == "complex":
        if split == "train":
            # filter out qids that do have "expert" in user_categories
            fr_data = {
                qid: q_data
                for qid, q_data in fr_data.items()
                if "expert"
                not in [
                    cat["category_name"]
                    for cat in q_data["user_categories"]
                    if cat["categorization_name"] == "user-expertise"
                ]
            }
        elif split == "test":
            # only include qids that do have "expert" in user_categories
            fr_data = {
                qid: q_data
                for qid, q_data in fr_data.items()
                if "expert"
                in [
                    cat["category_name"]
                    for cat in q_data["user_categories"]
                    if cat["categorization_name"] == "user-expertise"
                ]
            }
    elif dataset_type == "open-ended":
        if split == "train":
            # filter out qids that do have "open-ended" in question_categories
            fr_data = {
                qid: q_data
                for qid, q_data in fr_data.items()
                if "open-ended"
                not in [
                    cat["category_name"]
                    for cat in q_data["question_categories"]
                    if cat["categorization_name"] == "answer-type"
                ]
            }
        elif split == "test":
            # only include qids that do have "open-ended" in question_categories
            fr_data = {
                qid: q_data
                for qid, q_data in fr_data.items()
                if "open-ended"
                in [
                    cat["category_name"]
                    for cat in q_data["question_categories"]
                    if cat["categorization_name"] == "answer-type"
                ]
            }
    return fr_data


def reduce_query_embedding(
    train_embeddings: list[list[float]],
    test_embeddings: list[list[float]],
    reduced_size=REDUCED_EMBEDDING_SIZE,
):
    """
    fit_tranform only on the train set.
    """
    pca = PCA(n_components=reduced_size)
    reduced_train_embeddings = pca.fit_transform(train_embeddings)
    # save fitted pca model
    pca_save_fp = os.path.join(DATASET_SAVE_DIR, "pca.pkl")
    with open(pca_save_fp, "wb") as f:
        pickle.dump(pca, f)
    f.close()
    reduced_test_embeddings = pca.transform(test_embeddings)
    return reduced_train_embeddings, reduced_test_embeddings


def make_pointwise_dataset(
    fr_train,
    fr_test,
    base_metric: str = "bem",
    save_to_files=True,
    save_scaling_info=True,
):
    """
    Each row is a single query-action pair.
    Label: delta label either bem_delta or ac_delta
    No special transformation required

    Columns: qid (1) | question (1) | action (1) | emb_i (32) | query_length (1) | query_type (1)
            | overall_sim (1) | avg_sim (1) | max_sim (1) | var_sim (1) | moran (1) | cross_retriever_sim (1) | delta_label (1)
    """
    embedding_columns = [f"emb_{i}" for i in range(REDUCED_EMBEDDING_SIZE)]
    base_columns = [
        "qid",
        "question",
        "action",
        "query_length",
        "query_type",
        "overall_sim",
        "avg_sim",
        "max_sim",
        "var_sim",
        "moran",
        "cross_retriever_sim",
        "delta_label",
    ]
    all_columns = base_columns[:3] + embedding_columns + base_columns[3:]
    train_df = pd.DataFrame(columns=all_columns)
    test_df = pd.DataFrame(columns=all_columns)

    # get all embeddings and perform PCA
    train_embeddings, test_embeddings = [], []
    for qid, q_data in fr_train.items():
        train_embeddings.append(q_data["pre_retrieval_features"]["query_embedding"])
    for qid, q_data in fr_test.items():
        test_embeddings.append(q_data["pre_retrieval_features"]["query_embedding"])
    reduced_train_embeddings, reduced_test_embeddings = reduce_query_embedding(
        train_embeddings, test_embeddings
    )  # e.g., (1999, 32)
    # update df with embeddings
    train_embedding_dict = {
        f"emb_{i}": reduced_train_embeddings[:, i]
        for i in range(REDUCED_EMBEDDING_SIZE)
    }
    test_embedding_dict = {
        f"emb_{i}": reduced_test_embeddings[:, i] for i in range(REDUCED_EMBEDDING_SIZE)
    }
    # Add embeddings to dataframe
    for col_name, values in train_embedding_dict.items():
        train_df[col_name] = values
    for col_name, values in test_embedding_dict.items():
        test_df[col_name] = values

    # fill in the train_df with other values
    train_rows = []
    for qid, q_data in fr_train.items():
        # Get pre-retrieval features
        pre_features = q_data["pre_retrieval_features"]
        query_length = pre_features["query_length"]
        query_type = pre_features["query_type"]
        question = q_data["question"]

        # Add action 0 first
        row_data = {
            "qid": qid,
            "question": question,
            "action": 0,
            "query_length": query_length,
            "query_type": query_type,
            "overall_sim": np.nan,
            "avg_sim": np.nan,
            "max_sim": np.nan,
            "var_sim": np.nan,
            "moran": np.nan,
            "cross_retriever_sim": np.nan,
            "delta_label": 0,
        }
        train_rows.append(row_data)

        for action_key, action_data in q_data["routing_data"].items():
            if not action_data:  # Skip empty action data
                continue
            action_num = int(action_key[-1])

            # Get post-retrieval features
            post_features = action_data["post_retrieval_features"]
            overall_sim = post_features["overall_sim"]
            avg_sim = post_features["avg_sim"]
            max_sim = post_features["max_sim"]
            var_sim = post_features["var_sim"]
            moran = post_features["moran"]
            cross_retriever_sim = post_features["cross_retriever_sim"]

            # Get delta label based on base metric
            delta_label = action_data[f"{base_metric}_delta"]

            # Create row data
            row_data = {
                "qid": qid,
                "question": question,
                "action": action_num,
                "query_length": query_length,
                "query_type": query_type,
                "overall_sim": overall_sim,
                "avg_sim": avg_sim,
                "max_sim": max_sim,
                "var_sim": var_sim,
                "moran": moran,
                "cross_retriever_sim": cross_retriever_sim,
                "delta_label": delta_label,
            }
            train_rows.append(row_data)

    # Convert rows to DataFrame
    train_df_other = pd.DataFrame(train_rows)

    # Create a mapping of qid to its embedding
    train_embedding_map = {
        qid: reduced_train_embeddings[i] for i, qid in enumerate(fr_train.keys())
    }

    # Add embeddings to the dataframe
    for i in range(REDUCED_EMBEDDING_SIZE):
        train_df_other[f"emb_{i}"] = train_df_other["qid"].map(
            lambda x: train_embedding_map[x][i]
        )

    train_df = train_df_other[all_columns]  # Ensure correct column order

    # Fill in test_df with other values
    test_rows = []
    for qid, q_data in fr_test.items():
        # Get pre-retrieval features
        pre_features = q_data["pre_retrieval_features"]
        query_length = pre_features["query_length"]
        query_type = pre_features["query_type"]
        question = q_data["question"]

        # Add action 0 first
        row_data = {
            "qid": qid,
            "question": question,
            "action": 0,
            "query_length": query_length,
            "query_type": query_type,
            "overall_sim": np.nan,
            "avg_sim": np.nan,
            "max_sim": np.nan,
            "var_sim": np.nan,
            "moran": np.nan,
            "cross_retriever_sim": np.nan,
            "delta_label": 0,
        }
        test_rows.append(row_data)

        for action_key, action_data in q_data["routing_data"].items():
            if not action_data:  # Skip empty action data
                continue
            action_num = int(action_key[-1])

            # Get post-retrieval features
            post_features = action_data["post_retrieval_features"]
            overall_sim = post_features["overall_sim"]
            avg_sim = post_features["avg_sim"]
            max_sim = post_features["max_sim"]
            var_sim = post_features["var_sim"]
            moran = post_features["moran"]
            cross_retriever_sim = post_features["cross_retriever_sim"]

            # Get delta label based on base metric
            delta_label = action_data[f"{base_metric}_delta"]

            # Create row data
            row_data = {
                "qid": qid,
                "question": question,
                "action": action_num,
                "query_length": query_length,
                "query_type": query_type,
                "overall_sim": overall_sim,
                "avg_sim": avg_sim,
                "max_sim": max_sim,
                "var_sim": var_sim,
                "moran": moran,
                "cross_retriever_sim": cross_retriever_sim,
                "delta_label": delta_label,
            }
            test_rows.append(row_data)

    # Convert rows to DataFrame
    test_df_other = pd.DataFrame(test_rows)

    # Create a mapping of qid to its embedding
    test_embedding_map = {
        qid: reduced_test_embeddings[i] for i, qid in enumerate(fr_test.keys())
    }

    # Add embeddings to the dataframe
    for i in range(REDUCED_EMBEDDING_SIZE):
        test_df_other[f"emb_{i}"] = test_df_other["qid"].map(
            lambda x: test_embedding_map[x][i]
        )

    test_df = test_df_other[all_columns]  # Ensure correct column order

    # Per-qid min-max normalization for post-retrieval features and delta_label (ignore the nan values in action 0)
    post_retrieval_features = [
        "overall_sim",
        "avg_sim",
        "max_sim",
        "var_sim",
        "moran",
        "cross_retriever_sim",
    ]

    # Function to normalize a group, ignoring NaN values
    def normalize_group(group):
        # Skip action 0 (which has NaN values) for post-retrieval features
        non_nan_mask = ~group[post_retrieval_features].isna().any(axis=1)
        if not non_nan_mask.any():
            return group

        # Get min and max for each feature from non-NaN rows
        feature_mins = group.loc[non_nan_mask, post_retrieval_features].min()
        feature_maxs = group.loc[non_nan_mask, post_retrieval_features].max()

        # Normalize features
        for feature in post_retrieval_features:
            if feature_maxs[feature] != feature_mins[feature]:  # Avoid division by zero
                group.loc[non_nan_mask, feature] = (
                    group.loc[non_nan_mask, feature] - feature_mins[feature]
                ) / (feature_maxs[feature] - feature_mins[feature])

        # Normalize delta_label (include action 0)
        delta_min = group["delta_label"].min()
        delta_max = group["delta_label"].max()
        if delta_max != delta_min:
            group["delta_label"] = (group["delta_label"] - delta_min) / (
                delta_max - delta_min
            )
        else:
            group["delta_label"] = 0.5

        return group

    # Apply per-qid normalization
    train_df = train_df.groupby("qid").apply(normalize_group).reset_index(drop=True)
    test_df = test_df.groupby("qid").apply(normalize_group).reset_index(drop=True)

    # Dataset-level normalization for query_length (z-score normalization)
    # Use train set statistics to normalize both train and test
    train_query_length_mean = train_df["query_length"].mean()
    train_query_length_std = train_df["query_length"].std()

    # Normalize train set
    train_df["query_length"] = (
        train_df["query_length"] - train_query_length_mean
    ) / train_query_length_std

    # Normalize test set using train set statistics
    test_df["query_length"] = (
        test_df["query_length"] - train_query_length_mean
    ) / train_query_length_std

    # Save normalization statistics for future inference
    # For other features, they are per-query normalization, so we don't need to save stats.
    if save_scaling_info:
        normalization_stats = {
            "query_length_mean": train_query_length_mean,
            "query_length_std": train_query_length_std,
        }
        norm_stats_save_fp = os.path.join(DATASET_SAVE_DIR, "normalization_stats.pkl")
        with open(norm_stats_save_fp, "wb") as f:
            pickle.dump(normalization_stats, f)
        f.close()

    # Save datasets
    if save_to_files:
        train_df.to_csv(TRAIN_DATA_SAVE_FP, index=False)
        test_df.to_csv(TEST_DATA_SAVE_FP, index=False)
    return train_df, test_df


def make_pairwise_dataset(fr_train, fr_test, base_metric: str = "bem"):
    """
    First make pointwise dataset, then make pairwise dataset by transforming pointwise dataset.
    A pairwise dataset transforms each query's retriever list into pairs of retrievers, each labeled based on relative preference:
    Each row corresponds to a pair of actions (retrievers) for the same query (qid). The label represents which action in the pair is preferred.
    Each instance is a pair of retrievers for the same qid. Features are the differences between pairs:
    Label: sign(delta_A - delta_B)  # 1 if A > B, else -1
    Transformation Required: Construct pairwise differences explicitly during training.

    Columns: qid (1) | question (1) | action_A (1) | action_B (1) | emb_i (32) | query_length (1) | query_type (1)
        | post_feats_A (6) | post_feats_B (6)
        | preference_label_bool (1 if A > B, else -1) (1)
    """
    train_pointwise_df, test_pointwise_df = make_pointwise_dataset(
        fr_train, fr_test, base_metric=base_metric, save_to_files=False
    )

    # Print some information about the pointwise datasets
    print(f"Pointwise train set shape: {train_pointwise_df.shape}")
    print(f"Pointwise test set shape: {test_pointwise_df.shape}")
    print(f"Number of queries in train: {train_pointwise_df['qid'].nunique()}")
    print(f"Number of queries in test: {test_pointwise_df['qid'].nunique()}")
    print(
        f"Actions per query in train: {train_pointwise_df.groupby('qid')['action'].count().mean():.2f}"
    )

    # Print delta label statistics
    print("\nDelta label statistics from pointwise data:")
    print(
        f"Train min: {train_pointwise_df['delta_label'].min():.4f}, max: {train_pointwise_df['delta_label'].max():.4f}, mean: {train_pointwise_df['delta_label'].mean():.4f}"
    )
    print(
        f"Test min: {test_pointwise_df['delta_label'].min():.4f}, max: {test_pointwise_df['delta_label'].max():.4f}, mean: {test_pointwise_df['delta_label'].mean():.4f}"
    )

    # transform pointwise dataset to pairwise dataset
    def transform_to_pairwise(pointwise_df):
        # Get all unique qids
        qids = pointwise_df["qid"].unique()
        pairwise_rows = []

        post_retrieval_features = [
            "overall_sim",
            "avg_sim",
            "max_sim",
            "var_sim",
            "moran",
            "cross_retriever_sim",
        ]

        # For each qid, create pairs of actions
        for qid in tqdm(qids, desc="Transforming to pairwise"):
            qid_data = pointwise_df[pointwise_df["qid"] == qid]

            # Get the question for this qid (same for all actions)
            question = qid_data["question"].iloc[0]

            # Get all actions for this qid
            actions = qid_data["action"].unique()

            # Sort actions by delta_label value to break any correlations between action indices and performance
            action_delta_map = {
                row["action"]: row["delta_label"] for _, row in qid_data.iterrows()
            }

            # Print debug info for a few queries
            if qid in list(qids)[:3]:  # Print first 3 queries
                print(f"QID {qid} - Action to delta map: {action_delta_map}")

            # Generate all possible action pairs (both directions)
            for action_a in actions:
                for action_b in actions:
                    if action_a == action_b:
                        continue  # Skip same action pairs

                    # Get data for both actions
                    action_a_data = qid_data[qid_data["action"] == action_a].iloc[0]
                    action_b_data = qid_data[qid_data["action"] == action_b].iloc[0]

                    # Skip if both delta labels are identical - no preference
                    if action_a_data["delta_label"] == action_b_data["delta_label"]:
                        continue

                    # Create row data
                    row_data = {
                        "qid": qid,
                        "question": question,
                        "action_A": action_a,
                        "action_B": action_b,
                        "query_length": action_a_data[
                            "query_length"
                        ],  # same for both actions
                        "query_type": action_a_data[
                            "query_type"
                        ],  # same for both actions
                    }

                    # Add embeddings (same for both actions)
                    for k in range(REDUCED_EMBEDDING_SIZE):
                        row_data[f"emb_{k}"] = action_a_data[f"emb_{k}"]

                    # Add post-retrieval features for both actions
                    for feature in post_retrieval_features:
                        row_data[f"{feature}_A"] = action_a_data[feature]
                        row_data[f"{feature}_B"] = action_b_data[feature]

                    # Add preference label (1 if A > B, else -1)
                    delta_a = action_a_data["delta_label"]
                    delta_b = action_b_data["delta_label"]

                    # Make strictly greater than comparison to make clear preference
                    row_data["preference_label_bool"] = 1 if delta_a > delta_b else -1

                    pairwise_rows.append(row_data)

        # Create DataFrame
        pairwise_df = pd.DataFrame(pairwise_rows)

        # Print class distribution to verify we have balanced data
        preference_counts = pairwise_df["preference_label_bool"].value_counts()
        print(f"Preference distribution: {preference_counts}")

        # Ensure columns are in the correct order
        base_columns = ["qid", "question", "action_A", "action_B"]
        embedding_columns = [f"emb_{i}" for i in range(REDUCED_EMBEDDING_SIZE)]
        query_columns = ["query_length", "query_type"]
        post_feat_columns = [
            f"{feat}_{suffix}"
            for feat in post_retrieval_features
            for suffix in ["A", "B"]
        ]
        label_column = ["preference_label_bool"]

        all_columns = (
            base_columns
            + embedding_columns
            + query_columns
            + post_feat_columns
            + label_column
        )
        pairwise_df = pairwise_df[all_columns]

        return pairwise_df

    # Transform both train and test datasets
    train_pairwise_df = transform_to_pairwise(train_pointwise_df)
    test_pairwise_df = transform_to_pairwise(test_pointwise_df)

    # Save datasets
    train_pairwise_df.to_csv(TRAIN_DATA_SAVE_FP, index=False)
    test_pairwise_df.to_csv(TEST_DATA_SAVE_FP, index=False)

    print(
        f"Saved pairwise datasets. Train shape: {train_pairwise_df.shape}, Test shape: {test_pairwise_df.shape}"
    )

    # Generate additional diagnostics
    train_actions = set(train_pairwise_df["action_A"].unique()) | set(
        train_pairwise_df["action_B"].unique()
    )
    print(f"Actions in training set: {sorted(train_actions)}")

    # Sample a few rows to verify
    print("\nSample rows from training data:")
    sample_rows = train_pairwise_df.sample(min(5, len(train_pairwise_df)))
    for _, row in sample_rows.iterrows():
        print(
            f"QID: {row['qid']}, Action A: {row['action_A']}, Action B: {row['action_B']}, Preference: {row['preference_label_bool']}"
        )

    return train_pairwise_df, test_pairwise_df


def make_listwise_dataset(fr_train, fr_test, base_metric: str = "bem"):
    """
    First make pointwise dataset, then make listwise dataset by transforming pointwise dataset.
    Each instance corresponds to all retrievers for a single query (qid).
    Label: delta labels either bem_delta or ac_delta
    Transformation Required: Batch retrievers per query during training to compute listwise losses.

    Columns: qid (1) | question (1) | emb_i (32) | query_length (1) | query_type (1)
        | post_feats_all (6 features x 7 actions = 42 ; column names by {featureName}_{actionNum}) | actions (column names by action_{actionNum}) (7)
        | delta_labels (column names by delta_label_{actionNum}) (7)
    """
    train_pointwise_df, test_pointwise_df = make_pointwise_dataset(
        fr_train, fr_test, base_metric=base_metric, save_to_files=False
    )

    # transform pointwise dataset to listwise dataset
    def transform_to_listwise(pointwise_df):
        # Get all unique qids
        qids = pointwise_df["qid"].unique()
        listwise_rows = []

        # Define post-retrieval features
        post_retrieval_features = [
            "overall_sim",
            "avg_sim",
            "max_sim",
            "var_sim",
            "moran",
            "cross_retriever_sim",
        ]

        # Track how many actions each query has
        action_counts = {}

        # For each qid, create a row with all actions
        for qid in tqdm(qids, desc="Transforming to listwise"):
            qid_data = pointwise_df[pointwise_df["qid"] == qid]

            # Count actions for this query
            num_actions = len(qid_data)
            action_counts[num_actions] = action_counts.get(num_actions, 0) + 1

            # Get the question for this qid
            question = qid_data["question"].iloc[0]

            # Create row data with query features (same for all actions)
            row_data = {
                "qid": qid,
                "question": question,
                "query_length": qid_data["query_length"].iloc[0],
                "query_type": qid_data["query_type"].iloc[0],
            }

            # Add embeddings (same for all actions)
            for k in range(REDUCED_EMBEDDING_SIZE):
                row_data[f"emb_{k}"] = qid_data[f"emb_{k}"].iloc[0]

            # Get all possible action numbers (0-6)
            all_actions = set(range(7))  # Ensure all actions 0-6 are represented
            present_actions = set(qid_data["action"])
            missing_actions = all_actions - present_actions

            # Initialize all action columns with default values
            for action_num in all_actions:
                # Initialize action column
                row_data[f"action_{action_num}"] = action_num

                # Initialize delta_label columns with 0 for missing actions
                row_data[f"delta_label_{action_num}"] = (
                    0.0 if action_num in missing_actions else None
                )

                # Initialize feature columns with NaN for missing actions
                for feature in post_retrieval_features:
                    row_data[f"{feature}_{action_num}"] = (
                        np.nan if action_num in missing_actions else None
                    )

            # Fill in data for present actions
            for _, action_data in qid_data.iterrows():
                action_num = action_data["action"]

                # Add post-retrieval features
                for feature in post_retrieval_features:
                    row_data[f"{feature}_{action_num}"] = action_data[feature]

                # Add delta label
                row_data[f"delta_label_{action_num}"] = action_data["delta_label"]

            listwise_rows.append(row_data)

        # Print action count distribution
        print("Action count distribution:")
        for count, num_queries in sorted(action_counts.items()):
            print(
                f"  {count} actions: {num_queries} queries ({num_queries/len(qids)*100:.1f}%)"
            )

        # Create DataFrame
        listwise_df = pd.DataFrame(listwise_rows)

        # Ensure columns are in the correct order
        base_columns = ["qid", "question"]
        embedding_columns = [f"emb_{i}" for i in range(REDUCED_EMBEDDING_SIZE)]
        query_columns = ["query_length", "query_type"]

        # Get all possible action numbers (0-6)
        action_nums = list(range(7))  # Always use all actions 0-6

        # Create post-retrieval feature columns
        post_feat_columns = [
            f"{feat}_{action_num}"
            for feat in post_retrieval_features
            for action_num in action_nums
        ]

        # Create action and delta label columns
        action_columns = [f"action_{action_num}" for action_num in action_nums]
        delta_label_columns = [
            f"delta_label_{action_num}" for action_num in action_nums
        ]

        all_columns = (
            base_columns
            + embedding_columns
            + query_columns
            + post_feat_columns
            + action_columns
            + delta_label_columns
        )

        # Ensure all columns exist in the dataframe
        for col in all_columns:
            if col not in listwise_df.columns:
                # Add missing columns with NaN values
                listwise_df[col] = np.nan

        # Select columns in the specified order
        listwise_df = listwise_df[all_columns]

        # Validate delta labels
        for action_num in action_nums:
            delta_col = f"delta_label_{action_num}"
            # Print basic statistics
            print(
                f"{delta_col} - missing: {listwise_df[delta_col].isna().sum()}, "
                f"min: {listwise_df[delta_col].min():.4f}, max: {listwise_df[delta_col].max():.4f}"
            )

        return listwise_df

    # Transform both train and test datasets
    train_listwise_df = transform_to_listwise(train_pointwise_df)
    test_listwise_df = transform_to_listwise(test_pointwise_df)

    # Save datasets
    train_listwise_df.to_csv(TRAIN_DATA_SAVE_FP, index=False)
    test_listwise_df.to_csv(TEST_DATA_SAVE_FP, index=False)

    print(
        f"Listwise datasets saved. Train shape: {train_listwise_df.shape}, Test shape: {test_listwise_df.shape}"
    )

    # Verify that each query has all action columns
    for df_name, df in [("Train", train_listwise_df), ("Test", test_listwise_df)]:
        # Check if all required delta label columns exist
        missing_cols = [
            f"delta_label_{i}" for i in range(7) if f"delta_label_{i}" not in df.columns
        ]
        if missing_cols:
            print(f"WARNING: {df_name} dataset is missing columns: {missing_cols}")

        # Sample a few rows to verify
        if len(df) > 0:
            sample_row = df.iloc[0]
            print(f"\nSample row from {df_name.lower()} data:")
            print(f"QID: {sample_row['qid']}")
            for i in range(7):
                if f"delta_label_{i}" in sample_row:
                    print(
                        f"Action {i} delta: {sample_row[f'delta_label_{i}']:.4f}"
                        if not pd.isna(sample_row[f"delta_label_{i}"])
                        else f"Action {i} delta: NaN"
                    )

    return train_listwise_df, test_listwise_df


if __name__ == "__main__":
    # Load full recall data
    with open(FULL_RECALL_TRAIN_FP, "r") as f:
        fr_train_data = json.load(f)
    f.close()
    with open(FULL_RECALL_TEST_FP, "r") as f:
        fr_test_data = json.load(f)
    f.close()

    # Filter train data based on dataset type
    fr_train_data = filter_data(fr_train_data, DATASET_TYPE, split="train")
    fr_test_data = filter_data(fr_test_data, DATASET_TYPE, split="test")

    # Make dataset
    if LEARNING_METHOD == "pointwise":
        make_pointwise_dataset(fr_train_data, fr_test_data, BASE_METRIC)
    elif LEARNING_METHOD == "pairwise":
        make_pairwise_dataset(fr_train_data, fr_test_data, BASE_METRIC)
    elif LEARNING_METHOD == "listwise":
        make_listwise_dataset(fr_train_data, fr_test_data, BASE_METRIC)
