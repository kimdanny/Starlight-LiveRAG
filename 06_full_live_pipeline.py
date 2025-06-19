"""
Full pipeline: from test queries to final outputs
Selected router model:
    Pairwise XGBoost LTRR trained on AC-based utility labels with balanced train/test split
"""

import time
import os
import json
import copy
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
from featurization.pre_retrieval import PreRetrievalFeaturization
from featurization.post_retrieval import PostRetrievalFeaturization
from sklearn.decomposition import PCA
from routers.advanced_routers import PairwiseXGBoostRouter
from retrievers.OpenSearchConnection import OpenSearchConnection
from retrievers.PineconeConnection import PineconeConnection
from retrievers.query_all_retrievers import query_all_retrievers
from retrievers.individual_retrievers import (
    bm25,
    bm25_stochastic,
    bm25_regularize,
    e5base,
    e5base_stochastic,
    e5base_regularize,
)
from generators.falcon_local_v2 import FalconLLM
from validate_answer_jsonl import run_validation

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
LIVE_DAY_DIR = os.path.join(CUR_DIR_PATH, "liveday")
QUESTIONS_FP = os.path.join(LIVE_DAY_DIR, "questions.jsonl")
ANSWER_SAVE_FP = os.path.join(LIVE_DAY_DIR, "answers.jsonl")

FULL_RECALL_FP = os.path.join(LIVE_DAY_DIR, "full-recall-test.json")
ROUTER_TEST_FP = os.path.join(LIVE_DAY_DIR, "router-test.csv")

REDUCED_EMBEDDING_SIZE = 32
PCA_FP = os.path.join(
    CUR_DIR_PATH,
    "data",
    "router-training",
    "ac-based",
    "balanced",
    "pairwise",
    "pca.pkl",
)
SCALER_FP = os.path.join(
    CUR_DIR_PATH,
    "data",
    "router-training",
    "ac-based",
    "balanced",
    "pairwise",
    "normalization_stats.pkl",
)

XGBOOST_DIR = os.path.join(
    CUR_DIR_PATH, "trained-ltrr-models", "ac-based", "balanced", "pairwise"
)

oc = OpenSearchConnection()
pc = PineconeConnection()


def convert_dict_to_list(dict_form: dict) -> list:
    """
    dict form: {"texts": ["", "",], "doc_ids": ["", "",]}
    list form: [{"text": "", "doc_id": ""}, {...}, ]
    """
    list_form = []
    for i in range(len(dict_form["texts"])):
        list_form.append(
            {"doc_id": dict_form["doc_ids"][i], "text": dict_form["texts"][i]}
        )
    return list_form


def reduce_query_embedding(
    test_embeddings: list[list[float]],
):
    """
    transform query embedding with the saved pca model
    Reducing dimension to 32
    """
    # load pca model from PCA_FP
    pca = pickle.load(open(PCA_FP, "rb"))
    reduced_test_embeddings = pca.transform(test_embeddings)
    return reduced_test_embeddings


def make_pointwise_dataset(
    fr_test,
    save_to_files=True,
):
    """
    Each row is a single query-action pair.
    **No delta_labels**

    Columns: qid (1) | question (1) | action (1) | emb_i (32) | query_length (1) | query_type (1)
            | overall_sim (1) | avg_sim (1) | max_sim (1) | var_sim (1) | moran (1) | cross_retriever_sim (1)
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
        # "delta_label",
    ]
    all_columns = base_columns[:3] + embedding_columns + base_columns[3:]
    test_df = pd.DataFrame(columns=all_columns)

    # get all embeddings and perform PCA
    test_embeddings = []
    for qid, q_data in fr_test.items():
        test_embeddings.append(q_data["pre_retrieval_features"]["query_embedding"])
    reduced_test_embeddings = reduce_query_embedding(test_embeddings)  # e.g., (500, 32)
    # update df with embeddings
    test_embedding_dict = {
        f"emb_{i}": reduced_test_embeddings[:, i] for i in range(REDUCED_EMBEDDING_SIZE)
    }
    # Add embeddings to dataframe
    for col_name, values in test_embedding_dict.items():
        test_df[col_name] = values

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
            # "delta_label": 0,
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

            # # Get delta label based on base metric
            # delta_label = action_data[f"{base_metric}_delta"]

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
                # "delta_label": delta_label,
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

    # Per-qid min-max normalization for post-retrieval features (ignore the nan values in action 0)
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

        # # Normalize delta_label (include action 0)
        # delta_min = group["delta_label"].min()
        # delta_max = group["delta_label"].max()
        # if delta_max != delta_min:
        #     group["delta_label"] = (group["delta_label"] - delta_min) / (
        #         delta_max - delta_min
        #     )
        # else:
        #     group["delta_label"] = 0.5

        return group

    # Apply per-qid normalization
    test_df = test_df.groupby("qid").apply(normalize_group).reset_index(drop=True)

    # Dataset-level normalization for query_length (z-score normalization)
    normalization_stats = pickle.load(open(SCALER_FP, "rb"))
    train_query_length_mean = normalization_stats["query_length_mean"]
    train_query_length_std = normalization_stats["query_length_std"]

    # Normalize test set using train set statistics
    test_df["query_length"] = (
        test_df["query_length"] - train_query_length_mean
    ) / train_query_length_std

    # Save datasets
    if save_to_files:
        test_df.to_csv(ROUTER_TEST_FP, index=False)
    return test_df


def make_pairwise_dataset(fr_test) -> str:
    """
    First make pointwise dataset, then make pairwise dataset by transforming pointwise dataset.
    A pairwise dataset transforms each query's retriever list into pairs of retrievers, each labeled based on relative preference:
    Each row corresponds to a pair of actions (retrievers) for the same query (qid).
    Each instance is a pair of retrievers for the same qid. Features are the differences between pairs:
    **No preference labels**

    Columns: qid (1) | question (1) | action_A (1) | action_B (1) | emb_i (32) | query_length (1) | query_type (1)
        | post_feats_A (6) | post_feats_B (6)

    Returns Saved router test file path
    """
    test_pointwise_df = make_pointwise_dataset(fr_test, save_to_files=False)

    # Print some information about the pointwise datasets
    print(f"Pointwise test set shape: {test_pointwise_df.shape}", flush=True)
    print(
        f"Number of queries in test: {test_pointwise_df['qid'].nunique()}", flush=True
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

            # Generate all possible action pairs (both directions)
            for action_a in actions:
                for action_b in actions:
                    if action_a == action_b:
                        continue  # Skip same action pairs

                    # Get data for both actions
                    action_a_data = qid_data[qid_data["action"] == action_a].iloc[0]
                    action_b_data = qid_data[qid_data["action"] == action_b].iloc[0]

                    # # Skip if both delta labels are identical - no preference
                    # if action_a_data["delta_label"] == action_b_data["delta_label"]:
                    #     continue

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

                    # # Add preference label (1 if A > B, else -1)
                    # delta_a = action_a_data["delta_label"]
                    # delta_b = action_b_data["delta_label"]

                    # # Make strictly greater than comparison to make clear preference
                    # row_data["preference_label_bool"] = 1 if delta_a > delta_b else -1

                    pairwise_rows.append(row_data)

        # Create DataFrame
        pairwise_df = pd.DataFrame(pairwise_rows)

        # # Print class distribution to verify we have balanced data
        # preference_counts = pairwise_df["preference_label_bool"].value_counts()
        # print(f"Preference distribution: {preference_counts}")

        # Ensure columns are in the correct order
        base_columns = ["qid", "question", "action_A", "action_B"]
        embedding_columns = [f"emb_{i}" for i in range(REDUCED_EMBEDDING_SIZE)]
        query_columns = ["query_length", "query_type"]
        post_feat_columns = [
            f"{feat}_{suffix}"
            for feat in post_retrieval_features
            for suffix in ["A", "B"]
        ]
        # label_column = ["preference_label_bool"]

        all_columns = (
            base_columns
            + embedding_columns
            + query_columns
            + post_feat_columns
            # + label_column
        )
        pairwise_df = pairwise_df[all_columns]

        return pairwise_df

    # Transform both train and test datasets
    test_pairwise_df = transform_to_pairwise(test_pointwise_df)

    # Save datasets
    test_pairwise_df.to_csv(ROUTER_TEST_FP, index=False)

    print(f"Saved pairwise datasets. Test Shape: {test_pairwise_df.shape}", flush=True)

    # Some diagnostics
    test_actions = set(test_pairwise_df["action_A"].unique()) | set(
        test_pairwise_df["action_B"].unique()
    )
    print(f"Actions in test set: {sorted(test_actions)}", flush=True)

    # Sample a few rows to verify
    print("\nSample rows from test data:", flush=True)
    sample_rows = test_pairwise_df.sample(min(5, len(test_pairwise_df)))
    for _, row in sample_rows.iterrows():
        print(
            f"QID: {row['qid']}, Action A: {row['action_A']}, Action B: {row['action_B']}",
            flush=True,
        )

    return ROUTER_TEST_FP


def step_one() -> None:
    # 0. Load all questions and make full-recall-like json file
    # the full recall like json should look like:
    # {
    #     0: {
    #         "question": "this is question of qid 0",
    #         "routing_data": {
    #             "action_1": {
    #                 "retriever_alias": "bm25",
    #                 "top_hits": [
    #                     {
    #                         "doc_id": "doc id",
    #                         "text": "this is the text of the chunk"
    #                     },
    #                     {
    #                         "doc_id": "doc id",
    #                         "text": "this is the text of the chunk"
    #                     },
    #                     {
    #                         "doc_id": "doc id",
    #                         "text": "this is the text of the chunk"
    #                     },
    #                     {
    #                         "doc_id": "doc id",
    #                         "text": "this is the text of the chunk"
    #                     },
    #                     {
    #                         "doc_id": "doc id",
    #                         "text": "this is the text of the chunk"
    #                     }
    #                 ],
    #                 "post_retrieval_features": {
    #                     "overall_sim": 0.5,
    #                     "avg_sim": 0.5,
    #                     "max_sim": 0.8,
    #                     "var_sim": 0.3,
    #                     "moran": 0.6,
    #                     "cross_retriever_sim": 0.4
    #                 }
    #             },
    #             "action_2": {},
    #             "action_3": {},
    #             "action_4": {},
    #             "action_5": {},
    #             "action_6": {}
    #         },
    #         "pre_retrieval_features": {
    #             "query_embedding": [...],
    #             "query_length": 6.0,
    #             "query_type": 0.0
    #         }
    #     },
    #     1: {},
    #     2: {}
    # }
    answer_jsons = []
    with open(QUESTIONS_FP, "r") as f:
        for line in f:
            # appending "id" and "question"
            answer_jsons.append(line.strip())
    f.close()

    questions_dict = {
        json.loads(q)["id"]: json.loads(q)["question"] for q in answer_jsons
    }
    questions: list[str] = list(questions_dict.values())

    target_dict = {}
    for qid, question in questions_dict.items():
        target_dict[qid] = {}
        target_dict[qid]["question"] = question
        target_dict[qid]["routing_data"] = {
            "action_1": {"retriever_alias": "bm25", "top_k": 5},
            "action_2": {"retriever_alias": "bm25_stochastic", "top_k": 5},
            "action_3": {"retriever_alias": "bm25_regularize", "top_k": 5},
            "action_4": {"retriever_alias": "e5base", "top_k": 5},
            "action_5": {"retriever_alias": "e5base_stochastic", "top_k": 5},
            "action_6": {"retriever_alias": "e5base_regularize", "top_k": 5},
        }

    with open(FULL_RECALL_FP, "w") as f:
        json.dump(target_dict, f, indent=2)
    f.close()

    # 1-1. query all retrievers and fill in routing data in full recall dataset
    with open(FULL_RECALL_FP, "r") as f:
        fr_data: dict = json.load(f)
    f.close()

    target_data = copy.deepcopy(fr_data)
    for qid, entry in tqdm(fr_data.items()):
        question = entry["question"]
        all_retr_res = query_all_retrievers(question, oc=oc, pc=pc)
        for i, (alias, retr_res) in enumerate(all_retr_res.items()):
            assert (
                alias
                == target_data[qid]["routing_data"][f"action_{i+1}"]["retriever_alias"]
            )
            target_data[qid]["routing_data"][f"action_{i+1}"]["top_hits"] = (
                convert_dict_to_list(retr_res)
            )

    # save to file
    with open(FULL_RECALL_FP, "w") as f:
        json.dump(target_data, f, indent=2)
    f.close()

    # 1-2. featurize all the questions (pre and post) and attach to the full recall json file
    # Pre-retrieval features
    with open(FULL_RECALL_FP, "r") as f:
        fr_data: dict = json.load(f)
    f.close()
    pre_featurizer = PreRetrievalFeaturization(
        feature_names=["query_embedding", "query_length", "query_type"]
    )
    pre_retrieval_features = pre_featurizer.batch_featurize(questions)
    for i, qid in enumerate(fr_data):
        fr_data[qid]["pre_retrieval_features"] = {}
        fr_data[qid]["pre_retrieval_features"]["query_embedding"] = (
            pre_retrieval_features["query_embedding"][i]
        )
        fr_data[qid]["pre_retrieval_features"]["query_length"] = pre_retrieval_features[
            "query_length"
        ][i]
        fr_data[qid]["pre_retrieval_features"]["query_type"] = pre_retrieval_features[
            "query_type"
        ][i]

    # save to file
    with open(FULL_RECALL_FP, "w") as f:
        json.dump(fr_data, f, indent=2)
    f.close()

    # post-retrieval features
    with open(FULL_RECALL_FP, "r") as f:
        fr_data: dict = json.load(f)
    f.close()
    target_data = copy.deepcopy(fr_data)

    post_featurizer = PostRetrievalFeaturization(
        feature_names=[
            "overall_sim",
            "avg_sim",
            "max_sim",
            "var_sim",
            "moran",
            "cross_retriever_sim",
        ]
    )
    for qid, entry in tqdm(fr_data.items()):
        question = entry["question"]
        all_actions_docs = {
            action_data["retriever_alias"]: [
                hit["text"] for hit in action_data["top_hits"]
            ]
            for _, action_data in entry["routing_data"].items()
        }
        post_retrieval_features = post_featurizer.featurize(question, all_actions_docs)
        for action_key, action_data in entry["routing_data"].items():
            # Initialize post_retrieval_features dictionary once per action
            target_data[qid]["routing_data"][action_key]["post_retrieval_features"] = {}
            # Add each feature to the dictionary
            for feature_name, feature_value in post_retrieval_features[
                action_data["retriever_alias"]
            ].items():
                target_data[qid]["routing_data"][action_key]["post_retrieval_features"][
                    feature_name
                ] = feature_value

    with open(FULL_RECALL_FP, "w") as f:
        json.dump(target_data, f, indent=2)
    f.close()

    # 1-3. make router test set from the full recall test file
    with open(FULL_RECALL_FP, "r") as f:
        fr_data: dict = json.load(f)
    f.close()
    make_pairwise_dataset(fr_data)
    print("pairwise dataset created\n", flush=True)


def step_two() -> None:
    answer_jsons = []
    with open(QUESTIONS_FP, "r") as f:
        for line in f:
            # appending "id" and "question"
            answer_jsons.append(line.strip())
    f.close()
    questions: list[str] = [json.loads(q)["question"] for q in answer_jsons]

    # 2-1. load router. Let the router load the router test set from saved file.
    #   LTRR predictions happens inside the router and the router only selects the top retrievers and give top actions for the questions
    router = PairwiseXGBoostRouter(model_dir=XGBOOST_DIR, router_test_fp=ROUTER_TEST_FP)
    # 2-2. get routing results for all questions
    actions: list[int] = router.batch_route()
    assert len(actions) == len(questions)

    with open(FULL_RECALL_FP, "r") as f:
        fr_data: dict = json.load(f)
    f.close()

    # # a mapping from action to retriever function
    # retriever_functions = {
    #     1: bm25,
    #     2: bm25_stochastic,
    #     3: bm25_regularize,
    #     4: e5base,
    #     5: e5base_stochastic,
    #     6: e5base_regularize,
    # }

    # 3. Process each question based on the action
    for i, action in enumerate(actions):
        question = questions[i]
        if action == 0:
            # No retrieval
            passages = []
        else:
            top_hits = fr_data[str(i)]["routing_data"][f"action_{action}"]["top_hits"]
            texts = [hit["text"] for hit in top_hits]
            doc_ids = [hit["doc_id"] for hit in top_hits]

            # Retrieve documents
            # retriever_func = retriever_functions[action]
            # if action in [1, 2, 3]:  # Use OpenSearchConnection

            #     texts, doc_ids = retriever_func(question, oc)
            # else:  # Use PineconeConnection
            #     texts, doc_ids = retriever_func(question, pc)

            # Create passages list
            passages = [
                {"passage": text, "doc_IDs": [doc_id]}
                for text, doc_id in zip(texts, doc_ids)
            ]

        # Update the answer_jsons with the retrieved passages
        answer_json = json.loads(answer_jsons[i])
        answer_json["passages"] = passages
        answer_jsons[i] = json.dumps(answer_json)

    # Save the updated answer_jsons to a file
    with open(ANSWER_SAVE_FP, "w") as f:
        for answer_json in answer_jsons:
            f.write(answer_json + "\n")
    f.close()


def step_three() -> None:
    # Initialize the Falcon LLM
    llm = FalconLLM()

    # Load the updated answers with passages
    with open(ANSWER_SAVE_FP, "r") as f:
        answer_jsons = [json.loads(line.strip()) for line in f]
    f.close()

    # Prepare questions and passages for batch processing
    questions = [answer_json["question"] for answer_json in answer_jsons]
    passages_list = [
        [p["passage"] for p in answer_json.get("passages", [])]
        for answer_json in answer_jsons
    ]

    # Query the LLM with the questions and retrieved passages
    responses = llm.batch_query(questions=questions, retrieved_docs_list=passages_list)

    # Attach the final prompt and answer to each JSON object
    for i, response in enumerate(responses):
        answer_json = answer_jsons[i]
        answer_json["final_prompt"] = response["final_prompt"]
        answer_json["answer"] = response["answer"]
        answer_jsons[i] = json.dumps(answer_json)

    # Save the updated answer_jsons to a file
    with open(ANSWER_SAVE_FP, "w") as f:
        for answer_json in answer_jsons:
            f.write(answer_json + "\n")
    f.close()


def step_four() -> None:
    # check jsonl file validity
    assert run_validation(ANSWER_SAVE_FP)
    print("Valid File", flush=True)


if __name__ == "__main__":
    step_one()
    step_two()
    step_three()
    step_four()
