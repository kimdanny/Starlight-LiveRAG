"""
Get mean downstream utility for each split for each model
For non-learned (baseline and basic routing methods), load from full-recall data
For learned methods, load from learned model's report
Also supports downstream utility of RAG with a single retriever
"""

import os
import json
import argparse
import random
import statistics

random.seed(42)


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
        "--single_retriever",
        type=str,
        required=False,
        choices=[
            "bm25",
            "bm25_stochastic",
            "bm25_regularize",
            "e5base",
            "e5base_stochastic",
            "e5base_regularize",
        ],
    )
    parser.add_argument(
        "--non_learned_algo",
        type=str,
        required=False,
        choices=[
            "noRetrieval",
            "random",
            "overall_sim",
            "avg_sim",
            "max_sim",
            "var_sim",
            "moran",
        ],
    )
    parser.add_argument(
        "--ltrr_algo",
        type=str,
        required=False,
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


def get_utility_scores_non_learned_models(
    fr_data: dict, model: str, higher_the_better: bool
) -> list:
    utilities = []
    for qid, q_data in fr_data.items():
        action_value_map = {}
        for action_key, action_data in q_data["routing_data"].items():
            action_num = int(action_key.split("_")[-1])
            model_value = action_data["post_retrieval_features"][model]
            metric_value = action_data[f"{BASE_METRIC}_score"]
            action_value_map.update({action_num: (model_value, metric_value)})

        # Sort actions by model value (first element of tuple) in descending order
        sorted_actions = sorted(
            action_value_map.items(),
            key=lambda x: x[1][0],  # Sort by first element of value tuple (model_value)
            reverse=True if higher_the_better else False,  # Descending order
        )
        # Get the metric value (second element of tuple) of the highest-ranked action
        utility_score = sorted_actions[0][1][1]
        utilities.append(utility_score)
    return utilities


if __name__ == "__main__":
    args = get_args()
    BASE_METRIC = str(args.base_metric)
    DATASET_TYPE = str(args.dataset_type)

    CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    FULL_RECALL_TEST_FP = os.path.join(CUR_DIR_PATH, "data", "full-recall", "test.json")

    if args.ltrr_algo is not None:
        LTRR_LEARNING_METHOD = str(args.ltrr_algo.split("-")[0])
        LTRR_MODEL_NAME = str(args.ltrr_algo.split("-")[1])
        LTRR_MODEL_DIR = os.path.join(
            CUR_DIR_PATH,
            "trained-ltrr-models",
            f"{BASE_METRIC}-based",
            f"{DATASET_TYPE}",
            f"{LTRR_LEARNING_METHOD}",
        )
        LTRR_REPORT_FP = os.path.join(LTRR_MODEL_DIR, f"{LTRR_MODEL_NAME}-report.json")

    DS_RESULT_DIR = os.path.join(
        CUR_DIR_PATH,
        "downstream_results",
        f"{BASE_METRIC}-based",
        f"{DATASET_TYPE}",
    )
    os.makedirs(DS_RESULT_DIR, exist_ok=True)

    with open(FULL_RECALL_TEST_FP, "r") as f:
        fr_test = json.load(f)
    f.close()
    # Filter test data based on dataset type
    fr_test = filter_data(fr_test, DATASET_TYPE, split="test")

    if True:
        # get oracle utility
        utilities = []
        for qid, q_data in fr_test.items():
            baseline_utility = q_data[f"baseline_{BASE_METRIC}_score"]
            oracle_utility = max(
                q_data["routing_data"][f"action_{i}"][f"{BASE_METRIC}_score"]
                for i in range(1, 7)
            )
            oracle_utility = max(baseline_utility, oracle_utility)
            utilities.append(oracle_utility)
        avg_utility = statistics.mean(utilities)
        std_utility = statistics.stdev(utilities)
        with open(os.path.join(DS_RESULT_DIR, f"oracle.txt"), "w") as f:
            f.write(f"avg_rag_utility: {round(avg_utility, 4)}\n")
            f.write(f"std_rag_utility: {round(std_utility, 4)}")
        f.close()

    if args.single_retriever is not None:
        model = args.single_retriever

        retriever_to_action_num = {
            "bm25": 1,
            "bm25_stochastic": 2,
            "bm25_regularize": 3,
            "e5base": 4,
            "e5base_stochastic": 5,
            "e5base_regularize": 6,
        }
        action_num = retriever_to_action_num[model]
        utilities = []
        for qid, q_data in fr_test.items():
            utility_score = q_data["routing_data"][f"action_{action_num}"][
                f"{BASE_METRIC}_score"
            ]
            utilities.append(utility_score)
        avg_utility = statistics.mean(utilities)
        std_utility = statistics.stdev(utilities)
        with open(os.path.join(DS_RESULT_DIR, f"{model}.txt"), "w") as f:
            f.write(f"avg_rag_utility: {round(avg_utility, 4)}\n")
            f.write(f"std_rag_utility: {round(std_utility, 4)}")
        f.close()

    # getting downstream utility for non-learned models
    if args.non_learned_algo is not None:
        model = args.non_learned_algo

        if model == "noRetrieval":
            utilities = []
            for qid, q_data in fr_test.items():
                utilities.append(q_data[f"baseline_{BASE_METRIC}_score"])
        elif model == "random":
            utilities = []
            for qid, q_data in fr_test.items():
                random_num = random.randint(0, 6)
                if random_num == 0:
                    utilities.append(q_data[f"baseline_{BASE_METRIC}_score"])
                else:
                    utility_score = q_data["routing_data"][f"action_{random_num}"][
                        f"{BASE_METRIC}_score"
                    ]
                    utilities.append(utility_score)
        # for post-retrieval-feature-based models
        elif model in ["overall_sim", "avg_sim", "max_sim", "moran"]:
            utilities = get_utility_scores_non_learned_models(
                fr_test, model=model, higher_the_better=True
            )
        elif model in ["var_sim"]:
            utilities = get_utility_scores_non_learned_models(
                fr_test, model=model, higher_the_better=False
            )
        else:
            raise ValueError(f"Invalid non-learned model: {model}")

        avg_utility = statistics.mean(utilities)
        std_utility = statistics.stdev(utilities)

        with open(os.path.join(DS_RESULT_DIR, f"{model}.txt"), "w") as f:
            f.write(f"avg_rag_utility: {round(avg_utility, 4)}\n")
            f.write(f"std_rag_utility: {round(std_utility, 4)}")
        f.close()

    # getting downstream utility for learned models
    if args.ltrr_algo is not None:
        model = args.ltrr_algo
        with open(LTRR_REPORT_FP, "r") as f:
            report_dict = json.load(f)
        f.close()
        report_dict: dict = report_dict["per_query_predicted_rankings"]

        utilities = []
        for qid, pred_dict in report_dict.items():
            top_choice = int(pred_dict["predicted_rankings"][1])
            if top_choice != 0:
                utility = fr_test[qid]["routing_data"][f"action_{top_choice}"][
                    f"{BASE_METRIC}_score"
                ]
            else:
                utility = fr_test[qid][f"baseline_{BASE_METRIC}_score"]
            utilities.append(utility)

        avg_utility = statistics.mean(utilities)
        std_utility = statistics.stdev(utilities)

        with open(os.path.join(DS_RESULT_DIR, f"{model}.txt"), "w") as f:
            f.write(f"avg_rag_utility: {round(avg_utility, 4)}\n")
            f.write(f"std_rag_utility: {round(std_utility, 4)}")
        f.close()
