"""
Gathering full recall dataset (questions, anwers, all routing actions, and metrics)
Process:
    1) qid, question, gt_docs, gt_answer,
        placeholders for routing_data with retriever_alias, top_k

    2) baseline_answer, baseline_bem_score, baseline_ac_score

    3) routing_data's top_hits for all six actions,
    Take six actions as a batch, and get
        rag_answer, bem_score, ac_score, bem_delta, ac_delta, faithfulness_score, retrieval_precision, retrieval_recall

    4) pre_retrieval_features, and post_retrieval_features

"""

import numpy as np
import copy
from tqdm import tqdm
import os
import json
import argparse
from generators.falcon_local_v2 import FalconLLM
from metrics.string_metrics import BEM, AnswerCorrectness, Faithfulness
import time


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        required=True,
        help="Train or Test split of DataMorgana dataset",
    )
    parser.add_argument(
        "--file_num",
        type=int,
        required=False,
        help="Train set file number to process",
    )
    return parser.parse_args()


ARGS = get_args()
CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DM_DATA_FP = os.path.join(CUR_DIR_PATH, "data", "datamorgana", f"{ARGS.split}.json")

FULL_RECALL_DATA_DIR = os.path.join(CUR_DIR_PATH, "data", "full-recall")
if ARGS.split == "train":
    FULL_RECALL_DATA_FP = os.path.join(
        FULL_RECALL_DATA_DIR, f"{ARGS.split}-{ARGS.file_num}.json"
    )
else:
    FULL_RECALL_DATA_FP = os.path.join(FULL_RECALL_DATA_DIR, f"{ARGS.split}.json")


##########
# Step 1
##########
def step_one() -> None:
    """
    1) qid, question, question_categories, user_categories, gt_docs, gt_answer,
        placeholders for routing_data with retriever_alias, top_k
    """
    target_dict = {}

    with open(DM_DATA_FP, "r") as dm_f:
        dm_data: list[dict] = json.load(dm_f)

    for i, entry in enumerate(tqdm(dm_data)):
        qid = f"{ARGS.split}-{str(i).zfill(4)}"
        target_dict[qid] = {}
        target_dict[qid]["question"] = entry["question"]
        target_dict[qid]["question_categories"] = entry["question_categories"]
        target_dict[qid]["user_categories"] = entry["user_categories"]
        target_dict[qid]["gt_docs"] = entry["document_ids"]
        target_dict[qid]["gt_answer"] = entry["answer"]
        target_dict[qid]["routing_data"] = {
            "action_1": {"retriever_alias": "bm25", "top_k": 5},
            "action_2": {"retriever_alias": "bm25_stochastic", "top_k": 5},
            "action_3": {"retriever_alias": "bm25_regularize", "top_k": 5},
            "action_4": {"retriever_alias": "e5base", "top_k": 5},
            "action_5": {"retriever_alias": "e5base_stochastic", "top_k": 5},
            "action_6": {"retriever_alias": "e5base_regularize", "top_k": 5},
        }
    with open(FULL_RECALL_DATA_FP, "w") as fr_f:
        json.dump(target_dict, fr_f, indent=2)
    fr_f.close()
    dm_f.close()


##########
# Step 2
##########
def step_two(substep: str = "all") -> None:
    """
    baseline_answer, baseline_bem_score, baseline_ac_score
    """
    # 2.1 batch inference baseline_answer
    if substep in {"1", "all"}:
        with open(FULL_RECALL_DATA_FP, "r") as f:
            fr_data: dict = json.load(f)
        f.close()

        llm = FalconLLM()
        questions = []
        for qid in fr_data:
            questions.append(fr_data[qid]["question"])

        batch_response = llm.batch_query(questions=questions, batch_size=8)
        assert len(batch_response) == len(fr_data)
        for i, res in enumerate(batch_response):
            qid = f"{ARGS.split}-{str(i).zfill(4)}"
            fr_data[qid]["baseline_answer"] = res["answer"]
            fr_data[qid]["baseline_answer_details"] = {
                "fallback": str(res["used_fallback"]),
                "full_prompt": res["final_prompt"],
                "full_response": res["response"],
            }

        # save to file
        with open(FULL_RECALL_DATA_FP, "w") as f:
            json.dump(fr_data, f, indent=2)
        f.close()

    # 2.2 batch eval baseline_bem_score, baseline_ac_score
    if substep in {"2", "all"}:
        with open(FULL_RECALL_DATA_FP, "r") as f:
            fr_data: dict = json.load(f)
        f.close()

        bem_eval_data: list[dict] = []
        for qid, entry in fr_data.items():
            bem_eval_data.append(
                {
                    "question": entry["question"],
                    "reference": entry["gt_answer"],
                    "candidate": entry["baseline_answer"],
                }
            )
        ac_eval_data = {
            "question": [e["question"] for e in bem_eval_data],
            "ground_truth": [e["reference"] for e in bem_eval_data],
            "answer": [e["candidate"] for e in bem_eval_data],
        }
        # eval models
        bem = BEM(use_gpu=True)
        ac = AnswerCorrectness()

        bem_scores = bem.batch_evaluate(bem_eval_data, batch_size=16)
        ac_scores = ac.batch_evaluate(ac_eval_data)

        assert len(bem_scores) == len(ac_scores) == len(fr_data)

        for i, (bem_score, ac_score) in enumerate(zip(bem_scores, ac_scores)):
            qid = f"{ARGS.split}-{str(i).zfill(4)}"
            fr_data[qid]["baseline_bem_score"] = bem_score
            fr_data[qid]["baseline_ac_score"] = ac_score

        # save to file
        with open(FULL_RECALL_DATA_FP, "w") as f:
            json.dump(fr_data, f, indent=2)
        f.close()


def step_two_run_ac_from_middle(start_qid: str, batch_size: int = 200):
    with open(FULL_RECALL_DATA_FP, "r") as f:
        fr_data: dict = json.load(f)
    f.close()

    # Find the starting index based on start_qid
    start_index = list(fr_data.keys()).index(start_qid)

    # Prepare evaluation data starting from start_qid
    qids = list(fr_data.keys())[start_index:]
    total_batches = (len(qids) + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, len(qids))
        batch_qids = qids[batch_start:batch_end]

        bem_eval_data: list[dict] = []
        for qid in batch_qids:
            entry = fr_data[qid]
            bem_eval_data.append(
                {
                    "question": entry["question"],
                    "reference": entry["gt_answer"],
                    "candidate": entry["baseline_answer"],
                }
            )
        ac_eval_data = {
            "question": [e["question"] for e in bem_eval_data],
            "ground_truth": [e["reference"] for e in bem_eval_data],
            "answer": [e["candidate"] for e in bem_eval_data],
        }
        del bem_eval_data

        # eval models
        ac = AnswerCorrectness()
        ac_scores = ac.batch_evaluate(ac_eval_data)

        # Ensure we only update from start_qid onwards
        assert len(ac_scores) == len(batch_qids)

        for i, ac_score in enumerate(ac_scores):
            qid = batch_qids[i]
            fr_data[qid]["baseline_ac_score"] = ac_score

        # save to file
        with open(FULL_RECALL_DATA_FP, "w") as f:
            json.dump(fr_data, f, indent=2)
        f.close()

        # Sleep for 5 seconds to avoid rate limit
        time.sleep(5)

        # Reload the file to ensure continuity
        with open(FULL_RECALL_DATA_FP, "r") as f:
            fr_data = json.load(f)
        f.close()


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


def process_questions_with_llm(fr_data: dict, llm: FalconLLM) -> dict:
    """
    Collect all (question, docs) pairs for batch processing and update target data.
    """
    target_data = copy.deepcopy(fr_data)
    questions_batch = []  # Collect all questions
    docs_batch: list[list] = []  # Collect all docs
    qid_action_pairs = []  # To track which qid and action each response belongs to

    # Collect all (question, docs) pairs
    for qid, question_data in fr_data.items():
        question = question_data["question"]
        routing_data = question_data["routing_data"]

        for action_key, action_data in routing_data.items():
            if "top_hits" in action_data:
                docs: list[str] = [hit["text"] for hit in action_data["top_hits"]]
                questions_batch.append(question)
                docs_batch.append(docs)
                qid_action_pairs.append((qid, action_key))

    # Query the LLM with the batch
    responses = llm.batch_query(
        questions=questions_batch, retrieved_docs_list=docs_batch
    )

    # Update routing data with responses
    for (qid, action_key), response in zip(qid_action_pairs, responses):
        routing_data = target_data[qid]["routing_data"]
        routing_data[action_key]["final_prompt"] = response.get("final_prompt", "")
        routing_data[action_key]["rag_answer"] = response.get("answer", "")

    return target_data


def step_three(substep: str = "all") -> None:
    """
    routing_data's top_hits for all six actions,
    Take six actions as a batch, and get
        rag_answer, bem_score, ac_score, bem_delta, ac_delta, faithfulness_score, retrieval_precision, retrieval_recall
    """
    # 3.1 attach retrieval results
    if substep in {"1", "all"}:
        from retrievers.OpenSearchConnection import OpenSearchConnection
        from retrievers.PineconeConnection import PineconeConnection
        from retrievers.query_all_retrievers import query_all_retrievers

        oc = OpenSearchConnection()
        pc = PineconeConnection()

        with open(FULL_RECALL_DATA_FP, "r") as f:
            fr_data: dict = json.load(f)
        f.close()

        target_data = copy.deepcopy(fr_data)
        for qid, entry in tqdm(fr_data.items()):
            question = entry["question"]
            all_retr_res = query_all_retrievers(question, oc=oc, pc=pc)
            for i, (alias, retr_res) in enumerate(all_retr_res.items()):
                assert (
                    alias
                    == target_data[qid]["routing_data"][f"action_{i+1}"][
                        "retriever_alias"
                    ]
                )
                target_data[qid]["routing_data"][f"action_{i+1}"]["top_hits"] = (
                    convert_dict_to_list(retr_res)
                )

        # save to file
        with open(FULL_RECALL_DATA_FP, "w") as f:
            json.dump(target_data, f, indent=2)
        f.close()

    # 3.2 rag_answer: per action --> get "final_prompt" and "rag_answer"
    if substep in {"2", "all"}:
        llm = FalconLLM()

        with open(FULL_RECALL_DATA_FP, "r") as f:
            fr_data: dict = json.load(f)
        f.close()

        # Define batch size
        batch_size = 100  # Adjust the batch size as needed
        qids = list(fr_data.keys())
        total_batches = (len(qids) + batch_size - 1) // batch_size

        for batch_num in range(total_batches):
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, len(qids))
            batch_qids = qids[batch_start:batch_end]

            # Prepare batch data
            batch_data = {qid: fr_data[qid] for qid in batch_qids}

            # Process batch
            target_data = process_questions_with_llm(batch_data, llm)

            # Update fr_data with processed batch
            fr_data.update(target_data)

            # Save updated fr_data to file
            with open(FULL_RECALL_DATA_FP, "w") as f:
                json.dump(fr_data, f, indent=2)
            f.close()

            # Re-open the file to ensure continuity
            with open(FULL_RECALL_DATA_FP, "r") as f:
                fr_data = json.load(f)
            f.close()

    # Following substeps: metrics: bem_score, ac_score, bem_delta, ac_delta,
    # faithfulness_score, retrieval_precision, retrieval_recall

    # 3.3 BEM evaluation
    if substep in {"3", "all"}:
        print("Processing BEM scores...", flush=True)
        with open(FULL_RECALL_DATA_FP, "r") as f:
            fr_data: dict = json.load(f)
        f.close()

        # eval model
        bem = BEM(use_gpu=True)
        target_data = copy.deepcopy(fr_data)

        # Prepare data for batch evaluation
        bem_eval_data = []
        qid_action_pairs = []  # To track which qid and action each score belongs to

        for qid, entry in fr_data.items():
            question = entry["question"]
            baseline_bem_score = entry["baseline_bem_score"]

            for action_key, action_data in entry["routing_data"].items():
                rag_answer = action_data["rag_answer"]

                # Collect data for evaluation
                bem_eval_data.append(
                    {
                        "question": question,
                        "reference": entry["gt_answer"],
                        "candidate": rag_answer,
                    }
                )
                qid_action_pairs.append((qid, action_key))

        # Perform batch evaluation
        print("BEM batch eval", flush=True)
        bem_scores = bem.batch_evaluate(bem_eval_data)

        # Update target_data with scores
        for (qid, action_key), bem_score in zip(qid_action_pairs, bem_scores):
            baseline_bem_score = target_data[qid]["baseline_bem_score"]
            bem_delta = bem_score - baseline_bem_score

            # Update routing data
            target_data[qid]["routing_data"][action_key]["bem_score"] = bem_score
            target_data[qid]["routing_data"][action_key]["bem_delta"] = bem_delta

        # save to file
        with open(FULL_RECALL_DATA_FP, "w") as f:
            json.dump(target_data, f, indent=2)
        f.close()

    # 3.4 Retrieval precision and recall
    if substep in {"4", "all"}:
        print("Processing retrieval precision and recall...", flush=True)
        with open(FULL_RECALL_DATA_FP, "r") as f:
            fr_data: dict = json.load(f)
        f.close()

        target_data = copy.deepcopy(fr_data)

        for qid, entry in target_data.items():
            gt_docs: set[str] = set(entry["gt_docs"])

            for action_key, action_data in entry["routing_data"].items():
                # Retrieval metrics calculation
                retrieved_docs = set([hit["doc_id"] for hit in action_data["top_hits"]])
                n_overlapping_docs = len(gt_docs.intersection(retrieved_docs))
                precision = (
                    n_overlapping_docs / len(retrieved_docs) if retrieved_docs else 0
                )
                recall = n_overlapping_docs / len(gt_docs) if gt_docs else 0

                action_data["retrieval_precision"] = precision
                action_data["retrieval_recall"] = recall

        # save to file
        with open(FULL_RECALL_DATA_FP, "w") as f:
            json.dump(target_data, f, indent=2)
        f.close()

    # 3.5 Answer Correctness evaluation
    if substep in {"5", "all"}:
        print("Processing Answer Correctness scores...", flush=True)
        with open(FULL_RECALL_DATA_FP, "r") as f:
            fr_data: dict = json.load(f)
        f.close()

        # eval model
        ac = AnswerCorrectness()
        target_data = copy.deepcopy(fr_data)

        # Prepare data for batch evaluation
        ac_eval_data = {"question": [], "ground_truth": [], "answer": []}
        qid_action_pairs = []  # To track which qid and action each score belongs to
        save_interval = 600  # Save results every 600 items

        for qid, entry in fr_data.items():
            question = entry["question"]

            for action_key, action_data in entry["routing_data"].items():
                rag_answer = action_data["rag_answer"]

                ac_eval_data["question"].append(question)
                ac_eval_data["ground_truth"].append(entry["gt_answer"])
                ac_eval_data["answer"].append(rag_answer)
                qid_action_pairs.append((qid, action_key))

                # Process in chunks of save_interval
                if len(qid_action_pairs) >= save_interval:
                    # Perform batch evaluation for current chunk
                    print(
                        f"AC batch eval for {len(qid_action_pairs)} items", flush=True
                    )
                    ac_scores = ac.batch_evaluate(ac_eval_data, max_workers=8)

                    # Update target_data with scores
                    for (qid, action_key), ac_score in zip(qid_action_pairs, ac_scores):
                        baseline_ac_score = target_data[qid]["baseline_ac_score"]
                        ac_delta = ac_score - baseline_ac_score

                        # Update routing data
                        target_data[qid]["routing_data"][action_key][
                            "ac_score"
                        ] = ac_score
                        target_data[qid]["routing_data"][action_key][
                            "ac_delta"
                        ] = ac_delta

                    # Save intermediate results
                    print(
                        f"Saving intermediate results after processing {len(qid_action_pairs)} items",
                        flush=True,
                    )
                    with open(FULL_RECALL_DATA_FP, "w") as f:
                        json.dump(target_data, f, indent=2)
                    f.close()

                    # Reset for next chunk
                    ac_eval_data = {"question": [], "ground_truth": [], "answer": []}
                    qid_action_pairs = []

        # Process any remaining items
        if qid_action_pairs:
            print(
                f"AC batch eval for remaining {len(qid_action_pairs)} items", flush=True
            )
            ac_scores = ac.batch_evaluate(ac_eval_data, max_workers=12)

            # Update target_data with scores
            for (qid, action_key), ac_score in zip(qid_action_pairs, ac_scores):
                baseline_ac_score = target_data[qid]["baseline_ac_score"]
                ac_delta = ac_score - baseline_ac_score

                # Update routing data
                target_data[qid]["routing_data"][action_key]["ac_score"] = ac_score
                target_data[qid]["routing_data"][action_key]["ac_delta"] = ac_delta

        # Save final results
        with open(FULL_RECALL_DATA_FP, "w") as f:
            json.dump(target_data, f, indent=2)
        f.close()

    # 3.6 Faithfulness evaluation
    if substep in {"6", "all"}:
        print("Processing Faithfulness scores...", flush=True)
        with open(FULL_RECALL_DATA_FP, "r") as f:
            fr_data: dict = json.load(f)
        f.close()

        # eval model
        faith = Faithfulness()
        target_data = copy.deepcopy(fr_data)

        # Prepare data for batch evaluation
        faith_eval_data = {"question": [], "answer": [], "contexts": []}
        qid_action_pairs = []  # To track which qid and action each score belongs to

        for qid, entry in fr_data.items():
            question = entry["question"]

            for action_key, action_data in entry["routing_data"].items():
                rag_answer = action_data["rag_answer"]
                top_hits = action_data["top_hits"]

                faith_eval_data["question"].append(question)
                faith_eval_data["answer"].append(rag_answer)
                faith_eval_data["contexts"].append([hit["text"] for hit in top_hits])
                qid_action_pairs.append((qid, action_key))

        # Perform batch evaluation
        print("Faith batch eval", flush=True)
        faithfulness_scores = faith.batch_evaluate(faith_eval_data)

        # Update target_data with scores
        for (qid, action_key), faithfulness_score in zip(
            qid_action_pairs, faithfulness_scores
        ):
            # Update routing data
            target_data[qid]["routing_data"][action_key][
                "faithfulness_score"
            ] = faithfulness_score

        # save to file
        with open(FULL_RECALL_DATA_FP, "w") as f:
            json.dump(target_data, f, indent=2)
        f.close()


def step_four(substep: str = "all") -> None:
    """
    pre_retrieval_features, and post_retrieval_features
    Store raw features not normalized
    """
    # 4.1 pre-retrieval features
    if substep in {"1", "all"}:
        from featurization.pre_retrieval import PreRetrievalFeaturization

        with open(FULL_RECALL_DATA_FP, "r") as f:
            fr_data: dict = json.load(f)
        f.close()

        prf = PreRetrievalFeaturization(
            feature_names=["query_embedding", "query_length", "query_type"]
        )
        questions = [fr_data[qid]["question"] for qid in fr_data]
        pre_retrieval_features = prf.batch_featurize(questions)
        # example pre_retrieval_features:
        # {
        #     "query_embedding": [[0.1, 0.2, 0.3, ...], [0.1, 0.2, 0.3, ...], ...],
        #     "query_length": [1, 2, 3, ...],
        #     "query_type": [1, 0, 1, ...]
        # }
        for i, qid in enumerate(fr_data):
            fr_data[qid]["pre_retrieval_features"] = {}
            fr_data[qid]["pre_retrieval_features"]["query_embedding"] = (
                pre_retrieval_features["query_embedding"][i]
            )
            fr_data[qid]["pre_retrieval_features"]["query_length"] = (
                pre_retrieval_features["query_length"][i]
            )
            fr_data[qid]["pre_retrieval_features"]["query_type"] = (
                pre_retrieval_features["query_type"][i]
            )

        # save to file
        with open(FULL_RECALL_DATA_FP, "w") as f:
            json.dump(fr_data, f, indent=2)
        f.close()

    # 4.2 post-retrieval features
    if substep in {"2", "all"}:
        from featurization.post_retrieval import PostRetrievalFeaturization

        with open(FULL_RECALL_DATA_FP, "r") as f:
            fr_data: dict = json.load(f)
        f.close()
        target_data = copy.deepcopy(fr_data)

        prf = PostRetrievalFeaturization(
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
            post_retrieval_features = prf.featurize(question, all_actions_docs)
            for action_key, action_data in entry["routing_data"].items():
                # Initialize post_retrieval_features dictionary once per action
                target_data[qid]["routing_data"][action_key][
                    "post_retrieval_features"
                ] = {}
                # Add each feature to the dictionary
                for feature_name, feature_value in post_retrieval_features[
                    action_data["retriever_alias"]
                ].items():
                    target_data[qid]["routing_data"][action_key][
                        "post_retrieval_features"
                    ][feature_name] = feature_value

        # save to file
        with open(FULL_RECALL_DATA_FP, "w") as f:
            json.dump(target_data, f, indent=2)
        f.close()


if __name__ == "__main__":
    print("step 1", flush=True)
    step_one()

    print("step 2", flush=True)
    step_two()

    print("step 3", flush=True)
    step_three()

    print("step 4", flush=True)
    step_four()
