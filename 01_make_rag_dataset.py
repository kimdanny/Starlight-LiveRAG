"""
Make a RAG dataset with DataMorgana
"""

import json
import time
from typing import Dict, List
from dotenv import load_dotenv
import requests
import os
from datamorgana.dm_question_config import (
    answer_type_categorization,
    premise_categorization,
    phrasing_categorization,
    linguistic_variation_categorization,
)
from datamorgana.dm_user_config import user_expertise_categorization
from sys import exit

load_dotenv()
BASE_URL = "https://api.ai71.ai/v1/"
API_KEY = os.getenv("AI71_API_KEY")

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_SAVE_DIR_PATH = os.path.join(CUR_DIR_PATH, "data", "datamorgana")
TRAIN_SET_FP = os.path.join(DATA_SAVE_DIR_PATH, "train.json")
TEST_SET_FP = os.path.join(DATA_SAVE_DIR_PATH, "test.json")


def check_budget():
    resp = requests.get(
        f"{BASE_URL}check_budget",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    resp.raise_for_status()
    print(json.dumps(resp.json(), indent=4))


def get_all_requests(print_result=False) -> dict:
    """
    Can track all the requests using the get_all_requests endpoint.
    """
    resp = requests.get(
        f"{BASE_URL}get_all_requests",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    resp.raise_for_status()
    result = resp.json()
    if print_result:
        print(json.dumps(result, indent=2))
    return result


def bulk_generate(
    n_questions: int,
    question_categorizations: List[Dict],
    user_categorizations: List[Dict],
):
    resp = requests.post(
        f"{BASE_URL}bulk_generation",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "n_questions": n_questions,
            "question_categorizations": question_categorizations,
            "user_categorizations": user_categorizations,
        },
    )
    resp.raise_for_status()
    request_id = resp.json()["request_id"]
    print(json.dumps(resp.json(), indent=4))

    result = wait_for_generation_to_finish(request_id)
    return result


def wait_for_generation_to_finish(request_id: str):
    first_print = True
    while True:
        resp = requests.get(
            f"{BASE_URL}fetch_generation_results",
            headers={"Authorization": f"Bearer {API_KEY}"},
            params={"request_id": request_id},
        )
        resp.raise_for_status()
        if resp.json()["status"] == "completed":
            print("completed")
            print(json.dumps(resp.json(), indent=4))
            return resp.json()
        else:
            if first_print:
                first_print = False
                print("Waiting for generation to finish...", end="")
            else:
                print(".", end="")
            time.sleep(5)


def append_to_json_file(file_path, new_content):
    # Read existing content
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # If file doesn't exist or is empty/invalid, start with an empty list
        existing_data = []

    # # Ensure both are lists
    # if not isinstance(existing_data, list):
    #     existing_data = [existing_data]

    # if not isinstance(new_content, list):
    #     new_content = [new_content]

    # Combine the lists
    combined_data = existing_data + new_content

    # Write back to file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)

    return len(combined_data)


def generate(split: str, n: int):
    assert split in {"train", "test"}
    save_fp = TRAIN_SET_FP if split == "train" else TEST_SET_FP

    results = bulk_generate(
        n_questions=n,
        question_categorizations=[
            answer_type_categorization,
            premise_categorization,
            phrasing_categorization,
            linguistic_variation_categorization,
        ],
        user_categorizations=[user_expertise_categorization],
    )
    time.sleep(2)
    print("Fetching generated dataset")
    response = requests.get(results["file"])
    qa_pairs = [json.loads(line) for line in response.text.splitlines()]
    final_length = append_to_json_file(save_fp, new_content=qa_pairs)
    print(f"Saved to {save_fp}; Total # of items: {final_length}")
    return final_length


if __name__ == "__main__":
    check_budget()

    if str(input("if you want to generate test set, type 'test': ")) == "test":
        # test set generation
        while True:
            final_length = generate(split="test", n=100)
            if final_length >= 2000:
                break
            time.sleep(10)  # to prevent rate limit

        check_budget()

    if str(input("if you want to generate train set, type 'train': ")) == "train":
        # train set generation
        while True:
            final_length = generate(split="train", n=100)
            if final_length >= 8000:
                break
            time.sleep(10)  # to prevent rate limit

        check_budget()
