import boto3
from functools import cache
from opensearchpy import OpenSearch, AWSV4SignerAuth, RequestsHttpConnection
from aws.aws_connection import AWS_PROFILE_NAME, AWS_REGION_NAME, get_ssm_value
from typing import Tuple
import numpy as np


class OpenSearchConnection:
    """
    Connection to base BM25 retrieval through OpenSearch Fineweb Index hosted on LiveRAG AWS
    """

    def __init__(self) -> None:
        self.INDEX_NAME = "fineweb10bt-512-0w-e5-base-v2"
        self.client = self.get_client()

    @cache
    def get_client(self):
        credentials = boto3.Session(profile_name=AWS_PROFILE_NAME).get_credentials()
        auth = AWSV4SignerAuth(credentials, region=AWS_REGION_NAME)
        host_name = get_ssm_value(
            "/opensearch/endpoint", profile=AWS_PROFILE_NAME, region=AWS_REGION_NAME
        )
        aos_client = OpenSearch(
            hosts=[{"host": host_name, "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )
        return aos_client

    def query_opensearch(self, query: str, top_k: int = 10) -> dict:
        """Query an OpenSearch index and return the results."""
        results = self.client.search(
            index=self.INDEX_NAME,
            body={"query": {"match": {"text": query}}, "size": top_k},
        )
        return results

    def batch_query_opensearch(
        self, queries: list[str], top_k: int = 10, n_parallel: int = 10
    ) -> list[dict]:
        """Sends a list of queries to OpenSearch and returns the results. Configuration of Connection Timeout might be needed for serving large batches of queries"""
        request = []
        for query in queries:
            req_head = {"index": self.INDEX_NAME}
            req_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text"],
                    }
                },
                "size": top_k,
            }
            request.extend([req_head, req_body])

        return self.client.msearch(body=request)

    def get_texts_docids_scores(
        self, hits: list[dict]
    ) -> Tuple[list[str], list[str], np.ndarray]:
        """
        :param hits
            list of retrieval result, which can get from query_opensearch()['hits']['hits']
        """
        texts = [hit["_source"]["text"] for hit in hits]
        doc_ids = [hit["_source"]["doc_id"] for hit in hits]
        scores = [hit["_score"] for hit in hits]
        return texts, doc_ids, np.array(scores)

    def show_opensearch_results(self, results: dict):
        for match in results["hits"]["hits"]:
            print("chunk:", match["_id"], "score:", match["_score"])
            print(match["_source"]["text"])
            print()
