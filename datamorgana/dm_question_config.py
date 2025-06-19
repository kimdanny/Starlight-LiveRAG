"""
When defining categorizations, keep in mind:
You can create your own categorizations—these are just examples.
Each categorization can include as many categories as you like, as long as their probabilities sum to 1.
The descriptions of the categories are injected into the LLM prompt during question generation.
    To ensure high-quality outputs, it’s important to write them clearly and thoughtfully.
"""

answer_type_categorization = {
    "categorization_name": "answer-type",
    "categories": [
        {
            "name": "factoid",
            "description": "a question seeking a specific, concise piece of information or a fact about a particular subject.",
            "probability": 0.4,
            "is_multi_doc": False,
        },
        {
            "name": "open-ended",
            "description": "a question seeking a detailed or exploratory response, encouraging discussion or elaboration.",
            "probability": 0.2,
            "is_multi_doc": False,
        },
        {
            "name": "multi-aspect",
            "description": "A question about two different aspects of the same entity/concept. For example: 'What are the advantages of AI-powered diagnostics, and what are the associated risks of bias in medical decision-making?', 'How do cryptocurrencies enable financial inclusion, and what are the security risks associated with them?'. The information required to answer the question needs to come from two documents, specifically, the first document must provide information about the first aspect, while the second must provide information about the second aspect.",
            "probability": 0.2,
            "is_multi_doc": True,
        },
        {
            "name": "comparison",
            "description": "a comparison question that requires comparing two related concepts or entities. The comparison must be natural and reasonable, i.e., comparing two entities by a common attribute which is meaningful and relevant to both entities. For example: 'Who is older, Glenn Hughes or Ross Lynch?', 'Are Pizhou and Jiujiang in the same province?', 'Pyotr Ilyich Tchaikovsky and Giuseppe Verdi have this profession in common'. The information required to answer the question needs to come from two documents, specifically, the first document must provide information about the first entity/concept, while the second must provide information about the second entity/concept.",
            "probability": 0.2,
            "is_multi_doc": True,
        },
    ],
}

premise_categorization = {
    "categorization_name": "premise",
    "categories": [
        {
            "name": "without-premise",
            "description": "a question that does not contain any premise or any information about the user.",
            "probability": 0.7,
        },
        {
            "name": "with-premise",
            "description": "a question starting with a very short premise, where the user reveals one's needs or some information about himself.",
            "probability": 0.3,
        },
    ],
}

phrasing_categorization = {
    "categorization_name": "phrasing",
    "categories": [
        {
            "name": "concise and natural",
            "description": "a concise, direct, and natural question consisting of a few words.",
            "probability": 0.25,
        },
        {
            "name": "verbose and natural",
            "description": "a relatively long question consisting of more than 9 words.",
            "probability": 0.25,
        },
        {
            "name": "short search query",
            "description": "a question phrased as a typed web query for search engines (only keywords, without punctuation and without a natural-sounding structure). It consists of less than 7 words.",
            "probability": 0.25,
        },
        {
            "name": "long search query",
            "description": "a question phrased as a typed web query for search engines (only keywords without punctuation and without a natural-sounding structure). It consists of more than 6 words.",
            "probability": 0.25,
        },
    ],
}

linguistic_variation_categorization = {
    "categorization_name": "linguistic-variation",
    "categories": [
        {
            "name": "similar-to-document",
            "description": "a question that is written using the same or similar terminology and phrases appearing in the documents.",
            "probability": 0.5,
        },
        {
            "name": "distant-from-document",
            "description": "a question that is written using the terms completely different from the ones appearing in the documents.",
            "probability": 0.5,
        },
    ],
}
