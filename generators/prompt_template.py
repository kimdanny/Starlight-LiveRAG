SYSTEM_PROMPT = "You are a helpful assistant Falcon3 from TII, try to follow instructions as much as possible."

USER_SUBPROMPT_INFO = """Document {DOC_i}: {DOC_i_TEXT}"""

##############
# rag prompts
##############
# rag with explicit reasoning
USER_PROMPT_WITH_INFO = """You must answer a user question, based on the information (web documents) provided.
Before answering the question, you must conduct your reasoning inside <think> and </think>. During this reasoning step, think about the intent of the user's question and focus on evaluating the relevance and helpfulness of each document to the question. This is important because the information comes from web retrieval and may include irrelevant content. By explicitly reasoning through the relevance and utility of each document, you should seek to ensure that your final answer is accurate and grounded in the pertinent information. After that, think step by step to get to the right answer.
Generation format: you need to surround your reasoning with <think> and </think>, and need to surround your answer with <answer> and </answer>.
For example: <think> your reasoning </think>; <answer> your answer </answer>.

User Question: {QUESTION}

Information:
{INFO_PROMPT}

Show your reasoning between <think> and </think>, and provide your final answer between <answer> and </answer>.
"""

# rag fallback
USER_FALLBACK_PROMPT_WITH_INFO = """Using the information provided below, please answer the user's question. Consider the intent of the user's question and focus on the most relevant information that directly addresses what they're asking. Make sure your response is accurate and based on the provided information.

Question: {QUESTION}

Information:
{INFO_PROMPT}

Provide a clear and direct answer to "{QUESTION}" based on the information above.
"""


#################
# non-rag prompts
#################
# non-rag with explicit reasoning
USER_PROMPT_WITHOUT_INFO = """You must answer the given user question enclosed within <question> and </question>.
Before answering the question, you must conduct your reasoning inside <think> and </think>. During this reasoning step, think about the intent of the user's question then think step by step to get to the right answer.
Generation format: you need to surround your reasoning with <think> and </think>, and need to surround your answer with <answer> and </answer>.
For example: <think> your reasoning </think>; <answer> your answer </answer>.

User Question:
{QUESTION}

Show your reasoning between <think> and </think>, and provide your final answer between <answer> and </answer>.
"""

# non-rag fallback
USER_FALLBACK_PROMPT_WITHOUT_INFO = """Answer the following question with ONLY the answer. No explanations, reasoning, or additional context.
Question: {QUESTION}"""
