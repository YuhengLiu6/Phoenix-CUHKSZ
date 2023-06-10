# from rank_bm25 import BM25Okapi
# import numpy as np
# corpus = [
#     "Hello there good man!",
#     "It is quite windy in London",
#     "How is the weather today?"
# ]

# tokenized_corpus = [doc.split(" ") for doc in corpus]

# bm25 = BM25Okapi(tokenized_corpus)

# query = "windy London"
# tokenized_query = query.split(" ")

# doc_scores = bm25.get_scores(tokenized_query)
# print(doc_scores)
# ans = bm25.get_top_n(tokenized_query, corpus, n=2)
# print(ans)

import jieba
from tqdm import tqdm

questions = [
    "今天天气如何",
    "明天会下雨",
    "后天会出太阳吗",   
]
tokenized_corpus = [list(jieba.cut(q)) for q in tqdm(questions)]
print(tokenized_corpus)