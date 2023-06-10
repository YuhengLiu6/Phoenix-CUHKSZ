import json
import os

import numpy as np
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
from tqdm import tqdm
import gradio as gr



from rank_bm25 import BM25Okapi
import jieba


openai.api_key = "EMPTY"
openai.api_base = "http://10.26.1.135:8888/v1"

prompt = '请根据给定的信息，来回答用户的问题，不仅要回答用户问题，还要尽可能多的提供相关信息。\n' \
         '如果你无法根据给定的信息来回答用户的问题，可以忽略给定的信息，自由回答。\n' \
        '如果给定的信息与事实情况不相符合，可以忽略给定的信息，自由回答。\n' \
        '回答时请用与问题相同的语言。\n\n' \
        '回答的字数限制在200个词以内。\n\n' \
        '以下是给定的信息：\n{}\n\n' \
        '以下是用户的问题：\n{}'
 


class Retriever(object):
    def __init__(self, knowledge_dir='documents', k=1):
        self.k = k
        self.knowledge_dir = knowledge_dir
        self.questions, self.answers = self._read_files()
        self.embeddings = self._embed_questions()
        # self.embeddings = np.load("embeddings_2.npy")

    def _read_files(self):
        questions, answers = [], []
        for root, dirs, files in os.walk(self.knowledge_dir):
            for filename in files:
                if filename.endswith('.json'):
                    filepath = os.path.join(root, filename)
                    data = json.load(open(filepath))
                    for sample in data:
                        questions.append(sample["question"])
                        answers.append(sample["answer"])
        return questions, answers

    # #更换词向量模型3--bm25
    def _embed_questions(self):
        tokenized_corpus = [list(jieba.cut(q)) for q in tqdm(self.questions)]
        bm25 = BM25Okapi(tokenized_corpus)
        return bm25

    def _find_similar_qas(self, question):
        tokenized_query = list(jieba.cut(question))
        doc_scores = self.embeddings.get_scores(tokenized_query)
        topk = self.embeddings.get_top_n(tokenized_query, self.questions, n=1)

        qas = [f"Q: {topk} A: {self.answers[self.questions.index(topk)]}" for topk in topk]
        print("qas:", qas)
        return qas

    def answer_question(self, question, verbose=False):
        question = self.modify_text(question)
        qas = self._find_similar_qas(question)
        qas = "\n".join(qas)
        input = prompt.format(qas, question)
        #限制input的长度
        if len(input) > 2048:
            input = input[:2048]
        output = openai.ChatCompletion.create(
            model="text-embedding-ada-002",
            messages=[{"role": "user", "content": input}],
            max_tokens=2048,
        )
        answer = output.choices[0].message.content
        if verbose:
            print(f"[The question]\n{question}")
            print(f"[The retrieved QAs]\n{qas}")
            print(f"[The answer]\n{answer}")
        retrieved_qas = qas.split("\n")
        return answer, retrieved_qas
    
    #将输入的文本修改为以问号结尾
    def check_question_mark(self, text):
        if len(text) > 0:
            last_char = text[-1]
            if last_char != '？':
                # 结尾不是问号，进行修改
                if last_char == '。' or last_char == '！':
                    # 删除句号或感叹号，并添加问号
                    text = text[:-1] + '？'
                else:
                    # 添加问号
                    text += '？'
        else:
            # 输入为空，返回空字符串
            text = ''
        return text
    
    def add_brackets(self, text):
        keyword = "深圳"
        if keyword in text:
            index = text.find(keyword)
            if index > 0 and text[index - 1] != "（" and text[index - 1] != "(":
                text = text[:index] + "（" + text[index:]
            if index + len(keyword) < len(text) and text[index + len(keyword)] != "）" and text[index + len(keyword)] != ")":
                text = text[:index + len(keyword) + 1] + "）" + text[index + len(keyword)+1:]
        return text

    def modify_text(self, text):
        modified_text = self.check_question_mark(text)
        final_text = self.add_brackets(modified_text)
        return final_text








if __name__ == '__main__':
    retriever = Retriever(knowledge_dir="data/data_20230609")
    Q = input("请输入您的问题：")
    ans = retriever.answer_question(Q, verbose=True)

    # demo = gr.Interface(
    #     fn=retriever.answer_question,
    #     inputs=["text", "checkbox"],
    #     outputs=["text", "text"],
    #     title="香港中文大学（深圳）---- 问答系统",
    # )
    # demo.launch()

    # with open("questions.md", "r", encoding="utf-8") as w:

    # input_text = input("请输入一段中文文本：")
    # modified1_text = retriever.check_question_mark(input_text)
    # final_text = retriever.add_brackets(modified1_text)
    # print("修改后的文本：", final_text)

    



