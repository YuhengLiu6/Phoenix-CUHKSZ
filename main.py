import json
import os

import numpy as np
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
from tqdm import tqdm


from sentence_transformers import SentenceTransformer
embedding_model2 = SentenceTransformer('all-MiniLM-L6-v2')

import gradio as gr


openai.api_key = "EMPTY"
openai.api_base = "http://10.26.1.135:8888/v1"

prompt = '请根据给定的信息，来回答用户的问题，不仅要回答用户问题，还要尽可能多的提供相关信息。\n' \
         '如果你无法根据给定的信息来回答用户的问题，可以忽略给定的信息，自由回答。\n' \
        '如果给定的信息与事实情况不相符合，可以忽略给定的信息，自由回答。\n' \
        '回答时请用与问题相同的语言。\n\n' \
        '如果给定的信息中有url，则显示出来。\n\n' \
        '以下是给定的信息：\n{}\n\n' \
        '以下是用户的问题：\n{}'
 


class Retriever(object):
    def __init__(self, knowledge_dir='documents', k=1):
        self.k = k
        self.knowledge_dir = knowledge_dir
        self.questions, self.answers = self._read_files()
        self.embeddings = self._embed_questions()
        # self.embeddings = np.load("embeddings_1.npy")
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

    # def _embed_questions(self):
    #     embeddings = np.array([get_embedding(q, engine="text-embedding-ada-002") for q in tqdm(self.questions)])
    #     with open("embeddings_1.npy", "wb") as f:
    #         np.save(f, embeddings)
    #     return embeddings

    #更换模型2
    def _embed_questions(self):
        embeddings = np.array([embedding_model2.encode(q) for q in tqdm(self.questions)])
        with open("embeddings_2.npy", "wb") as f:
            np.save(f, embeddings)
        return embeddings

    def _find_similar_qas(self, question):
        # query_embedding = np.array(get_embedding(question, engine="text-embedding-ada-002"))
        query_embedding = np.array(embedding_model2.encode(question))
        sim = cosine_similarity(query_embedding, self.embeddings.T)
        topk = sim.argsort()[::-1][:self.k]

        qas = [f"Q: {self.questions[idx]} A: {self.answers[idx]}" for i, idx in enumerate(topk)]
        return qas

    def answer_question(self, question, verbose=False):
        question = self.modify_text(question)
        qas = self._find_similar_qas(question)
        qas = "\n".join(qas)
        input = prompt.format(qas, question)
        output = openai.ChatCompletion.create(
            model="text-embedding-ada-002",
            messages=[{"role": "user", "content": input}],
            max_tokens=500,
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
    # Q = input("请输入您的问题：")
    # ans = retriever.answer_question(Q, verbose=True)

    demo = gr.Interface(
        fn=retriever.answer_question,
        inputs=["text", "checkbox"],
        outputs=["text", "text"],
        title="香港中文大学（深圳）---- 问答系统",
    )
    demo.launch()

    
    # input_text = input("请输入一段中文文本：")
    # modified1_text = retriever.check_question_mark(input_text)
    # final_text = retriever.add_brackets(modified1_text)
    # print("修改后的文本：", final_text)

    



